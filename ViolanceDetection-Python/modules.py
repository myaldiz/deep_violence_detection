import tensorflow as tf
import threading
from model import model, tower_loss, tower_accuracy
from preprocess import read_clips_from_video, shuffle_list
from preprocess import get_frames_data, process_frames
import os

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)

            # TODO: fix the tower gradient mean into weighted mean
            # for multi tower implementation
            grad = tf.reduce_mean(grad, 0)
            # grad = tf.metrics.mean(grad, weights=[])

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads


def create_graph(model_settings):
    queue = model_settings['queue']
    tower_accuracies = []
    tower_losses = []
    with tf.variable_scope(tf.get_variable_scope()):
        enum_val = enumerate(zip(model_settings['devices_to_run'], model_settings['batch_sizes']))
        for device_index, (device_name, batch_size) in enum_val:
            with tf.name_scope('Tower_%d' % device_index) as scope:
                # Input prep for the tower
                if model_settings['input_from_placeholders']:
                    feed_input = [model_settings['images_placeholder']]
                    feed_label = [model_settings['labels_placeholder']]
                else:
                    if model_settings['dequeue_immediately']:
                        read_batch_size = tf.math.minimum(queue.size(), batch_size)
                        # Network will read at least one example
                        read_batch_size = tf.math.maximum(1, read_batch_size)
                    else:
                        read_batch_size = tf.convert_to_tensor(batch_size)

                    model_settings['read_batch_size'] = read_batch_size
                    feed_input, feed_label = queue.dequeue_many(read_batch_size)

                with tf.device(device_name):
                    model_out = model(feed_input, model_settings)

                    loss = tower_loss(scope, model_out, feed_label)
                    accuracy = tower_accuracy(model_out, feed_label)
                    tower_losses.append(loss)
                    tower_accuracies.append(accuracy)

                    tf.get_variable_scope().reuse_variables()

    tower_mean_loss = tf.reduce_mean(tower_losses)
    tower_mean_accuracy = tf.reduce_mean(tower_accuracies)
    features_stacked = tf.concat(model_settings['clip_features'], 0)

    tf.summary.scalar('Total_Loss', tower_mean_loss)
    tf.summary.scalar('Top1_Correct_Predictions', tower_mean_accuracy)

    model_settings['tower_mean_loss'] = tower_mean_loss
    model_settings['tower_mean_accuracy'] = tower_mean_accuracy
    model_settings['tower_losses'] = tower_losses
    model_settings['stacked_features'] = features_stacked


def create_training_op(model_settings):
    opt = model_settings['optimizer']
    tower_losses = model_settings['tower_losses']
    global_step = model_settings['global_step']

    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
        enum_val = enumerate(zip(model_settings['devices_to_run'], model_settings['batch_sizes']))
        for device_index, (device_name, batch_size) in enum_val:
            with tf.device(device_name):
                with tf.name_scope('Tower_%d' % device_index) as scope:
                    loss = tower_losses[device_index]
                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)
    grads = average_gradients(tower_grads)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Learning rate decay through learning rate only, using
    # variable_averages = tf.train.ExponentialMovingAverage(model_settings['moving_decay'], global_step)
    # variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # train_op = tf.group(apply_gradient_op, variable_averages_op)
    train_op = apply_gradient_op

    summary_op = tf.summary.merge_all()

    model_settings['train_op'], model_settings['summary_op'] = train_op, summary_op


def set_placeholders(model_settings):
    if model_settings['read_from_frames']:
        shape = (None, None, None, 3)
        images_placeholder = tf.placeholder(tf.float32, shape=shape, name="input_clip")
    else:
        images_placeholder = tf.placeholder(tf.float32, shape=model_settings['input_shape'], name="input_clip")

    labels_placeholder = tf.placeholder(tf.int64, shape=(), name="labels")
    dropout_placeholder = tf.placeholder_with_default(model_settings['dropout'], shape=())

    model_settings['images_placeholder'] = images_placeholder
    model_settings['labels_placeholder'] = labels_placeholder
    model_settings['dropout_placeholder'] = dropout_placeholder
    print('Finished setting placeholders..')


def set_queue(model_settings):
    clips = model_settings['images_placeholder']
    labels_placeholder = model_settings['labels_placeholder']

    queue = tf.FIFOQueue(model_settings['queue_size'],
                         [tf.float32, tf.int64],
                         shapes=[model_settings['input_shape'],
                                 labels_placeholder.shape],
                         name='Input_Queue')
    # Process frames through tensorflow
    if model_settings['read_from_frames']:
        clips = process_frames(model_settings)

    enqueue_op = queue.enqueue([clips,
                                labels_placeholder],
                               name='Enqueue_Operation')

    model_settings['queue'], model_settings['enqueue_op'] = queue, enqueue_op
    print('Finished setting queue..')


# Read single clip at a time
def load_and_enqueue(sess, model_settings, thread_index):
    coord, enqueue_op = model_settings['coord'], model_settings['enqueue_op']
    dir_videos, label_clips = model_settings['input_list']
    num_thread = model_settings['num_thread']
    images_placeholder = model_settings['images_placeholder']
    labels_placeholder = model_settings['labels_placeholder']
    data_size = len(dir_videos)

    read_index = thread_index
    for read_index in range(thread_index, data_size, num_thread):
        if coord.should_stop():
            # print('Stop_requested: %d' % thread_index)
            break

        video_dir, label = dir_videos[read_index], label_clips[read_index]
        video_dir = model_settings['data_home'] + video_dir

        if model_settings['read_from_frames']:
            input_clip = get_frames_data(video_dir, model_settings)
        else:
            input_clip = read_clips_from_video(video_dir, model_settings)

        sess.run(enqueue_op, feed_dict={images_placeholder: input_clip,
                                        labels_placeholder: label})


def queue_thread_runner(sess, model_settings):
    coord = model_settings['coord']
    queue = model_settings['queue']

    while not coord.should_stop():
        # Shuffle the list
        if not model_settings['is_testing']:
            model_settings['input_list'] = shuffle_list(*model_settings['input_list'])

        # Create threads to enqueue and start them
        t = []
        for i in range(model_settings['num_thread']):
            t.append(threading.Thread(target=load_and_enqueue,
                                      args=(sess, model_settings, i)))
            t[i].start()

        # Wait threads to finish
        coord.join(t)
        # if in testing mode
        if model_settings['is_testing']:
            break

    print('Threads stopped!')


def start_queue_threads(sess, model_settings):
    runner_thread = threading.Thread(target=queue_thread_runner,
                                     args=(sess, model_settings))
    runner_thread.start()


def stop_thread_runner(sess, model_settings):
    coord = model_settings['coord']
    queue = model_settings['queue']
    coord.request_stop()

    print('Emptying queues..')
    for i in range(model_settings['num_thread']):
        queue_size = sess.run(queue.size())
        if queue_size < model_settings['total_batch']:
            break
        sess.run(queue.dequeue_many(model_settings['total_batch']))


def make_dirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def time2string(t):
    out_str = str(t).split(' ')
    interm = '-'.join(out_str[1].split(':')).split('.')
    out_str[1] = interm[0]
    # out_str.append(interm[1])
    return '__'.join(out_str)
