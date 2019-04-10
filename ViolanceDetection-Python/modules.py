import tensorflow as tf
import threading
from model import model, tower_loss, tower_accuracy
from preprocess import read_clip


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
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads


def create_graph(model_settings):
    queue = model_settings['queue']
    tower_accuracy = []
    tower_loss = []
    with tf.variable_scope(tf.get_variable_scope()):
        for gpu_index in range(model_settings['num_gpu']):
            with tf.device('/gpu:%d' % gpu_index):
                with tf.name_scope('Tower_%d' % gpu_index) as scope:
                    ind_begin, ind_end = (gpu_index * model_settings['batch_size'],
                                          (gpu_index + 1) * model_settings['batch_size'])

                    feed_input, feed_label = queue.dequeue_many(model_settings['batch_size'])
                    model_out = model(feed_input, model_settings)

                    # TODO: Pass loss through model settings
                    loss = tower_loss(scope, model_out, feed_label)
                    accuracy = tower_accuracy(model_out, feed_label)
                    tower_loss.append(loss)
                    tower_accuracy.append(accuracy)

                    tf.get_variable_scope().reuse_variables()

    tower_mean_loss = tf.reduce_mean(tower_loss)
    tower_mean_accuracy = tf.reduce_mean(tower_accuracy)
    tf.summary.scalar('Total_Loss', tower_mean_loss)
    tf.summary.scalar('Top1_Correct_Predictions', tower_mean_accuracy)

    model_settings['tower_mean_loss'] = tower_mean_loss
    model_settings['tower_mean_accuracy'] = tower_mean_accuracy
    model_settings['tower_loss'] = tower_loss


def create_training_op(model_settings):
    global_step = tf.get_variable('Global_Step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    opt, tower_loss = model_settings['optimizer'], model_settings['tower_loss']

    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
        for gpu_index in range(model_settings['num_gpu']):
            with tf.device('/gpu:%d' % gpu_index):
                with tf.name_scope('Tower_%d' % gpu_index) as scope:
                    loss = tower_loss[gpu_index]
                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)
    grads = average_gradients(tower_grads)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    variable_averages = tf.train.ExponentialMovingAverage(model_settings['moving_decay'], global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    train_op = tf.group(apply_gradient_op, variable_averages_op)

    summary_op = tf.summary.merge_all()

    model_settings['train_op'], model_settings['summary_op'] = train_op, summary_op


def set_placeholders(model_settings):
    total_batch = model_settings['total_batch']
    images_placeholder = tf.placeholder(tf.float32, shape=model_settings['input_shape'], name="input_clip")
    labels_placeholder = tf.placeholder(tf.int64, shape=(), name="labels")
    dropout_placeholder = tf.placeholder_with_default(model_settings['dropout'], shape=())
    # tf.summary.image('input_batch_sample_image', images_placeholder[:,0], total_batch)

    model_settings['images_placeholder'] = images_placeholder
    model_settings['labels_placeholder'] = labels_placeholder
    model_settings['dropout_placeholder'] = dropout_placeholder
    return


def set_queue(model_settings):
    images_placeholder = model_settings['images_placeholder']
    labels_placeholder = model_settings['labels_placeholder']

    queue = tf.FIFOQueue(model_settings['queue_size'],
                         [tf.float32, tf.int64],
                         shapes=[model_settings['input_shape'],
                                 labels_placeholder.shape],
                         name='Input_Queue')

    enqueue_op = queue.enqueue([model_settings['images_placeholder'],
                                model_settings['labels_placeholder']],
                               name='Enqueue_Operation')

    model_settings['queue'], model_settings['enqueue_op'] = queue, enqueue_op


# Read single clip at a time
def load_and_enqueue(sess, model_settings, thread_index):
    coord, enqueue_op = model_settings['coord'], model_settings['enqueue_op']
    dir_videos, label_clips = model_settings['train_list']
    num_thread = model_settings['num_thread']
    images_placeholder = model_settings['images_placeholder']
    labels_placeholder = model_settings['labels_placeholder']
    data_size = len(dir_videos)

    read_index = thread_index
    while not coord.should_stop():
        video_dir, label = dir_videos[read_index], label_clips[read_index]
        video_dir = model_settings['data_home'] + video_dir
        input_clip = read_clip(video_dir, model_settings)

        # if input clip is not empty
        if (input_clip.shape == model_settings['input_shape']):
            sess.run(enqueue_op, feed_dict={images_placeholder: input_clip,
                                            labels_placeholder: label})
        read_index = (read_index + num_thread) % data_size
    # print('Stop_requested: %d' % thread_index)


def start_queue_threads(sess, model_settings):
    t = []
    for i in range(model_settings['num_thread']):
        t.append(threading.Thread(target=load_and_enqueue, args=(sess, model_settings, i)))
        t[i].start()
    return t


def stop_threads(sess, threads, model_settings):
    coord = model_settings['coord']
    model_settings['coord'].request_stop()
    queue = model_settings['queue']
    for i in range(model_settings['num_thread']):
        if (sess.run(queue.size()) < model_settings['total_batch']):
            break
        sess.run(queue.dequeue_many(model_settings['batch_size']))
    model_settings['coord'].join(threads)
    print('Finished')
