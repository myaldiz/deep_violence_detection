from model import *
from modules import set_placeholders, set_queue, create_graph
from modules import create_training_op, start_queue_threads
from preprocess import *
from default_settings import model_settings


def show_running_info(model_settings, duration, step, loss, accuracy):
    batch_size, num_gpu = model_settings['batch_size'], model_settings['num_gpu']
    num_examples_per_step = model_settings['total_batch']
    examples_per_sec = num_examples_per_step / duration
    sec_per_batch = duration / num_gpu
    format_str = ('%s: step %d, (%.1f examples/sec; %.3f'
                  'sec/batch), (accuracy: %f loss: %f)')
    format_tuple = (datetime.now(), step,
                    examples_per_sec, sec_per_batch, accuracy, loss)
    print(format_str % format_tuple)


def run_training(sess, model_settings):
    model_settings['optimizer'] = tf.train.AdamOptimizer(model_settings['learning_rate'])

    # Set placeholders, queue operations and thread coordinator
    set_placeholders(model_settings)
    set_queue(model_settings)

    # Create Graph
    create_graph(model_settings)
    create_training_op(model_settings)

    if (model_settings['fc_pretrained'] == True):
        for fc_variable in tf.get_collection('fc_layer'):
            tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, fc_variable)

    # Read training file locations and shuffle
    train_dir_locations = model_settings['train_test_loc'] + \
                          model_settings['train_file_name']
    dir_clips, label_clips = get_data_dir(train_dir_locations)
    model_settings['train_list'] = shuffle_list(dir_clips, label_clips)

    # Initialize file thread Coordinator and Start input reading threads
    model_settings['coord'] = tf.train.Coordinator()
    read_threads = start_queue_threads(sess, model_settings)

    # Save Only Model Variables
    saver = tf.train.Saver(tf.model_variables())
    model_settings['saver'] = saver

    # Initialize variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore saved model variables
    if (model_settings['read_pretrained_model'] == True):
        saver.restore(sess, model_settings['model_read_dir'])

    # Tensorboard summary writers
    summary_writer = tf.summary.FileWriter(model_settings['checkpoint_dir'])
    summary_writer.add_graph(sess.graph)
    model_settings['summary_writer'] = summary_writer

    train_op, summary_op = model_settings['train_op'], model_settings['summary_op']

    # Single tower loss implementation for now
    loss_op, acc_op = model_settings['tower_mean_loss'], model_settings['tower_mean_accuracy']

    # time.sleep(15)

    print('Training begins:')
    for step in range(model_settings['max_steps']):
        start_time = time.time()
        # print(sess.run(model_settings['queue'].size()))

        if (step + 1) % model_settings['checkpoints'] != 0:
            _, tower_mean_loss, tower_mean_acc = sess.run([train_op, loss_op, acc_op])
        else:
            _, summary_str, tower_mean_loss, tower_mean_acc = sess.run([train_op, summary_op, loss_op, acc_op])
            summary_writer.add_summary(summary_str, step)

        show_running_info(model_settings, time.time() - start_time,
                          step, tower_mean_loss, tower_mean_acc)
    stop_threads(sess, read_threads, model_settings)


def save_graph(sess, model_settings):
    model_settings['saver'].save(sess,
                                 model_settings['model_save_dir'],
                                 write_meta_graph=False)


if __name__ == '__main__':
    # run_training()
    pass
