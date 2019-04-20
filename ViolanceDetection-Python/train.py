import tensorflow as tf
from modules import set_placeholders, set_queue, create_graph
from modules import create_training_op, start_queue_threads
from modules import stop_thread_runner
from preprocess import *
from default_settings import model_settings, set_model_settings
from datetime import datetime


def show_running_info(model_settings, duration, step, loss, accuracy):
    batch_size, num_gpu = model_settings['batch_size'], model_settings['num_gpu']
    num_examples_per_step = model_settings['total_batch']
    epoch = model_settings['current_epoch']
    examples_per_sec = num_examples_per_step / duration
    sec_per_batch = duration / num_gpu
    format_str = ('Epoch: %d, %s: step %d, (%.1f examples/sec; %.3f'
                  'sec/batch), (accuracy: %f loss: %f)')
    format_tuple = (epoch, datetime.now(), step,
                    examples_per_sec, sec_per_batch, accuracy, loss)
    print(format_str % format_tuple)


def run_training(model_settings, sess):
    model_settings['optimizer'] = tf.train.AdamOptimizer(model_settings['learning_rate'])

    set_model_settings(model_settings)
    # Set placeholders, queue operations and thread coordinator
    set_placeholders(model_settings)
    set_queue(model_settings)

    # Read training file locations
    train_dir_locations = model_settings['train_test_loc'] + \
                          model_settings['train_file_name']
    model_settings['input_list'] = get_data_dir(train_dir_locations)

    # Initialize file thread Coordinator and Start input reading threads
    model_settings['coord'] = tf.train.Coordinator()
    start_queue_threads(sess, model_settings)

    # Create Graph
    create_graph(model_settings)
    create_training_op(model_settings)

    # Save Only Model Variables
    saver = tf.train.Saver(tf.model_variables())
    model_settings['saver'] = saver

    # Initialize variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore saved model variables
    if model_settings['read_pretrained_model']:
        saver.restore(sess, model_settings['model_read_dir'])
        print('Read the models successfully..')

    # check last variable restored
    # var = sess.run(tf.model_variables()[-1])
    # print(tf.model_variables()[-1], var[:10])

    # Tensorboard summary writers
    summary_writer = tf.summary.FileWriter(model_settings['checkpoint_dir'])
    summary_writer.add_graph(sess.graph)
    model_settings['summary_writer'] = summary_writer

    train_op, summary_op = model_settings['train_op'], model_settings['summary_op']
    loss_op, acc_op = model_settings['tower_mean_loss'], model_settings['tower_mean_accuracy']

    print('Training begins:')
    for step in range(model_settings['max_steps']):
        start_time = time.time()
        # Number of data enqueued
        # print(sess.run(model_settings['queue'].size()))

        if (step + 1) % model_settings['checkpoints'] != 0:
            _, tower_mean_loss, tower_mean_acc = sess.run([train_op, loss_op, acc_op])
        else:
            _, summary_str, tower_mean_loss, tower_mean_acc = \
                sess.run([train_op, summary_op, loss_op, acc_op])
            summary_writer.add_summary(summary_str, step)

        show_running_info(model_settings, time.time() - start_time,
                          step, tower_mean_loss, tower_mean_acc)

    # Stop the input queue threads
    stop_thread_runner(sess, model_settings)

    # Save the graph
    if model_settings['save_graph']:
        save_graph(model_settings, sess)


def save_graph(model_settings, sess):
    saver = tf.train.Saver(tf.get_collection('all_variables'))
    save_dir = model_settings['model_save_dir']
    saver.save(sess, save_dir, write_meta_graph=False)


def main():
    with tf.Session() as sess:
        run_training(model_settings, sess)


if __name__ == '__main__':
    main()
