import sys, os
import tensorflow as tf
from modules import set_placeholders, set_queue, create_graph
from modules import create_training_op, start_queue_threads
from modules import stop_thread_runner
from preprocess import *
from default_settings import model_settings, set_model_settings
from datetime import datetime


def show_running_info(model_settings, duration, step, loss, accuracy):
    num_examples_per_step = model_settings['total_batch']
    epoch = model_settings['current_epoch']
    examples_per_sec = num_examples_per_step / duration
    format_str = ('Epoch: %d, %s: step %d, (%.1f examples/sec; %.3f'
                  'sec/batch), (accuracy: %f loss: %f)')
    format_tuple = (epoch, datetime.now(), step,
                    examples_per_sec, duration, accuracy, loss)
    print(format_str % format_tuple)


def make_dirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def time2string(t):
    out_str = str(t).split(' ')
    interm = '-'.join(out_str[1].split(':')).split('.')
    out_str[1] = interm[0]
    # out_str.append(interm[1])
    return '__'.join(out_str)


def save_graph(model_settings, sess):
    if model_settings['save_graph']:
        saver = model_settings['saver']
        save_dir = model_settings['model_save_dir']
        save_dir += time2string(model_settings['start_time']) + '/'
        make_dirs(save_dir)
        save_dir += model_settings['model_name']
        try:
            saver.save(sess, save_dir,
                       global_step=model_settings['global_step'],
                       write_meta_graph=False)
            print('Variables saved successfully..')
        except:
            print('Error occurred while saving the model: ', sys.exc_info()[0])


def set_summary_writers(model_settings, sess):
    summary_dir = model_settings['checkpoint_dir'] + model_settings['model_name']
    summary_dir += '/' + time2string(model_settings['start_time']) + '/'
    make_dirs(summary_dir)
    summary_writer = tf.summary.FileWriter(summary_dir)
    summary_writer.add_graph(sess.graph)
    return summary_writer


def run_training(model_settings, sess):
    model_settings['optimizer'] = tf.train.AdamOptimizer(model_settings['learning_rate'])

    set_model_settings(model_settings)
    # Set placeholders, queue operations and thread coordinator
    set_placeholders(model_settings)
    set_queue(model_settings)

    # Read training file locations
    train_dir_locations = model_settings['train_test_loc'] + \
                          model_settings['train_file_name']
    input_list = get_data_dir(train_dir_locations,
                              model_settings)
    data_size = len(input_list[0])
    model_settings['input_list'] = input_list

    # Initialize file thread Coordinator and Start input reading threads
    model_settings['coord'] = tf.train.Coordinator()
    start_queue_threads(sess, model_settings)

    # Create Graph
    create_graph(model_settings)
    create_training_op(model_settings)

    # Model Load&Saver
    loader = tf.train.Saver(tf.model_variables())
    saver = tf.train.Saver(tf.get_collection('all_variables'))
    model_settings['saver'] = saver

    # Initialize variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore saved model variables
    if model_settings['read_pretrained_model']:
        loader.restore(sess, model_settings['model_read_loc'])
        print('Read the models successfully..')

    # check last variable restored
    # var = sess.run(tf.get_collection('all_variables')[-1])
    # print(tf.get_collection('all_variables')[-1], var[:10])

    # Set Tensorboard summary writers
    summary_writer = set_summary_writers(model_settings, sess)

    train_op, summary_op = model_settings['train_op'], model_settings['summary_op']
    loss_op, acc_op = model_settings['tower_mean_loss'], model_settings['tower_mean_accuracy']

    print('Training data size: ', data_size)
    print('Training begins:')
    for step in range(model_settings['max_steps']):
        start_time = time.time()
        model_settings['global_step'] = step
        # Number of data enqueued
        # print(sess.run(model_settings['queue'].size()))

        if (step + 1) % model_settings['summary_checkpoints'] != 0:
            _, tower_mean_loss, tower_mean_acc = sess.run([train_op, loss_op, acc_op])
        else:
            _, summary_str, tower_mean_loss, tower_mean_acc = \
                sess.run([train_op, summary_op, loss_op, acc_op])
            summary_writer.add_summary(summary_str, step)

        # Graph saving checkpoints
        if (step + 1) % model_settings['model_save_checkpoints'] == 0:
            save_graph(model_settings, sess)

        show_running_info(model_settings, time.time() - start_time,
                          step, tower_mean_loss, tower_mean_acc)

        model_settings['current_epoch'] = (step * model_settings['total_batch']) \
                                          // data_size

        if model_settings['current_epoch'] == model_settings['max_epoch']:
            break

    # Stop the input queue threads
    stop_thread_runner(sess, model_settings)

    # Save the graph
    save_graph(model_settings, sess)


def main():
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        # with tf.Session() as sess:
        try:
            run_training(model_settings, sess)
        except KeyboardInterrupt:
            print('Keyboard interrupt occurred')
            stop_thread_runner(sess, model_settings)
            print('Stopping request sent to threads..')
            save_graph(model_settings, sess)


if __name__ == '__main__':
    main()
