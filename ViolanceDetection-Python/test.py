import tensorflow as tf
from modules import set_placeholders, set_queue, create_graph
from modules import start_queue_threads
from preprocess import *
from default_settings import model_settings, set_model_settings
from datetime import datetime


# def run_testing(sess, model_settings):
#     # Convert to multi-tower setting
#     print('Testing begins:')
#     loss_op, acc_op = model_settings['tower_loss'][0], model_settings['tower_accuracy'][0]
#     np_mean = np.load('crop_mean.npy').reshape([model_settings['frames_per_clip'],
#                                                 model_settings['crop_size'],
#                                                 model_settings['crop_size'],
#                                                 model_settings['channels']])
#
#     data_index = 0
#     epoch_finished = False
#     batch_accuracy = []
#     batch_loss = []
#     while not epoch_finished:
#         start_time = time.time()
#
#         test_images, test_labels, data_index, epoch_finished = read_clip_label_sequentially(
#             dir_clips, label_clips, np_mean, data_index,
#             batch_size=model_settings['batch_size'] * model_settings['num_gpu'],
#             num_frames_per_clip=model_settings['frames_per_clip'],
#             crop_size=model_settings['crop_size'])
#
#         img_process_duration = time.time() - start_time
#
#         feed_dict = {model_settings['images_placeholder']: test_images,
#                      model_settings['labels_placeholder']: test_labels,
#                      model_settings['dropout_placeholder']: 1.0}
#
#         accuracy, loss = sess.run([acc_op, loss_op], feed_dict=feed_dict)
#
#         batch_accuracy.append(accuracy)
#         batch_loss.append(loss)
#         print('Data index :%d' % data_index)
#
#     test_accuracy = np.mean(np.array(batch_accuracy))
#     test_loss = np.mean(np.array(batch_loss))
#     print('Test accuracy %f, test loss %f!' % (test_accuracy, test_loss))
#
#
#
#
#
# def calculate_features(sess, model_settings):
#     # Convert to multi-tower setting
#     feature_extract_op = model_settings['clip_features']
#     np_mean = np.load('crop_mean.npy').reshape([model_settings['frames_per_clip'],
#                                                 model_settings['crop_size'],
#                                                 model_settings['crop_size'],
#                                                 model_settings['channels']])
#
#     dir_clips, label_clips = get_UCF101_dir(model_settings['feature_file_loc'])
#     dir_clips, label_clips = shuffle_list(dir_clips, label_clips)
#
#     data_index = 0
#     epoch_finished = False
#     x_data = []
#     y_data = []
#     data_scanned = 0
#     while (not epoch_finished):
#         # start_time = time.time()
#
#         test_images, test_labels, data_index, epoch_finished = read_clip_label_sequentially(
#             dir_clips, label_clips, np_mean, data_index,
#             batch_size=model_settings['batch_size'] * model_settings['num_gpu'],
#             num_frames_per_clip=model_settings['frames_per_clip'],
#             crop_size=model_settings['crop_size'])
#         data_scanned = data_scanned + model_settings['batch_size']
#         # img_process_duration = time.time() - start_time
#
#         feed_dict = {model_settings['images_placeholder']: test_images,
#                      model_settings['labels_placeholder']: test_labels,
#                      model_settings['dropout_placeholder']: 1.0}
#
#         x_data.append(sess.run(feature_extract_op, feed_dict=feed_dict))
#         y_data.append(test_labels)
#
#         print('Data index :%d' % data_index)
#     X_data = np.array(x_data)
#     Y_data = np.array(y_data)
#     X_data = X_data.reshape([data_scanned, 4096])
#     Y_data = Y_data.reshape([data_scanned])
#     return X_data, Y_data


def run_testing(model_settings, sess):
    model_settings['is_testing'] = True
    set_model_settings(model_settings)

    set_placeholders(model_settings)
    set_queue(model_settings)

    # Read training file locations
    test_dir_locations = model_settings['train_test_loc'] + \
                         model_settings['test_file_name']
    model_settings['input_list'] = get_data_dir(test_dir_locations,
                                                model_settings)

    # Initialize file thread Coordinator and Start input reading threads
    model_settings['coord'] = tf.train.Coordinator()
    start_queue_threads(sess, model_settings)

    # Create Graph
    create_graph(model_settings)

    # Save Only Model Variables
    saver = tf.train.Saver(tf.get_collection('all_variables'))

    # Initialize variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore saved model variables
    if model_settings['read_pretrained_model']:
        saver.restore(sess, model_settings['model_read_loc'])
        print('Read the models successfully..')

    loss_op = model_settings['tower_mean_loss']
    acc_op = model_settings['tower_mean_accuracy']
    read_batch_size = model_settings['read_batch_size']

    data_size = len(model_settings['input_list'][0])

    print('Total testing examples:', data_size)
    print('Testing begins..')
    batch_accuracy = []
    batch_loss = []
    batch_size = []
    total_num_tested_data = 0

    while total_num_tested_data != data_size:
        tower_mean_loss, tower_mean_acc, \
        cur_read_batch = sess.run([loss_op, acc_op, read_batch_size])
        total_num_tested_data += cur_read_batch

        batch_accuracy.append(tower_mean_acc)
        batch_loss.append(tower_mean_loss)
        batch_size.append(cur_read_batch)

        show_running_info(model_settings, batch_accuracy,
                          batch_loss, batch_size, data_size)

    print('----------------------------------------------------------------')
    print('Testing finished,')
    print('')
    print('Accuracy:', np.average(batch_accuracy, weights=batch_size))


def show_running_info(model_settings, batch_accuracy, batch_loss, batch_size, data_size):
    start_time = model_settings['start_time']
    time_spent = datetime.now() - start_time
    total_examples_read = np.sum(batch_size)
    percentage_read = (total_examples_read / data_size)
    total_time = time_spent / percentage_read
    time_left = total_time - time_spent
    average_accuracy = np.average(batch_accuracy, weights=batch_size)
    average_loss = np.average(batch_loss, weights=batch_size)

    format_str = (
        'Step: %d/%d, Percentage Finished: %.3f, Time left: %s'
        ' - Cur accuracy: %.2f, Cur loss: %.4f -||- '
        'Average accuracy: %.3f, Average loss: %.4f -'
    )
    format_tuple = (
        total_examples_read, data_size, percentage_read,
        str(time_left)[:12], batch_accuracy[-1],
        batch_loss[-1], average_accuracy, average_loss
    )
    print(format_str % format_tuple)


# TODO:
# - Implement Extract Features method
# - Implement classification of the video feed
#   - Showing top 3 classification
# Fix multiple device support

# To run at the background
# nohup python test.py &> test_out.txt &
# kill -2 [pid]
def main():
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        # with tf.Session() as sess:
        run_testing(model_settings, sess)


if __name__ == '__main__':
    main()
