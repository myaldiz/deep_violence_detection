from modules import set_placeholders, set_queue, create_graph
from modules import start_queue_threads
from preprocess import *
from default_settings import model_settings, set_model_settings
from datetime import datetime

from sklearn import svm
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics


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
    model_settings['input_list'] = shuffle_list(*model_settings['input_list'])

    # Initialize file thread Coordinator and Start input reading threads
    model_settings['coord'] = tf.train.Coordinator()
    start_queue_threads(sess, model_settings)

    # Create Graph
    create_graph(model_settings)

    # Save Only Model Variables
    loader = tf.train.Saver(tf.get_collection('all_variables'))

    # Initialize variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore saved model variables
    if model_settings['read_pretrained_model']:
        loader.restore(sess, model_settings['model_read_loc'])
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


def extract_features(model_settings, sess):
    model_settings['is_testing'] = True
    model_settings['read_from_frames'] = False
    model_settings['data_home'] = model_settings['feature_data_home']
    set_model_settings(model_settings)

    set_placeholders(model_settings)
    set_queue(model_settings)

    # Read training file locations
    test_dir_locations = model_settings['feature_loc'] + \
                         model_settings['feature_extract_file_name']
    model_settings['input_list'] = get_data_dir(test_dir_locations,
                                                model_settings)


    # Initialize file thread Coordinator and Start input reading threads
    model_settings['coord'] = tf.train.Coordinator()
    start_queue_threads(sess, model_settings)

    # Create Graph
    create_graph(model_settings)

    # Save Only Model Variables
    loader = tf.train.Saver(tf.get_collection('all_variables'))

    # Initialize variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore saved model variables
    if model_settings['read_pretrained_model']:
        loader.restore(sess, model_settings['model_read_loc'])
        print('Read the models successfully..')

    features_op = model_settings['stacked_features']
    read_batch_size = model_settings['read_batch_size']

    data_size = len(model_settings['input_list'][0])

    print('Total testing examples:', data_size)
    print('Feature extraction begins..')
    features = []
    batch_size = []
    total_num_tested_data = 0

    while total_num_tested_data != data_size:
        cur_features, cur_read_batch = sess.run([features_op, read_batch_size])
        total_num_tested_data += cur_read_batch

        features.append(cur_features)
        batch_size.append(cur_read_batch)
        print('%d/%d finished' % (total_num_tested_data, data_size))

    print('----------------------------------------------------------------')
    print('Feature extraction finished,')
    features = np.concatenate(features, axis=0)
    return features


def train_svm(X_data, model_settings):
    y_data = model_settings['input_list'][1]
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
    classifier = svm.LinearSVC()
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print('Accuracy: ', score)
    return classifier


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


def main2():
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        features = extract_features(model_settings, sess)

    print(features.shape)
    classifier = train_svm(features, model_settings)


if __name__ == '__main__':
    main2()
