from preprocess import *
from default_settings import *
from model import *

def run_testing(sess, model_settings):
    
    #Convert to multi-tower setting
    print('Testing begins:')
    loss_op, acc_op = model_settings['tower_loss'][0], model_settings['tower_accuracy'][0]
    np_mean = np.load('crop_mean.npy').reshape([model_settings['frames_per_clip'],
                                                model_settings['crop_size'],
                                                model_settings['crop_size'],
                                                model_settings['channels']])
    
    data_index = 0
    epoch_finished = False
    batch_accuracy = []
    batch_loss = []
    while(not epoch_finished):
        start_time = time.time()
        
        test_images, test_labels, data_index, epoch_finished = read_clip_label_sequentially(
            dir_clips, label_clips, np_mean, data_index,
            batch_size=model_settings['batch_size'] * model_settings['num_gpu'],
            num_frames_per_clip=model_settings['frames_per_clip'],
            crop_size= model_settings['crop_size'])
        
        img_process_duration = time.time() - start_time
        
        feed_dict = {model_settings['images_placeholder'] : test_images,
                    model_settings['labels_placeholder'] : test_labels,
                    model_settings['dropout_placeholder'] : 1.0}
        
        accuracy, loss = sess.run([acc_op, loss_op], feed_dict=feed_dict)
        
        batch_accuracy.append(accuracy)
        batch_loss.append(loss)
        print('Data index :%d' % data_index)
    
    test_accuracy = np.mean(np.array(batch_accuracy))
    test_loss = np.mean(np.array(batch_loss))
    print('Test accuracy %f, test loss %f!' % (test_accuracy, test_loss))


def calculate_features(sess, model_settings):
#Convert to multi-tower setting
    feature_extract_op = model_settings['clip_features']
    np_mean = np.load('crop_mean.npy').reshape([model_settings['frames_per_clip'],
                                                model_settings['crop_size'],
                                                model_settings['crop_size'],
                                                model_settings['channels']])
    
    dir_clips, label_clips = get_UCF101_dir(model_settings['feature_file_loc'])
    dir_clips, label_clips = shuffle_list(dir_clips, label_clips)
    
    data_index = 0
    epoch_finished = False
    x_data = []
    y_data = []
    data_scanned = 0
    while(not epoch_finished):
        #start_time = time.time()
        
        test_images, test_labels, data_index, epoch_finished = read_clip_label_sequentially(
            dir_clips, label_clips, np_mean, data_index,
            batch_size=model_settings['batch_size'] * model_settings['num_gpu'],
            num_frames_per_clip=model_settings['frames_per_clip'],
            crop_size= model_settings['crop_size'])
        data_scanned = data_scanned + model_settings['batch_size']
        #img_process_duration = time.time() - start_time
        
        feed_dict = {model_settings['images_placeholder'] : test_images,
                    model_settings['labels_placeholder'] : test_labels,
                    model_settings['dropout_placeholder'] : 1.0}
        
        x_data.append(sess.run(feature_extract_op, feed_dict=feed_dict))
        y_data.append(test_labels)
        
        print('Data index :%d' % data_index)
    X_data = np.array(x_data)
    Y_data = np.array(y_data)
    X_data = X_data.reshape([data_scanned, 4096])
    Y_data = Y_data.reshape([data_scanned])
    return X_data, Y_data


if __name__ == '__main__':
    pass