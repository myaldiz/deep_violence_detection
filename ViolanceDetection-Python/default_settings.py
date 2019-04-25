import numpy as np
from datetime import datetime
from modules import make_dirs, time2string

model_settings = {
    # Training settings
    'max_epoch': 8,
    'learning_rate': 1e-4,  # 1e-4 from previous code
    'lr_decay': 0.5,
    'decay_epoch': 3,
    'weight_decay': 0.001,
    'dropout': 0.5,
    'summary_checkpoints': 50,  # Number of steps to create checkpoint
    'model_save_checkpoints': 300,
    'batch_sizes': [30],  # Batch per device -- Original total is 30
    'read_pretrained_model': True,  # ----Parameters to load----
    'load_fc6_fc7': True,
    'load_softmax_linear': False,
    'train_conv': False,  # ----Parameters to train----
    'train_fc6_fc7': True,
    'train_softmax_linear': True,
    'save_graph': True,
    'is_testing': False,

    # Neural-Network settings
    'frames_per_batch': 16,  # Number of frames in a batch
    'video_fps': 12,  # FPS of frames extracted
    'crop_size': 112,  # Input frames dimension
    'channels': 3,
    'trans_max': 15,  # Translation factor for pre-processing

    # System settings
    'devices_to_run': ['/gpu:0'],  # Multiple devices are not supported yet :(
    'num_thread': 8,  # Number of threads to read video files
    'queue_size': 2400,  # Queue size for reading input

    # Directory settings
    'read_from_frames': True,
    'model_name': 'UCF_finetune',
    'checkpoint_dir': './checkpoints/',
    'model_save_dir': './models/',
    # 'model_read_loc': './models/s1m-ucf101.model',
    'model_read_loc': './models/2019-04-25__16-10-31/UCF_finetune-1592',
    # 'model_read_loc': './models/UCF_finetuneFC_last.model',
    'data_home': '../datasets/UCF-101-Frames/',
    'train_test_loc': '../datasets/UCF-ActionRecognitionSplits',
    'train_file_name': '/trainlist01.txt',
    # 'train_file_name': '/train_small.txt',
    'test_file_name': '/testlist01.txt',
    'mean_clip_loc': '../datasets/PreprocessData/crop_mean.npy',

    # Feature Extraction Directory Settings
    'feature_data_home': '../datasets/HockeyFights/',
    'feature_loc': '../datasets/HockeyFights-Splits',
    'feature_extract_file_name': '/datalist.txt'
}


def time2string(t):
    out_str = str(t).split(' ')
    interm = '-'.join(out_str[1].split(':')).split('.')
    out_str[1] = interm[0]
    # out_str.append(interm[1])
    return '__'.join(out_str)


def set_model_settings(model_settings):
    # # Settings for the local computer
    # model_settings['queue_size'] = 100
    # model_settings['num_thread'] = 1
    # model_settings['batch_sizes'] = [5]
    # model_settings['devices_to_run'] = ['/cpu:0']

    # Storage of variables RAM:'/cpu:0' GPU:'/gpu:0'
    model_settings['variable_storage'] = model_settings['devices_to_run'][0]
    # model_settings['variable_storage'] = '/cpu:0'
    # Total number of batch
    model_settings['total_batch'] = np.sum(model_settings['batch_sizes'])

    # Input shape for placeholders
    model_settings['input_shape'] = (model_settings['frames_per_batch'],
                                     model_settings['crop_size'],
                                     model_settings['crop_size'],
                                     model_settings['channels'])

    # Mean clip for input
    model_settings['np_mean'] = np.load(model_settings['mean_clip_loc']). \
        reshape(model_settings['input_shape'])

    if model_settings['is_testing']:
        test_settings = {
            'input_from_placeholders': False,
            'dequeue_immediately': True,
            'dropout': 1.0,
            'read_pretrained_model': True,
            'train_conv': False,
            'train_fc6_fc7': False,
            'train_softmax_linear': False,
        }
        for key, value in test_settings.items():
            model_settings[key] = value

    else:
        model_settings['input_from_placeholders'] = False
        model_settings['dequeue_immediately'] = False

    model_settings['start_time'] = datetime.now()
    out_str = ""
    print('==-----------------Current model settings-----------------==')
    for key, value in model_settings.items():
        if type(value) != type(np.array([])):
            out_str += str(key) + ': ' + str(value) + '\n'

    print(out_str)
    if not model_settings['is_testing']:
        summary_dir = model_settings['checkpoint_dir'] + model_settings['model_name']
        summary_dir += '/' + time2string(model_settings['start_time']) + '/'
        make_dirs(summary_dir)
        summary_dir += 'network_settings.txt'
        with open(summary_dir, 'w') as file:
            file.write(out_str)
    print('-----------------==Current model settings==-----------------')
