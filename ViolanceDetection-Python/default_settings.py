import numpy as np
from datetime import datetime

model_settings = {
    # Training settings
    'max_epoch': 12,
    'moving_decay': 0.9999, 'weight_decay': 0.00005, 'dropout': 0.5,
    'learning_rate': 3e-4,  # 1e-4 from previous code
    'summary_checkpoints': 100,  # Number of steps to create checkpoint
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
    # 'model_read_loc' : '../ViolanceDetection-Jupyter/models/s1m-ucf101.model',
    'model_read_loc': './models/s1m-ucf101.model',
    # 'model_read_loc': './models/2019-04-23__16-19-36/UCF_finetune-699',
    'data_home': '../datasets/UCF-101-Frames/',
    'train_test_loc': '../datasets/UCF-ActionRecognitionSplits',
    'train_file_name': '/trainlist01.txt',
    # 'train_file_name': '/train_small.txt',
    'test_file_name': '/testlist01.txt',
    'mean_clip_loc': '../datasets/PreprocessData/crop_mean.npy'
}


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
        model_settings['input_from_placeholders'] = False
        model_settings['dequeue_immediately'] = True
        model_settings['dropout'] = 1.0
    else:
        model_settings['input_from_placeholders'] = False
        model_settings['dequeue_immediately'] = False

    model_settings['start_time'] = datetime.now()
    print('==-----------------Current model settings-----------------==')
    for key, value in model_settings.items():
        if type(value) != type(np.array([])):
            print(key, ': ', value)
    print('-----------------==Current model settings==-----------------')
