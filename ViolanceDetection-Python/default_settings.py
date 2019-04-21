import numpy as np
from datetime import datetime

model_settings = {
    # Training settings
    'current_epoch': 1,
    'max_steps': 100,
    'moving_decay': 0.9999, 'weight_decay': 0.00005, 'dropout': 0.5,
    'learning_rate': 1e-3,  # 1e-4 from previous code
    'checkpoints': 200,  # Number of steps to create checkpoint
    'batch_size': 30,  # Batch per GPU
    'read_pretrained_model': True,
    'load_fc_layers': True,
    'train_conv': False,
    'train_fc': True,
    'save_graph': True,
    'is_testing': False,

    # Neural-Network settings
    'frames_per_batch': 16,  # Number of frames in a batch
    'video_fps': 12,  # FPS of frames extracted
    'crop_size': 112,  # Input frames dimension
    'channels': 3,
    'trans_max': 10,  # Translation factor for pre-processing

    # System settings
    'run_on_cpu': False,  # Training device
    'num_gpu': 1,  # Number of GPU's in the system
    'variable_storage': '/gpu:0',  # Storage of variables RAM:'/cpu:0' GPU:'/gpu:0'
    'num_thread': 1,  # Number of threads to read video files
    'queue_size': 300,  # Queue size for reading input

    # Directory settings
    'model_name': 'UCF_finetune',
    'checkpoint_dir': './checkpoints/',
    'model_save_dir': './models/',
    # 'model_read_loc' : '../ViolanceDetection-Jupyter/models/s1m-ucf101.model',
    'model_read_loc': './models/UCF_finetuneFC_last.model',
    'data_home': '../datasets/UCF-101/',
    'train_test_loc': '../datasets/UCF-ActionRecognitionSplits',
    'train_file_name': '/trainlist01.txt',
    # 'train_file_name': '/train_small.txt',
    'test_file_name': '/testlist01.txt',
    'mean_clip_loc': '../datasets/PreprocessData/crop_mean.npy'
}


def set_model_settings(model_settings):
    # Total number of batch
    model_settings['total_batch'] = model_settings['batch_size'] * model_settings['num_gpu']

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
        model_settings['trans_max'] = 0
    else:
        model_settings['input_from_placeholders'] = False
        model_settings['dequeue_immediately'] = False

    model_settings['start_time'] = datetime.now()
