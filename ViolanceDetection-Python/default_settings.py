import numpy as np

model_settings = {
    # Training settings
    'max_steps': 10000,
    'moving_decay': 0.9999, 'weight_decay': 0.00005, 'dropout': 0.5,
    'learning_rate': 1e-4,  # 1e-4 from previous code
    'checkpoints': 200,  # Number of steps to create checkpoint
    'batch_size': 5,  # Batch per GPU
    'read_pretrained_model': False,
    'load_fc_layers': False,
    'train_conv': True,
    'train_fc': True,

    # Neural-Network settings
    'frames_per_batch': 16,  # Number of frames in a batch
    'video_fps': 12,  # FPS of frames extracted
    'crop_size': 112,  # Input frames dimension
    'channels': 3,

    # System settings
    'running_device': '/cpu:0',  # Training device
    'num_gpu': 1,  # Number of GPU's in the system
    'variable_storage': '/gpu:0',  # Storage of variables RAM:'/cpu:0' GPU:'/gpu:0'
    'num_thread': 8,  # Number of threads to read video files
    'queue_size': 500,  # Queue size for reading input

    # Directory settings
    'checkpoint_dir': './checkpoints',
    # 'model_read_dir' : './models/s1m_mod.model',
    # 'model_save_dir' : './models/C3D_1.model',
    'data_home': '../datasets/UCF-101/',
    'train_test_loc': '../datasets/UCF-ActionRecognitionSplits',
    'train_file_name': '/trainlist01.txt',
    'test_file_name': '/testlist01.txt',
    'mean_clip_loc': '../datasets/PreprocessData/crop_mean.npy'
}

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
