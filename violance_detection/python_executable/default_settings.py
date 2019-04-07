model_settings = {'max_steps' : 10000, 'batch_size' : 25, 'frames_per_clip' : 16, 
                  'crop_size' : 112, 'num_gpu' : 1, 'channels' : 3,
                  'checkpoint_dir' : './checkpoints/allrand_fc_train',
                  'model_read_dir' : './models/s1m_mod.model',
                  'model_save_dir' : './models/C3D_1.model',
                  'moving_decay' : 0.9999, 'weight_decay' : 0.00005, 'dropout' : 0.5,
                  'learning_rate' : 1e-4, 'checkpoints' : 100,
                  'train_file_loc' : './data/dir_files/train_ucf.csv',
                  'test_file_loc' : './data/dir_files/test_ucf.csv', 
                  'data_home' : './data/',
                  'feature_file_loc' : './data/dir_files/test_ucf_full.lst',
                  'num_thread' : 8, 'queue_size' : 1500,
                  'read_pretrained_model' : True}

model_settings['total_batch'] = model_settings['batch_size'] * model_settings['num_gpu']
model_settings['input_shape'] = (model_settings['frames_per_clip'],
                                 model_settings['crop_size'],
                                 model_settings['crop_size'],
                                 model_settings['channels'])
model_settings['np_mean'] = np.load('crop_mean.npy').reshape(model_settings['input_shape'])