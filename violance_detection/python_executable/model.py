import os
import time
import cv2
import numpy as np
import PIL.Image as Image
import random
import tensorflow as tf
from datetime import datetime
import threading 


def _variable_on_cpu(name, shape, initializer, trainable):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    return var


def _variable_with_weight_decay(name, shape, wd, trainable):
    var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer(), trainable=trainable)
    #Check this weight_decay loss!
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_decay_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def conv3d_layer(tensor_in, w_name, b_name, shape_weight, shape_bias, wd, layer_name='conv', trainable=True):
    with tf.variable_scope(layer_name):
        w = _variable_with_weight_decay(w_name, shape_weight, wd, trainable=trainable)
        b = _variable_on_cpu(b_name, shape_bias, tf.constant_initializer(0.0), trainable=trainable)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, w)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, b)
        
        conv = tf.nn.conv3d(tensor_in, w, strides=[1,1,1,1,1], padding='SAME',name='conv')
        conv_b = tf.nn.bias_add(conv, b, name='conv_bias')
        act = tf.nn.relu(conv_b, 'relu')
        tf.summary.histogram('Convolution_Weights', w)
        tf.summary.histogram('Conv_Biases', b)
        tf.summary.histogram('Conv_Activations', act)
        return act


def fc_layer(tensor_in, w_name, b_name, shape_weight, shape_bias, wd, _dropout, layer_name='fc', trainable=True):
    with tf.variable_scope(layer_name):
        w = _variable_with_weight_decay(w_name, shape_weight, wd, trainable=trainable)
        b = _variable_on_cpu(b_name, shape_bias, tf.constant_initializer(0.0), trainable=trainable)
        #tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, w)
        #tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, b)
        
        tf.add_to_collection('fc_layer', w)
        tf.add_to_collection('fc_layer', b)
        
        fc_1 = tf.matmul(tensor_in, w, name='fc_linear')
        fc_b = tf.nn.bias_add(fc_1, b, name='fc_bias')
        act = tf.nn.relu(fc_b, 'relu')
        tensor_out = tf.nn.dropout(act, _dropout, name='dropout')
        tf.summary.histogram('FC_Weights', w)
        tf.summary.histogram('FC_Biases', b)
        tf.summary.histogram('FC_Activations', act)
        return tensor_out
        

def max_pool_3d(tensor_in, k, layer_name='max_pool'):
    with tf.variable_scope(layer_name):
        k_size=[1,k,2,2,1]
        tensor_out = tf.nn.max_pool3d(tensor_in, ksize=k_size, strides=k_size, padding='SAME')
        return tensor_out
    

def c3d_model(_X, model_settings):
    wd = model_settings['weight_decay']
    #Convolution 1a
    conv1 = conv3d_layer(_X, 'wc1a', 'bc1a', [3, 3, 3, 3, 64], [64], None, 'Convolution_1a', False)
    pool1 = max_pool_3d(conv1, 1, 'Max_Pooling_1')
    
    #Convolution 2a
    conv2 = conv3d_layer(pool1, 'wc2a', 'bc2a', [3, 3, 3, 64, 128], [128], None, 'Convolution_2a', False)
    pool2 = max_pool_3d(conv2, 2, 'Max_Pooling_2')
    
    #Convolution 3a, 3b
    conv3a = conv3d_layer(pool2, 'wc3a', 'bc3a',  [3, 3, 3, 128, 256], [256], None, 'Convolution_3a', False)
    conv3b = conv3d_layer(conv3a, 'wc3b', 'bc3b', [3, 3, 3, 256, 256], [256], None, 'Convolution_3b', False)
    pool3 = max_pool_3d(conv3b, 2, 'Max_Pooling_3')
    
    #Convolution 4a, 4b
    conv4a = conv3d_layer(pool3, 'wc4a', 'bc4a',  [3, 3, 3, 256, 512], [512], None, 'Convolution_4a', False)
    conv4b = conv3d_layer(conv4a, 'wc4b', 'bc4b', [3, 3, 3, 512, 512], [512], None, 'Convolution_4b', False)
    pool4 = max_pool_3d(conv4b, 2, 'Max_Pooling_4')
    
    #Convolution 4a, 4b
    conv5a = conv3d_layer(pool4, 'wc5a', 'bc5a',  [3, 3, 3, 512, 512], [512], None, 'Convolution_5a', False)
    conv5b = conv3d_layer(conv5a, 'wc5b', 'bc5b', [3, 3, 3, 512, 512], [512], None, 'Convolution_5b', False)
    pool5 = max_pool_3d(conv5b, 2, 'Max_Pooling_5')

    #Reshape weights for fc6
    pool5 = tf.transpose(pool5, perm=[0,1,4,2,3])
    dense1 = tf.reshape(pool5, [model_settings['batch_size'], 8192])

    #FC6 and FC7
    dense1 = fc_layer(dense1, 'wd1', 'bd1', [8192, 4096], [4096], wd,
                      model_settings['dropout_placeholder'], 'FC6', True)
    dense2 = fc_layer(dense1, 'wd2', 'bd2', [4096, 4096], [4096], wd,
                      model_settings['dropout_placeholder'], 'FC7', True)
    
    model_settings['clip_features'] = dense2
    
    with tf.variable_scope('Softmax_Linear'):
        w_out = _variable_with_weight_decay('wout', [4096, 101], wd, True)
        b_out = _variable_on_cpu('bout', [101], tf.constant_initializer(0.0), True)
        #tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, w_out)
        #tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, b_out)
        out = tf.matmul(dense2, w_out) + b_out
    
    return out


def calc_tower_loss(loss_var_scope, logit, labels):
    with tf.variable_scope(loss_var_scope):
        
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, 
                                                                       logits=logit,
                                                                       name='cross_entropy_per_class')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='Cross_Entropy_Mean')
        
        weight_losses = tf.get_collection('losses')
        tf.summary.scalar('Weight_decay_loss', tf.reduce_mean(weight_losses)) 
        
        tf.add_to_collection('losses', cross_entropy_mean)
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        
        return total_loss


#Updates:
##Add top-n correct predictions for ground up training
def calc_tower_accuracy(logit, labels):
    
    correct_predictions = tf.equal(labels, tf.argmax(logit, 1))
    total_correct = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return total_correct
    

def average_gradients(tower_grads):
    
    average_grads=[]
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads


def create_forward_graph(model_settings):
    
    queue = model_settings['queue']
    tower_accuracy = []
    tower_loss = []
    with tf.variable_scope(tf.get_variable_scope()):
        for gpu_index in range(model_settings['num_gpu']):
            with tf.device('/gpu:%d' % gpu_index):
                with tf.name_scope('Tower_%d' % gpu_index) as scope:
                    ind_begin, ind_end= (gpu_index * model_settings['batch_size'],
                                         (gpu_index+1) * model_settings['batch_size'])
                    
                    feed_input, feed_label = queue.dequeue_many(model_settings['batch_size'])
                    model_out = c3d_model(feed_input, model_settings)

                    loss = calc_tower_loss(scope, model_out, feed_label)
                    accuracy = calc_tower_accuracy(model_out, feed_label)
                    tower_loss.append(loss)
                    tower_accuracy.append(accuracy)

                    tf.get_variable_scope().reuse_variables()
                    
    tower_mean_loss = tf.reduce_mean(tower_loss)
    tower_mean_accuracy = tf.reduce_mean(tower_accuracy)
    tf.summary.scalar('Total_Loss', tower_mean_loss)
    tf.summary.scalar('Top1_Correct_Predictions', tower_mean_accuracy)
    
    model_settings['tower_mean_loss'] = tower_mean_loss
    model_settings['tower_mean_accuracy'] = tower_mean_accuracy
    model_settings['tower_loss'] = tower_loss
                    
def create_backward_graph(model_settings):
    
    global_step=tf.get_variable('Global_Step',[],
                                initializer=tf.constant_initializer(0),
                                trainable=False)
    
    opt, tower_loss = model_settings['optimizer'], model_settings['tower_loss']
    
    tower_grads=[]
    with tf.variable_scope(tf.get_variable_scope()):
        for gpu_index in range(model_settings['num_gpu']):
            with tf.device('/gpu:%d' % gpu_index):
                with tf.name_scope('Tower_%d' % gpu_index) as scope:
                    loss = tower_loss[gpu_index]
                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)
    grads = average_gradients(tower_grads)
    
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    variable_averages = tf.train.ExponentialMovingAverage(model_settings['moving_decay'], global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    
    train_op = tf.group(apply_gradient_op, variable_averages_op)

    summary_op = tf.summary.merge_all()
    
    model_settings['train_op'], model_settings['summary_op'] = train_op, summary_op