import tensorflow as tf


def _variable(model_settings, name, shape, initializer, trainable):
    with tf.device(model_settings['variable_storage']):
        var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    tf.add_to_collection('all_variables', var)
    return var


def _variable_with_weight_decay(model_settings, name, shape, trainable):
    wd = model_settings['weight_decay']
    var = _variable(model_settings, name, shape, tf.contrib.layers.xavier_initializer(),
                    trainable=trainable)
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_decay_loss')
    tf.add_to_collection('losses', weight_decay)
    return var


def conv3d_layer(model_settings, tensor_in,
                 w_name, b_name, shape_weight,
                 shape_bias, layer_name='conv', trainable=True):
    with tf.variable_scope(layer_name):
        w = _variable_with_weight_decay(model_settings, w_name,
                                        shape_weight, trainable=trainable)
        b = _variable(model_settings, b_name, shape_bias,
                      tf.constant_initializer(0.0), trainable=trainable)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, w)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, b)

        conv = tf.nn.conv3d(tensor_in, w, strides=[1, 1, 1, 1, 1], padding='SAME', name='conv')
        conv_b = tf.nn.bias_add(conv, b, name='conv_bias')
        act = tf.nn.relu(conv_b, 'relu')
        tf.summary.histogram('Convolution_Weights', w)
        tf.summary.histogram('Conv_Biases', b)
        tf.summary.histogram('Conv_Activations', act)
        return act


def fc_layer(model_settings, tensor_in, w_name,
             b_name, shape_weight, shape_bias,
             layer_name='fc', trainable=True):
    with tf.variable_scope(layer_name):
        w = _variable_with_weight_decay(model_settings, w_name,
                                        shape_weight, trainable=trainable)
        b = _variable(model_settings, b_name, shape_bias,
                      tf.constant_initializer(0.0), trainable=trainable)

        if model_settings['load_fc_layers']:
            tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, w)
            tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, b)

        fc_1 = tf.matmul(tensor_in, w, name='fc_linear')
        fc_b = tf.nn.bias_add(fc_1, b, name='fc_bias')
        act = tf.nn.relu(fc_b, 'relu')
        tensor_out = tf.nn.dropout(act,
                                   model_settings['dropout_placeholder'],
                                   name='dropout')
        tf.summary.histogram('FC_Weights', w)
        tf.summary.histogram('FC_Biases', b)
        tf.summary.histogram('FC_Activations', act)
        return tensor_out


def max_pool_3d(tensor_in, k, layer_name='max_pool'):
    with tf.variable_scope(layer_name):
        k_size = [1, k, 2, 2, 1]
        tensor_out = tf.nn.max_pool3d(tensor_in, ksize=k_size, strides=k_size, padding='SAME')
        return tensor_out


def model(_X, model_settings):
    wd = model_settings['weight_decay']
    train_conv = model_settings['train_conv']
    train_fc = model_settings['train_fc']

    # Convolution 1a
    conv1 = conv3d_layer(model_settings, _X, 'wc1a', 'bc1a',
                         [3, 3, 3, 3, 64], [64], 'Convolution_1a', train_conv)
    pool1 = max_pool_3d(conv1, 1, 'Max_Pooling_1')

    # Convolution 2a
    conv2 = conv3d_layer(model_settings, pool1, 'wc2a', 'bc2a',
                         [3, 3, 3, 64, 128], [128], 'Convolution_2a', train_conv)
    pool2 = max_pool_3d(conv2, 2, 'Max_Pooling_2')

    # Convolution 3a, 3b
    conv3a = conv3d_layer(model_settings, pool2, 'wc3a', 'bc3a',
                          [3, 3, 3, 128, 256], [256], 'Convolution_3a', train_conv)
    conv3b = conv3d_layer(model_settings, conv3a, 'wc3b', 'bc3b',
                          [3, 3, 3, 256, 256], [256], 'Convolution_3b', train_conv)
    pool3 = max_pool_3d(conv3b, 2, 'Max_Pooling_3')

    # Convolution 4a, 4b
    conv4a = conv3d_layer(model_settings, pool3, 'wc4a', 'bc4a',
                          [3, 3, 3, 256, 512], [512], 'Convolution_4a', train_conv)
    conv4b = conv3d_layer(model_settings, conv4a, 'wc4b', 'bc4b',
                          [3, 3, 3, 512, 512], [512], 'Convolution_4b', train_conv)
    pool4 = max_pool_3d(conv4b, 2, 'Max_Pooling_4')

    # Convolution 4a, 4b
    conv5a = conv3d_layer(model_settings, pool4, 'wc5a', 'bc5a',
                          [3, 3, 3, 512, 512], [512], 'Convolution_5a', train_conv)
    conv5b = conv3d_layer(model_settings, conv5a, 'wc5b', 'bc5b',
                          [3, 3, 3, 512, 512], [512], 'Convolution_5b', train_conv)
    pool5 = max_pool_3d(conv5b, 2, 'Max_Pooling_5')

    # Reshape weights for fc6
    pool5 = tf.transpose(pool5, perm=[0, 1, 4, 2, 3])
    dense1 = tf.reshape(pool5, [model_settings['batch_size'], 8192])

    # FC6 and FC7
    dense1 = fc_layer(model_settings, dense1, 'wd1', 'bd1', [8192, 4096],
                      [4096], 'FC6', train_fc)
    dense2 = fc_layer(model_settings, dense1, 'wd2', 'bd2', [4096, 4096],
                      [4096], 'FC7', train_fc)

    model_settings['clip_features'] = dense2

    # Last affine transformation for classification
    with tf.variable_scope('Softmax_Linear'):
        w_out = _variable_with_weight_decay(model_settings, 'wout', [4096, 101], train_fc)
        b_out = _variable(model_settings, 'bout', [101], tf.constant_initializer(0.0), train_fc)
        # tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, w_out)
        # tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, b_out)
        out = tf.matmul(dense2, w_out) + b_out

    return out


def tower_loss(loss_var_scope, logit, labels):
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


# Updates:
# - Add top-n correct predictions for ground up training
def tower_accuracy(logit, labels):
    correct_predictions = tf.equal(labels, tf.argmax(logit, 1))
    total_correct = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return total_correct
