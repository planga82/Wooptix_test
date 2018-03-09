"""Shortcuts for some graph operators."""

import tensorflow as tf
import numpy as np

w_initializer = tf.contrib.layers.variance_scaling_initializer
b_initializer = tf.constant_initializer

def conv(inputs, num_outputs, kernel_size, stride=1, rate=1,
        use_bias=True,
        batch_norm=False, is_training=False,
        activation_fn=tf.nn.relu, 
        scope=None, reuse=False):
    if batch_norm:
        normalizer_fn = tf.contrib.layers.batch_norm
        b_init = None
    else:
        normalizer_fn = None
        if use_bias:
            b_init = b_initializer(0.0)
        else:
            b_init = None

    output = tf.contrib.layers.convolution2d(
            inputs=inputs,
            num_outputs=num_outputs, kernel_size=kernel_size, 
            stride=stride, padding='SAME',
            rate=rate,
            weights_initializer=w_initializer(),
            weights_regularizer=tf.contrib.layers.l2_regularizer(1.0),
            biases_initializer=b_init,
            normalizer_fn=normalizer_fn,
            normalizer_params={
                'center':True, 'is_training':is_training,
                'variables_collections':{
                    'beta':[tf.GraphKeys.BIASES],
                    'moving_mean':[tf.GraphKeys.MOVING_AVERAGE_VARIABLES],
                    'moving_variance':[tf.GraphKeys.MOVING_AVERAGE_VARIABLES]},
                }, 
            activation_fn=activation_fn, 
            variables_collections={'weights':[tf.GraphKeys.WEIGHTS], 'biases':[tf.GraphKeys.BIASES]},
            outputs_collections=[tf.GraphKeys.ACTIVATIONS],
            scope=scope, reuse=reuse)
    return output


def fc(inputs, num_outputs,
        use_bias=True,
        batch_norm=False, is_training=False,
        activation_fn=tf.nn.relu, 
        scope=None, reuse=False):
    if batch_norm:
        normalizer_fn = tf.contrib.layers.batch_norm
        b_init = None
    else:
        normalizer_fn = None
        if use_bias:
            b_init = b_initializer(0.0)
        else:
            b_init = None

    output = tf.contrib.layers.fully_connected(
            inputs=inputs,
            num_outputs=num_outputs,
            weights_initializer=w_initializer(),
            weights_regularizer=tf.contrib.layers.l2_regularizer(1.0),
            biases_initializer=b_init,
            normalizer_fn=normalizer_fn,
            normalizer_params={
                'center':True, 'is_training':is_training,
                'variables_collections':{
                    'beta':[tf.GraphKeys.BIASES],
                    'moving_mean':[tf.GraphKeys.MOVING_AVERAGE_VARIABLES],
                    'moving_variance':[tf.GraphKeys.MOVING_AVERAGE_VARIABLES]},
                }, 
            activation_fn=activation_fn,
            outputs_collections=[tf.GraphKeys.ACTIVATIONS], 
            variables_collections={'weights':[tf.GraphKeys.WEIGHTS], 'biases':[tf.GraphKeys.BIASES]},
            scope=scope, reuse=reuse)
    return output


def max_pool2d(inputs, scope, kernel=2, stride=2, padding='SAME'):
    return tf.contrib.layers.max_pool2d(inputs, kernel, stride, scope=scope, padding=padding) 