################################################################################
# Michael Guerzhoy and Davi Frossard, 2016
# AlexNet implementation in TensorFlow, with weights
# Details:
# http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
# With code from https://github.com/ethereon/caffe-tensorflow
# Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

import numpy as np
import cv2
import os

import tensorflow as tf


# todo: fix file-read code
class AlexNet:
    train_x = np.zeros((1, 227, 227, 3)).astype(np.float32)
    train_y = np.zeros((1, 1000))
    x_dim = train_x.shape[1:]
    y_dim = train_y.shape[1]

    def __init__(self, network_model_file):
        # load model from file
        net_data = np.load(network_model_file, encoding="latin1").item()
        self.g = tf.Graph()

        with self.g.as_default():
            self._x = tf.placeholder(tf.float32, (None,) + AlexNet.x_dim)
            self._init_network(net_data)

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

        return

    def _init_network(self, network_data):
        # conv1
        # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
        k_h = 11
        k_w = 11
        c_o = 96
        s_h = 4
        s_w = 4
        conv1W = tf.Variable(network_data["conv1"][0])
        conv1b = tf.Variable(network_data["conv1"][1])
        conv1_in = self._conv(self._x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        conv1 = tf.nn.relu(conv1_in)

        # lrn1
        # lrn(2, 2e-05, 0.75, name='norm1')
        radius = 2
        alpha = 2e-05
        beta = 0.75
        bias = 1.0
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

        # maxpool1
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
        k_h = 3
        k_w = 3
        s_h = 2
        s_w = 2
        padding = 'VALID'
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        # conv2
        # conv(5, 5, 256, 1, 1, group=2, name='conv2')
        k_h = 5
        k_w = 5
        c_o = 256
        s_h = 1
        s_w = 1
        group = 2
        conv2W = tf.Variable(network_data["conv2"][0])
        conv2b = tf.Variable(network_data["conv2"][1])
        conv2_in = self._conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv2 = tf.nn.relu(conv2_in)

        # lrn2
        # lrn(2, 2e-05, 0.75, name='norm2')
        radius = 2
        alpha = 2e-05
        beta = 0.75
        bias = 1.0
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

        # maxpool2
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
        k_h = 3
        k_w = 3
        s_h = 2
        s_w = 2
        padding = 'VALID'
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        # conv3
        # conv(3, 3, 384, 1, 1, name='conv3')
        k_h = 3
        k_w = 3
        c_o = 384
        s_h = 1
        s_w = 1
        group = 1
        conv3W = tf.Variable(network_data["conv3"][0])
        conv3b = tf.Variable(network_data["conv3"][1])
        conv3_in = self._conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv3 = tf.nn.relu(conv3_in)

        # conv4
        # conv(3, 3, 384, 1, 1, group=2, name='conv4')
        k_h = 3
        k_w = 3
        c_o = 384
        s_h = 1
        s_w = 1
        group = 2
        conv4W = tf.Variable(network_data["conv4"][0])
        conv4b = tf.Variable(network_data["conv4"][1])
        conv4_in = self._conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv4 = tf.nn.relu(conv4_in)

        # conv5
        # conv(3, 3, 256, 1, 1, group=2, name='conv5')
        k_h = 3
        k_w = 3
        c_o = 256
        s_h = 1
        s_w = 1
        group = 2
        conv5W = tf.Variable(network_data["conv5"][0])
        conv5b = tf.Variable(network_data["conv5"][1])
        conv5_in = self._conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv5 = tf.nn.relu(conv5_in)

        # maxpool5
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        k_h = 3
        k_w = 3
        s_h = 2
        s_w = 2
        padding = 'VALID'
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        # fc6
        # fc(4096, name='fc6')
        fc6W = tf.Variable(network_data["fc6"][0])
        fc6b = tf.Variable(network_data["fc6"][1])
        maxpool5_flat = tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))])

        self.fc6_feature = tf.nn.xw_plus_b(maxpool5_flat, fc6W, fc6b)
        self.init = tf.global_variables_initializer()

        return

    def _conv(self, input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
        '''From https://github.com/ethereon/caffe-tensorflow
        '''
        c_i = input.get_shape()[-1]
        assert c_i % group == 0
        assert c_o % group == 0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

        if group == 1:
            conv = convolve(input, kernel)
        else:
            # using Tensorflow 1.5 api
            input_groups = tf.split(input, group, 3)  # tf.split(3, group, input)
            kernel_groups = tf.split(kernel, group, 3)  # tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            conv = tf.concat(output_groups, 3)  # tf.concat(3, output_groups)
        return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])

    def predict(self, image_data):
        with tf.Session(graph=self.g) as sess:
            sess.run(self.init)
            output_fc6 = sess.run(self.fc6_feature, feed_dict={self._x: [image_data]})

        return output_fc6
