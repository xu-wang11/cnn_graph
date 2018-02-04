#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# created by Xu Wang
# 2018-01-10
import collections
from .graph_model import GraphModel
import tensorflow as tf
from . import graph
import numpy as np
from tensorflow.python.ops.rnn_cell_impl import RNNCell

import filter


_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ('c', 'h'))


class LSTMStateTuple(_LSTMStateTuple):
    __slots__ = ()  # What is this??

    @property
    def dtype(self):
        (c, h) = self
        if not c.dtype == h.dtype:
            raise TypeError("Inconsistent internal state")
        return c.dtype


class GConvLSTMCell(RNNCell):
    def __init__(self, num_units, forget_bias=1.0,
                 state_is_tuple=True, activation=None, reuse=None,
                 laplacian=None, lmax=None, K=None, feat_in=None, nNode=None, filter_type="cheby_conv"):
        """

        :param num_units:
        :param forget_bias:
        :param state_is_tuple:
        :param activation:
        :param reuse:
        :param laplacian:
        :param lmax:
        :param K:
        :param feat_in:
        :param nNode:
        :param filter_type: filter_type is defined in filter.py including: cheby_conv, fourier_conv
        """

        super().__init__(_reuse=reuse)  # super what is it?

        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or tf.tanh
        self._laplacian = laplacian
        self._lmax = lmax
        self._K = K
        self._feat_in = feat_in
        self._nNode = nNode
        self.filter = getattr(filter, filter_type)

    @property
    def state_size(self):
        return (LSTMStateTuple((self._nNode, self._num_units), (self._nNode, self._num_units))
        if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + "myZeroState"):
            zero_state_c = tf.zeros([batch_size, self._nNode, self._num_units], name='c')
            zero_state_h = tf.zeros([batch_size, self._nNode, self._num_units], name='h')
            # print("When it called, I print batch_size", batch_size)
            return (zero_state_c, zero_state_h)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(value=state, num_or_size_splits=2, axis=1)
            laplacian = self._laplacian
            lmax = self._lmax
            K = self._K
            feat_in = self._feat_in

            # batch_size, nNode, feat_in = inputs.get_shape()
            nNode = self._nNode
            feat_out = self._num_units
            if K is None:
                K = 2
            scope = tf.get_variable_scope()
            with tf.variable_scope(scope) as scope:
                try:
                    if self.filter.__name__ == "cheby_conv":
                    # Need four diff Wconv weight + for Hidden weight
                        Wzxt = tf.get_variable("Wzxt", [K * feat_in, feat_out], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                        Wixt = tf.get_variable("Wixt", [K * feat_in, feat_out], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                        Wfxt = tf.get_variable("Wfxt", [K * feat_in, feat_out], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                        Woxt = tf.get_variable("Woxt", [K * feat_in, feat_out], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

                        Wzht = tf.get_variable("Wzht", [K * feat_out, feat_out], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                        Wiht = tf.get_variable("Wiht", [K * feat_out, feat_out], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                        Wfht = tf.get_variable("Wfht", [K * feat_out, feat_out], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                        Woht = tf.get_variable("Woht", [K * feat_out, feat_out], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

                    elif self.filter.__name__ == "fourier_conv":
                        Wzxt = tf.get_variable("Wzxt", [nNode, feat_out, feat_in], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                        Wixt = tf.get_variable("Wixt", [nNode, feat_out, feat_in], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                        Wfxt = tf.get_variable("Wfxt", [nNode, feat_out, feat_in], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                        Woxt = tf.get_variable("Woxt", [nNode, feat_out, feat_in], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

                        Wzht = tf.get_variable("Wzht", [nNode, feat_out, feat_out], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                        Wiht = tf.get_variable("Wiht", [nNode, feat_out, feat_out], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                        Wfht = tf.get_variable("Wfht", [nNode, feat_out, feat_out], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                        Woht = tf.get_variable("Woht", [nNode, feat_out, feat_out], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))


                except ValueError:
                    scope.reuse_variables()
                    if self.filter.__name__ == "cheby_conv":
                        # Need four diff Wconv weight + for Hidden weight
                        Wzxt = tf.get_variable("Wzxt", [K * feat_in, feat_out], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                        Wixt = tf.get_variable("Wixt", [K * feat_in, feat_out], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                        Wfxt = tf.get_variable("Wfxt", [K * feat_in, feat_out], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                        Woxt = tf.get_variable("Woxt", [K * feat_in, feat_out], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

                        Wzht = tf.get_variable("Wzht", [K * feat_out, feat_out], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                        Wiht = tf.get_variable("Wiht", [K * feat_out, feat_out], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                        Wfht = tf.get_variable("Wfht", [K * feat_out, feat_out], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                        Woht = tf.get_variable("Woht", [K * feat_out, feat_out], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

                    elif self.filter.__name__ == "fourier_conv":
                        Wzxt = tf.get_variable("Wzxt", [nNode, feat_out, feat_in], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                        Wixt = tf.get_variable("Wixt", [nNode, feat_out, feat_in], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                        Wfxt = tf.get_variable("Wfxt", [nNode, feat_out, feat_in], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                        Woxt = tf.get_variable("Woxt", [nNode, feat_out, feat_in], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

                        Wzht = tf.get_variable("Wzht", [nNode, feat_out, feat_out], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                        Wiht = tf.get_variable("Wiht", [nNode, feat_out, feat_out], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                        Wfht = tf.get_variable("Wfht", [nNode, feat_out, feat_out], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                        Woht = tf.get_variable("Woht", [nNode, feat_out, feat_out], dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

                bzt = tf.get_variable("bzt", [feat_out])
                bit = tf.get_variable("bit", [feat_out])
                bft = tf.get_variable("bft", [feat_out])
                bot = tf.get_variable("bot", [feat_out])

                # gconv Calculation
                # zxt = cheby_conv(inputs, laplacian, lmax, feat_out, K, Wzxt)
                # zht = cheby_conv(h, laplacian, lmax, feat_out, K, Wzht)
                zxt = self.filter(inputs, laplacian, lmax, feat_out, K, Wzxt)
                zht = self.filter(h, laplacian, lmax, feat_out, K, Wzht)
                zt = zxt + zht + bzt
                zt = tf.tanh(zt)

                # ixt = cheby_conv(inputs, laplacian, lmax, feat_out, K, Wixt)
                # iht = cheby_conv(h, laplacian, lmax, feat_out, K, Wiht)
                ixt = self.filter(inputs, laplacian, lmax, feat_out, K, Wixt)
                iht = self.filter(h, laplacian, lmax, feat_out, K, Wiht)
                it = ixt + iht + bit
                it = tf.sigmoid(it)

                # fxt = cheby_conv(inputs, laplacian, lmax, feat_out, K, Wfxt)
                # fht = cheby_conv(h, laplacian, lmax, feat_out, K, Wfht)
                fxt = self.filter(inputs, laplacian, lmax, feat_out, K, Wfxt)
                fht = self.filter(h, laplacian, lmax, feat_out, K, Wfht)
                ft = fxt + fht + bft
                ft = tf.sigmoid(ft)

                # oxt = cheby_conv(inputs, laplacian, lmax, feat_out, K, Woxt)
                # oht = cheby_conv(h, laplacian, lmax, feat_out, K, Woht)
                oxt = self.filter(inputs, laplacian, lmax, feat_out, K, Woxt)
                oht = self.filter(h, laplacian, lmax, feat_out, K, Woht)
                ot = oxt + oht + bot
                ot = tf.sigmoid(ot)

                # c
                new_c = ft * c + it * zt

                # h
                new_h = ot * tf.tanh(new_c)

                if self._state_is_tuple:
                    new_state = LSTMStateTuple(new_c, new_h)
                else:
                    new_state = tf.concat([new_c, new_h], 1)
                return new_h, new_state


class GconvModel(GraphModel):

    def __init__(self, laplacian, seq_num_closeness, seq_num_period, seq_num_trend, filter_num=64, conv_layer_num=4, filter='cheby_conv',
                 num_epochs=20, learning_rate=0.1, decay_rate=0.95, decay_steps=None, momentum=0.9,
                 regularization=0, dropout=0, batch_size=100, eval_frequency=200,
                 dir_name='', feature_num=6, kernel_num=2, in_feature_num=2, out_feature_num=2, infer_func='inference_glstm', lstm_layer_count = 1):

        super().__init__()

        self.feature_num = feature_num
        self.model_type = 'glstm'
        self.batch_size = batch_size
        self.in_feature_num = in_feature_num
        # self.num_time_steps = feature_num // 2
        self.num_time_steps_closeness = seq_num_closeness
        self.num_time_steps_period = seq_num_period
        self.num_time_steps_trend = seq_num_trend
        self.out_feature_num = out_feature_num
        self.laplacian = laplacian
        self.lmax = graph.lmax(self.laplacian)
        self.num_hidden = filter_num
        self.kernel_num = kernel_num
        # self.L, self.F, self.K, self.p, self.M = L, F, K, p, M
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        self.filter = filter
        self.reuse = None
        self.node_num = self.laplacian.shape[0]
        self.conv_layer_num = conv_layer_num
        self.infer_func = infer_func
        self.lstm_layer_count = lstm_layer_count
        self.build_graph(self.node_num, np.sum(self.feature_num), self.out_feature_num)

    def to_string(self):
        str = '|{0}| {1}| {2}| {3}| {4}| {5}| {6}| {7}| {8}| {9} {10}'.format(self.feature_num, self.batch_size, self.in_feature_num, self.num_time_steps_closeness, self.num_hidden, self.kernel_num, self.learning_rate, self.filter, self.conv_layer_num, self.lstm_layer_count, self.infer_func)
        return str

    def _inference(self, x, dropout):
        return getattr(self, self.infer_func)(x)

    def inference_lstm(self, x, dropout):
        return None

    def inference_glstm(self, x):
        A, B, C = x.get_shape().as_list()
        x = tf.reshape(x, [A, B, self.num_time_steps_closeness, int(C / self.num_time_steps_closeness)])
        x = tf.transpose(x, [0, 1, 3, 2])
        x = tf.unstack(x, self.num_time_steps_closeness, 3)
        lstm_output = self.glstm_layer(x, self.num_time_steps_closeness, self.lstm_layer_count)
            # Check the tf version here
        x = lstm_output[-1]  # tf.stack(outputs, axis=3)
        # output with full connected layer
        x = self.fc_layer(x, self.out_feature_num)
        return x

    def inference_glstm_period_no_expand(self, x):
        assert(self.num_time_steps_closeness == self.num_time_steps_period)
        num_dims = x.get_shape()
        x = tf.reshape(x, [int(num_dims[0]), int(num_dims[1]), self.num_time_steps_closeness, int(num_dims[2]) / self.num_time_steps_closeness])
        x = tf.transpose(x, [0, 1, 3, 2])
        x = tf.unstack(x, self.num_time_steps_closeness, 3)
        lstm_output = self.glstm_layer(x, self.num_time_steps_closeness, self.lstm_layer_count)
            # Check the tf version here
        lstm_output = lstm_output[-1]  # tf.stack(outputs, axis=3)
        x = lstm_output
        # output with full connected layer
        self.fc_layer(x, self.out_feature_num)
        return x

    def inference_gconv(self, x):
        with tf.name_scope('conv_init'):
            with tf.variable_scope('conv_init'):
                x = getattr(filter, self.filter)(x, self.laplacian, self.lmax, self.num_hidden, self.kernel_num)
                self.nets[x.name] = x
                x = self.activation_function(x, 'tanh')
                self.nets[x.name] = x
        with tf.name_scope('conv_layers'):
            for i in range(self.conv_layer_num):
                with tf.variable_scope('conv_layer_{}'.format(i)):
                    x = self.residual_layer(x, self.num_hidden, 'tanh', 'residual_layer_{0}'.format(i))
                    self.nets[x.name] = x
        with tf.name_scope('output_layer'):
            with tf.variable_scope('output_layer'):
                x = getattr(filter, self.filter)(x, self.laplacian, self.lmax, self.out_feature_num,
                                             self.kernel_num)
        # x = self.activation_function(x, 'tanh')
        return x

    def inference_gconv_period_no_expand(self, x):
        with tf.name_scope('conv_init'):
            with tf.variable_scope('conv_init'):
                x = getattr(filter, self.filter)(x, self.laplacian, self.lmax, self.num_hidden, self.kernel_num)
                self.nets[x.name] = x
                x = self.activation_function(x, 'relu')
                self.nets[x.name] = x
        with tf.name_scope('conv_layers'):
            for i in range(self.conv_layer_num):
                with tf.variable_scope('conv_layer_{}'.format(i)):
                    x = self.residual_layer(x, self.num_hidden, 'relu', 'residual_layer_{0}'.format(i))
                    self.nets[x.name] = x
        with tf.name_scope('output_layer'):
            with tf.variable_scope('output_layer'):
                x = getattr(filter, self.filter)(x, self.laplacian, self.lmax, self.out_feature_num,
                                             self.kernel_num)
        # x = self.activation_function(x, 'relu')
        return x

    def inference_gconv_period_expand(self, x):
        batch_size, node_num, feature_in = x.get_shape().as_list()
        x = tf.reshape(x, [batch_size, node_num, feature_in, 1])
        # x = tf.transpose(x, [0, 1, 3, 2])
        x_arr = tf.unstack(x, axis=2)
        x_0 = tf.concat(x_arr[0:self.num_time_steps_closeness * 2], axis=2)
        x_1 = tf.concat(x_arr[self.num_time_steps_closeness * 2: (self.num_time_steps_closeness + self.num_time_steps_period) * 2], axis=2)
        x_2 = tf.concat(x_arr[(self.num_time_steps_closeness + self.num_time_steps_period) * 2: (self.num_time_steps_closeness + self.num_time_steps_period + self.num_time_steps_trend) * 2], axis=2)
        x_arr = [x_0, x_1, x_2]
        output_arr = []
        for j in range(3):
            x = x_arr[j]
            with tf.name_scope('conv_init'):
                with tf.variable_scope('conv_init_{}'.format(j)):
                    x = getattr(filter, self.filter)(x, self.laplacian, self.lmax, self.num_hidden, self.kernel_num)
                    self.nets[x.name] = x
                    x = self.activation_function(x, 'tanh')
                    self.nets[x.name] = x
            with tf.name_scope('conv_layers'):
                for i in range(self.conv_layer_num):
                    with tf.variable_scope('conv_layer_{0}_{1}'.format(i, j)):
                        x = self.residual_layer(x, self.num_hidden, 'relu', 'residual_layer_{0}'.format(i))
                        self.nets[x.name] = x
            with tf.name_scope('output_layer'):
                with tf.variable_scope('output_layer_{}'.format(j)):
                    x = getattr(filter, self.filter)(x, self.laplacian, self.lmax, self.out_feature_num,
                                                 self.kernel_num)

            x = self.activation_function(x, 'relu')
            output_arr.append(x)
        x = tf.concat(output_arr, axis=2)
        with tf.name_scope('merge_layer'):
            with tf.variable_scope('merge_layer'):
                x = getattr(filter, self.filter)(x, self.laplacian, self.lmax, self.out_feature_num,
                                             self.kernel_num)

        # x = self.activation_function(x, 'tanh')
        return x

    def inference_glstm_gconv(self, x):
        batch_size, node_num, feature_in = x.get_shape().as_list()
        self.in_feature_num = int(feature_in / self.num_time_steps_closeness)
        x = tf.reshape(x, [int(batch_size), int(node_num), self.num_time_steps_closeness, self.in_feature_num])
        x = tf.transpose(x, [0, 1, 3, 2])
        x = tf.unstack(x, self.num_time_steps_closeness, 3)
        lstm_output = self.glstm_layer(x, self.num_time_steps_closeness, self.lstm_layer_count)
            # Check the tf version here
        lstm_output = lstm_output[-1]  # tf.stack(outputs, axis=3)
        x = lstm_output
        with tf.name_scope('conv_layers'):
            for i in range(self.conv_layer_num):
                with tf.variable_scope('conv_layer_{}'.format(i)):
                    x = self.residual_layer(x, self.num_hidden, 'relu', 'residual_layer_{0}'.format(i))
                    self.nets[x.name] = x
        with tf.name_scope('output_layer'):
            with tf.variable_scope('output_layer'):
                x = getattr(filter, self.filter)(x, self.laplacian, self.lmax, self.out_feature_num,
                                             self.kernel_num)
        # x = self.activation_function(x, 'tanh')
        return x

    def inference_glstm_period_no_expand_gconv(self, x):
        assert(self.num_time_steps_closeness == self.num_time_steps_period)
        batch_size, node_num, feature_in = x.get_shape().as_list()
        batch_size, node_num, feature_in = int(batch_size), int(node_num), int(feature_in)
        self.in_feature_num = int(feature_in / self.num_time_steps_closeness) 
        print("feature in: {0} num_time_steps: {1}\n".format(feature_in, self.num_time_steps_closeness))
        x = tf.reshape(x, [batch_size, node_num, self.num_time_steps_closeness, self.in_feature_num])
        x = tf.transpose(x, [0, 1, 3, 2])
        x = tf.unstack(x, self.num_time_steps_closeness, 3)
        lstm_output = self.glstm_layer(x, self.num_time_steps_closeness, self.lstm_layer_count)
            # Check the tf version here
        lstm_output = lstm_output[-1]  # tf.stack(outputs, axis=3)
        x = lstm_output
        # output with full connected layer
        with tf.name_scope('conv_layers'):
            for i in range(self.conv_layer_num):
                with tf.variable_scope('conv_layer_{}'.format(i)):
                    x = self.residual_layer(x, self.num_hidden, 'relu', 'residual_layer_{0}'.format(i))
                    self.nets[x.name] = x
        with tf.name_scope('output_layer'):
            with tf.variable_scope('output_layer'):
                x = getattr(filter, self.filter)(x, self.laplacian, self.lmax, self.out_feature_num,
                                             self.kernel_num)
        return x

    def inference_glstm_period_expand(self, x):
        N, M, F = x.get_shape().as_list()
        x = tf.reshape(x, [N, M, F, 1])
        # x = tf.transpose(x, [0, 1, 3, 2])
        x_arr = tf.unstack(x, axis=2)
        X_0 = tf.concat(x_arr[0:self.num_time_steps_closeness * 2], axis=2)
        X_1 = tf.concat(x_arr[self.num_time_steps_closeness * 2: (self.num_time_steps_closeness + self.num_time_steps_period) * 2], axis=2)
        X_2 = tf.concat(x_arr[(self.num_time_steps_closeness + self.num_time_steps_period) * 2: (self.num_time_steps_closeness + self.num_time_steps_period + self.num_time_steps_trend) * 2], axis=2)
        x_arr = [X_0, X_1, X_2]
        num_time_steps = [self.num_time_steps_closeness, self.num_time_steps_period, self.num_time_steps_trend]

        X = None
        for i in range(3):
            with tf.variable_scope('merge_{}'.format(i)):
                x = x_arr[i]
                batch_size, node_num, feature_in = x.get_shape().as_list()
                x = tf.reshape(x, [batch_size, node_num, num_time_steps[i],
                                   int(feature_in / num_time_steps[i])])
                x = tf.transpose(x, [0, 1, 3, 2])
                x = tf.unstack(x, num_time_steps[i], 3)
                outputs = self.glstm_layer(x, num_time_steps[i], self.lstm_layer_count)
                x = outputs[-1]
                x = self.fc_layer(x, self.out_feature_num)
                batch_size, node_num, feature_out = x.get_shape().as_list()
                with tf.variable_scope('weight_{}'.format(i)):
                    w = self._weight_variable([node_num, feature_out])
                if i == 0:
                    X = x * w
                else:
                    X = X + x * w
        x = X
        # x = self.activation_function(x, 'tanh')
        return x

    def inference_glstm_period_expand_gconv1(self, x):
        N, M, F = x.get_shape()
        N, M, F = int(N), int(M), int(F)
        x = tf.reshape(x, [N, M, F, 1])
        # x = tf.transpose(x, [0, 1, 3, 2])
        x_arr = tf.unstack(x, axis=2)
        X_0 = tf.concat(x_arr[0:self.num_time_steps_closeness * 2], axis=2)
        X_1 = tf.concat(x_arr[self.num_time_steps_closeness * 2: (self.num_time_steps_closeness + self.num_time_steps_period) * 2], axis=2)
        X_2 = tf.concat(x_arr[(self.num_time_steps_closeness + self.num_time_steps_period) * 2: (self.num_time_steps_closeness + self.num_time_steps_period + self.num_time_steps_trend) * 2], axis=2)
        x_arr = [X_0, X_1, X_2]
        num_time_steps = [self.num_time_steps_closeness, self.num_time_steps_period, self.num_time_steps_trend]

        X = None
        for i in range(3):
            with tf.variable_scope('merge_{}'.format(i)):
                x = x_arr[i]
                batch_size, node_num, feature_in = x.get_shape().as_list()
                x = tf.reshape(x, [batch_size, node_num, num_time_steps[i],
                                   int(feature_in / num_time_steps[i])])
                x = tf.transpose(x, [0, 1, 3, 2])
                x = tf.unstack(x, num_time_steps[i], 3)
                outputs = self.glstm_layer(x, num_time_steps[i], self.lstm_layer_count)
                x = outputs[-1]
                x = getattr(filter, self.filter)(x, self.laplacian, self.lmax, self.out_feature_num, self.kernel_num)
                batch_size, node_num, feature_out = x.get_shape().as_list()
                with tf.variable_scope('weight_{}'.format(i)):
                    w = self._weight_variable([node_num, feature_out])
                if i == 0:
                    X = x * w
                else:
                    X = X + x * w
        x = X
        # x = self.activation_function(x, 'tanh')
        return x

    def inference_glstm_period_expand_gconv2(self, x):
        N, M, F = x.get_shape()
        N, M, F = int(N), int(M), int(F)
        x = tf.reshape(x, [N, M, F, 1])
        # x = tf.transpose(x, [0, 1, 3, 2])
        x_arr = tf.unstack(x, axis=2)
        X_0 = tf.concat(x_arr[0:self.num_time_steps_closeness * 2], axis=2)
        X_1 = tf.concat(x_arr[self.num_time_steps_closeness * 2: (self.num_time_steps_closeness + self.num_time_steps_period) * 2], axis=2)
        X_2 = tf.concat(x_arr[(self.num_time_steps_closeness + self.num_time_steps_period) * 2: (self.num_time_steps_closeness + self.num_time_steps_period + self.num_time_steps_trend) * 2], axis=2)
        x_arr = [X_0, X_1, X_2]
        num_time_steps = [self.num_time_steps_closeness, self.num_time_steps_period, self.num_time_steps_trend]
        X = None
        output_arr = []
        for i in range(3):
            with tf.variable_scope('merge_{}'.format(i)):
                x = x_arr[i]
                batch_size, node_num, feature_in = x.get_shape().as_list()
                x = tf.reshape(x, [batch_size, node_num, num_time_steps[i],
                                   int(feature_in / num_time_steps[i])])
                x = tf.transpose(x, [0, 1, 3, 2])
                x = tf.unstack(x, num_time_steps[i], 3)
                outputs = self.glstm_layer(x, num_time_steps[i], self.lstm_layer_count)
                x = outputs[-1]
                x = getattr(filter, self.filter)(x, self.laplacian, self.lmax, self.out_feature_num, self.kernel_num)
                output_arr.append(x)
        x = tf.concat(output_arr, axis=2)
        with tf.variable_scope('final'):
            x = getattr(filter, self.filter)(x, self.laplacian, self.lmax, self.out_feature_num, self.kernel_num)
        # x = self.activation_function(x, 'tanh')
        return x

    def inference_glstm_period_expand_gconv3(self, x):
        N, M, F = x.get_shape()
        N, M, F = int(N), int(M), int(F)
        x = tf.reshape(x, [N, M, F, 1])
        # x = tf.transpose(x, [0, 1, 3, 2])
        x_arr = tf.unstack(x, axis=2)
        X_0 = tf.concat(x_arr[0:self.num_time_steps_closeness * 2], axis=2)
        X_1 = tf.concat(x_arr[self.num_time_steps_closeness * 2: (self.num_time_steps_closeness + self.num_time_steps_period) * 2], axis=2)
        X_2 = tf.concat(x_arr[(self.num_time_steps_closeness + self.num_time_steps_period) * 2: (self.num_time_steps_closeness + self.num_time_steps_period + self.num_time_steps_trend) * 2], axis=2)
        x_arr = [X_0, X_1, X_2]
        num_time_steps = [self.num_time_steps_closeness, self.num_time_steps_period, self.num_time_steps_trend]
        X = None
        output_arr = []
        for i in range(3):
            with tf.variable_scope('merge_{}'.format(i)):
                x = x_arr[i]
                batch_size, node_num, feature_in = x.get_shape().as_list()
                x = tf.reshape(x, [batch_size, node_num, num_time_steps[i],
                                   int(feature_in / num_time_steps[i])])
                x = tf.transpose(x, [0, 1, 3, 2])
                x = tf.unstack(x, num_time_steps[i], 3)
                outputs = self.glstm_layer(x, num_time_steps[i], self.lstm_layer_count)
                x = outputs[-1]
                # x = getattr(filter, self.filter)(x, self.laplacian, self.lmax, self.out_feature_num, self.kernel_num)
                output_arr.append(x)
        x = tf.concat(output_arr, axis=2)

        with tf.name_scope('conv_init'):
            with tf.variable_scope('conv_init'):
                x = getattr(filter, self.filter)(x, self.laplacian, self.lmax, self.num_hidden, self.kernel_num)
                self.nets[x.name] = x
                x = self.activation_function(x, 'relu')
                self.nets[x.name] = x
        with tf.name_scope('conv_layers'):
            for i in range(self.conv_layer_num):
                with tf.variable_scope('conv_layer_{}'.format(i)):
                    x = self.residual_layer(x, self.num_hidden, 'relu', 'residual_layer_{0}'.format(i))
                    self.nets[x.name] = x
        with tf.name_scope('output_layer'):
            with tf.variable_scope('output_layer'):
                x = getattr(filter, self.filter)(x, self.laplacian, self.lmax, self.out_feature_num,
                                             self.kernel_num)
        return x

    def glstm_layer(self, x, num_time_step, layer_count):
        with tf.name_scope("gconv_lstm_layer"):
            cell_arr = []
            first_layer_cell = GConvLSTMCell(num_units=self.num_hidden, forget_bias=1.0,
                                 laplacian=self.laplacian, lmax=self.lmax,
                                 feat_in=self.in_feature_num, K=self.kernel_num,
                                 nNode=self.node_num, filter_type=self.filter)
            cell = tf.nn.rnn_cell.DropoutWrapper(first_layer_cell, output_keep_prob=0.8)
            cell_arr.append(cell)
            for i in range(layer_count - 1):
                next_layer_cell = GConvLSTMCell(num_units=self.num_hidden, forget_bias=1.0,
                                      laplacian=self.laplacian, lmax=self.lmax,
                                      feat_in=self.num_hidden, K=self.kernel_num,
                                      nNode=self.node_num, filter_type = self.filter)
                next_layer_cell = tf.nn.rnn_cell.DropoutWrapper(next_layer_cell, output_keep_prob=0.8)
                cell_arr.append(next_layer_cell)
            cell = tf.nn.rnn_cell.MultiRNNCell(cell_arr, state_is_tuple=True)
            outputs, states = tf.nn.static_rnn(cell, x, dtype=tf.float32)
            return outputs

    def fc_layer(self, x, feature_out):
        batch_size, node_num, feature_in = x.get_shape().as_list()
        x = tf.reshape(x, [batch_size * node_num, feature_in])
        with tf.name_scope('weight'):
            w = tf.Variable(tf.random_normal([feature_in, feature_out]))
            b = tf.Variable(tf.random_normal([feature_out]))
        x = tf.matmul(x, w) + b
        x = tf.reshape(x, [batch_size, node_num, feature_out])
        return x

    def activation_function(self, x, activation):
        activation_funcs = {'tanh': lambda x: tf.nn.tanh(x), 'relu': lambda x: tf.nn.relu(x)}
        return activation_funcs[activation](x)

    def residual_layer(self, x, nfilter, activation, name_scope):
        flag = True
        if flag is True:
            x_identity = x
            with tf.variable_scope(name_scope):
                with tf.variable_scope('sublayer0'):
                    with tf.name_scope('filter'):
                        x = getattr(filter, self.filter)(x, self.laplacian, self.lmax, nfilter, self.kernel_num)
                    with tf.name_scope('activation'):
                        x = self.activation_function(x, activation)
                with tf.variable_scope('sublayer1'):
                    with tf.name_scope('filter2'):
                        x = getattr(filter, self.filter)(x, self.laplacian, self.lmax, nfilter, self.kernel_num)
                    with tf.name_scope('merge'):
                        x = x + x_identity
                    with tf.name_scope('activation2'):
                        x = self.activation_function(x, activation)
        else:
            with tf.variable_scope(name_scope):
                with tf.variable_scope('sublayer0nores'):
                    with tf.name_scope('filter'):
                        x = getattr(filter, self.filter)(x, self.laplacian, self.lmax, nfilter, self.kernel_num)
                    with tf.name_scope('activation'):
                        x = self.activation_function(x, activation)
                with tf.variable_scope('sublayer1nores'):
                    with tf.name_scope('filter2'):
                        x = getattr(filter, self.filter)(x, self.laplacian, self.lmax, nfilter, self.kernel_num)
                    with tf.name_scope('activation2'):
                        x = self.activation_function(x, activation)
        return x
