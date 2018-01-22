#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# created by Xu Wang
# 2018-01-09
import tensorflow as tf
import numpy as np
from . import graph
from .graph_model import GraphModel
from scipy.sparse import csr_matrix


class GraphConv(GraphModel):

    def __init__(self, L, F, K, p, M, _STACK_NUM=1, _nfilter=64, _nres_layer_count=4, filter='chebyshev5',
                 brelu='b1relu', pool='mpool1',
                 num_epochs=20, learning_rate=0.1, decay_rate=0.95, decay_steps=None, momentum=0.9,
                 regularization=0, dropout=0, batch_size=100, eval_frequency=200,
                 dir_name='', C_0=[6], model_name='ResGNN'):
        super().__init__()

        self.nfilter = _nfilter
        self.nres_layer_count = _nres_layer_count
        self.stack_num = _STACK_NUM
        self.model_name = model_name
        # Verify the consistency w.r.t. the number of layers.
        assert len(L) >= len(F) == len(K) == len(p)
        assert np.all(np.array(p) >= 1)
        p_log2 = np.where(np.array(p) > 1, np.log2(p), 0)
        assert np.all(np.mod(p_log2, 1) == 0)  # Powers of 2.
        assert len(L) >= 1 + np.sum(p_log2)  # Enough coarsening levels for pool sizes.
        assert _STACK_NUM > 0
        # Keep the useful Laplacians only. May be zero.
        M_0 = L[0].shape[0]
        j = 0
        self.L = []
        for pp in p:
            self.L.append(L[j])
            j += int(np.log2(pp)) if pp > 1 else 0
        L = self.L

        # Print information about NN architecture.
        Ngconv = len(p)
        Nfc = len(M)
        print('NN architecture')
        print('  input: M_0 = {}'.format(M_0))
        for i in range(Ngconv):
            print('  layer {0}: cgconv{0}'.format(i + 1))
            print('    representation: M_{0} * F_{1} / p_{1} = {2} * {3} / {4} = {5}'.format(
                i, i + 1, L[i].shape[0], F[i], p[i], L[i].shape[0] * F[i] // p[i]))
            F_last = F[i - 1] if i > 0 else 1
            print('    weights: F_{0} * F_{1} * K_{1} = {2} * {3} * {4} = {5}'.format(
                i, i + 1, F_last, F[i], K[i], F_last * F[i] * K[i]))
            if brelu == 'b1relu':
                print('    biases: F_{} = {}'.format(i + 1, F[i]))
            elif brelu == 'b2relu':
                print('    biases: M_{0} * F_{0} = {1} * {2} = {3}'.format(
                    i + 1, L[i].shape[0], F[i], L[i].shape[0] * F[i]))
        for i in range(Nfc):
            name = 'logits (softmax)' if i == Nfc - 1 else 'fc{}'.format(i + 1)
            print('  layer {}: {}'.format(Ngconv + i + 1, name))
            print('    representation: M_{} = {}'.format(Ngconv + i + 1, M[i]))
            M_last = M[i - 1] if i > 0 else M_0 if Ngconv == 0 else L[-1].shape[0] * F[-1] // p[-1]
            print('    weights: M_{} * M_{} = {} * {} = {}'.format(
                Ngconv + i, Ngconv + i + 1, M_last, M[i], M_last * M[i]))
            print('    biases: M_{} = {}'.format(Ngconv + i + 1, M[i]))

        # Store attributes and bind operations.
        self.L, self.F, self.K, self.p, self.M = L, F, K, p, M
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        self.filter = getattr(self, filter)
        self.brelu = getattr(self, brelu)
        self.pool = getattr(self, pool)

        # Build the computational graph.
        self.C_0 = C_0

        self.build_graph(M_0, np.sum(self.C_0), 2)

    def filter_in_fourier(self, x, L, Fout, K, U, W):
        # TODO: N x F x M would avoid the permutations
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        x = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        # Transform to Fourier domain
        x = tf.reshape(x, [M, Fin * N])  # M x Fin*N
        x = tf.matmul(U, x)  # M x Fin*N
        x = tf.reshape(x, [M, Fin, N])  # M x Fin x N
        # Filter
        x = tf.matmul(W, x)  # for each feature
        x = tf.transpose(x)  # N x Fout x M
        x = tf.reshape(x, [N * Fout, M])  # N*Fout x M
        # Transform back to graph domain
        x = tf.matmul(x, U)  # N*Fout x M
        x = tf.reshape(x, [N, Fout, M])  # N x Fout x M
        return tf.transpose(x, perm=[0, 2, 1])  # N x M x Fout

    def fourier(self, x, L, Fout, K):
        # assert K == L.shape[0]  # artificial but useful to compute number of parameters
        K = L.shape[0]
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Fourier basis
        _, U = graph.fourier(L)
        U = tf.constant(U.T, dtype=tf.float32)
        # Weights
        W = self._weight_variable([M, Fout, Fin], regularization=False)
        return self.filter_in_fourier(x, L, Fout, K, U, W)

    def chebyshev2(self, x, L, Fout, K):
        """
        Filtering with Chebyshev interpolation
        Implementation: numpy.

        Data: x of size N x M x F
            N: number of signals
            M: number of vertices
            F: number of features per signal per vertex
        """
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian. Copy to not modify the shared L.
        L = csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        # Transform to Chebyshev basis
        x = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x = tf.reshape(x, [M, Fin * N])  # M x Fin*N

        def chebyshev(x):
            return graph.chebyshev(L, x, K)

        x = tf.py_func(chebyshev, [x], [tf.float32])[0]  # K x M x Fin*N
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
        x = tf.reshape(x, [N * M, Fin * K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature.
        W = self._weight_variable([Fin * K, Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def chebyshev5(self, x, L, Fout, K):
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin * N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N

        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N

        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
        x = tf.reshape(x, [N * M, Fin * K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin * K, Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def b1relu(self, x):
        """Bias and ReLU. One bias per filter."""
        N, M, F = x.get_shape()
        # b = self._bias_variable([1, 1, int(F)], regularization=False)
        # batch_norm = tf.cond(self.is_train,
        #                      lambda: tf.contrib.layers.batch_norm(x + b, activation_fn=tf.nn.relu, is_training=True,
        #                                                           reuse=None, scope="train_norm"),
        #                      lambda: tf.contrib.layers.batch_norm(x + b, activation_fn=tf.nn.relu, is_training=False,
        #                                                           reuse=True, name="test_norm"))
        return tf.nn.relu(x)

    def b1tanh(self, x):
        """Bias and ReLU. One bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=False)
        return tf.nn.tanh(x + b)

    def b2relu(self, x):
        """Bias and ReLU. One bias per vertex per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, int(M), int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def mpool1(self, x, p):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.max_pool(x, ksize=[1, p, 1, 1], strides=[1, p, 1, 1], padding='SAME')
            # tf.maximum
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def apool1(self, x, p):
        """Average pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.avg_pool(x, ksize=[1, p, 1, 1], strides=[1, p, 1, 1], padding='SAME')
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def fc(self, x, Mout, relu=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout], regularization=True)
        b = self._bias_variable([Mout], regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x

    def activation_function(self, x, activation):
        activation_funcs = {'brelu': lambda x: self.brelu(x),
                            'brelu2': lambda x: self.b2relu(x),
                            'tanh': lambda x: tf.nn.tanh(x)}
        return activation_funcs[activation](x)

    def residual_layer(self, x, nfilter, activation, name_scope):
        flag = self.model_name == 'ResGNN'
        if flag is True:
            x_identity = x
            with tf.variable_scope(name_scope):
                with tf.variable_scope('sublayer0'):
                    with tf.name_scope('filter'):
                        x = self.filter(x, self.L[0], nfilter, self.K[0])
                    with tf.name_scope('activation'):
                        x = self.activation_function(x, activation)
                with tf.variable_scope('sublayer1'):
                    with tf.name_scope('filter2'):
                        x = self.filter(x, self.L[0], nfilter, self.K[0])
                    with tf.name_scope('merge'):
                        x = x + x_identity
                    with tf.name_scope('activation2'):
                        x = self.activation_function(x, activation)


        else:
            with tf.variable_scope(name_scope):
                with tf.variable_scope('sublayer0nores'):
                    with tf.name_scope('filter'):
                        x = self.filter(x, self.L[0], nfilter, self.K[0])
                    with tf.name_scope('activation'):
                        x = self.activation_function(x, activation)
                with tf.variable_scope('sublayer1nores'):
                    with tf.name_scope('filter2'):
                        x = self.filter(x, self.L[0], nfilter, self.K[0])
                    with tf.name_scope('activation2'):
                        x = self.activation_function(x, activation)
        return x

        # add some comments

    def _inference(self, x, dropout):
        # Graph convolutional layers.
        # x = tf.expand_dims(x, 2)  # N x M x F=1
        if self.stack_num == 1:
            return self.residual_network(x)
        else:
            print("stack dim")
            # x = tf.expand_dims(x, 3)
            N, M, F = x.get_shape()
            N, M, F = int(N), int(M), int(F)
            x = tf.reshape(x, [N, M, np.sum(self.C_0), 1])
            # x = tf.transpose(x, [0, 1, 3, 2])
            x_arr = tf.unstack(x, axis=2)
            X_0 = tf.concat(x_arr[0:12], axis=2)
            X_1 = tf.concat(x_arr[12:16], axis=2)
            nfilter = self.nfilter
            x_arr = [X_0, X_1]
            with tf.variable_scope('final_merge'):
                X = None
                for i in range(self.stack_num):
                    with tf.variable_scope('VC_{0}'.format(i)):
                        with tf.name_scope('C_{0}'.format(i)):
                            x1 = self.residual_network(x_arr[i])
                            x1 = tf.nn.relu(x1)
                            # x1 = self.filter(x_arr[i], self.L[0], nfilter, self.K[0])
                            N, M, F = x1.get_shape()
                        with tf.variable_scope('W_{0}'.format(i)):
                            w1 = self._weight_variable([M, F])
                        if i == 0:
                            X = x1 * w1
                        else:
                            X = X + x1 * w1
            # with tf.variable_scope('resnet'):
            #    X = self.residual_network(X)
            return X

    def residual_network(self, x):
        nfilter = self.nfilter
        nres_layer_count = self.nres_layer_count
        active_func = 'brelu'
        # the first layer is to convert N * M * F  to
        with tf.variable_scope('conv_init'):
            with tf.name_scope('filter'):
                x = self.filter(x, self.L[0], nfilter, self.K[0])
                self.nets[x.name] = x
            with tf.name_scope('activation'):
                x = self.activation_function(x, active_func)
                self.nets[x.name] = x
        # residual layer
        for i in range(nres_layer_count):
            x = self.residual_layer(x, nfilter, active_func, 'residual_layer_{0}'.format(i))
            self.nets[x.name] = x

        with tf.variable_scope('convN'):
            with tf.name_scope('filter'):
                x = self.filter(x, self.L[0], 2, self.K[0])
                self.nets[x.name] = x

        N, M, F = x.get_shape()
        # x = tf.reshape(x, [int(N), int(M * F)])  # N x M
        print(self.nets.keys())
        return x
