from . import graph

import tensorflow as tf
import sklearn
import scipy.sparse
import numpy as np
import os, time, collections, shutil
from tensorflow.python.ops.rnn_cell_impl import RNNCell
NFEATURES = 28 ** 2
NCLASSES = 10


# Common methods for all models

debug_keys=[]




class cgcnn(base_model):
    """
    Graph CNN which uses the Chebyshev approximation.

    The following are hyper-parameters of graph convolutional layers.
    They are lists, which length is equal to the number of gconv layers.
        F: Number of features.
        K: List of polynomial orders, i.e. filter sizes or number of hopes.
        p: Pooling size.
           Should be 1 (no pooling) or a power of 2 (reduction by 2 at each coarser level).
           Beware to have coarsened enough.

    L: List of Graph Laplacians. Size M x M. One per coarsening level.

    The following are hyper-parameters of fully connected layers.
    They are lists, which length is equal to the number of fc layers.
        M: Number of features per sample, i.e. number of hidden neurons.
           The last layer is the softmax, i.e. M[-1] is the number of classes.
    
    The following are choices of implementation for various blocks.
        filter: filtering operation, e.g. chebyshev5, lanczos2 etc.
        brelu: bias and relu, e.g. b1relu or b2relu.
        pool: pooling, e.g. mpool1.
    
    Training parameters:
        num_epochs:    Number of training epochs.
        learning_rate: Initial learning rate.
        decay_rate:    Base of exponential decay. No decay with 1.
        decay_steps:   Number of steps after which the learning rate decays.
        momentum:      Momentum. 0 indicates no momentum.

    Regularization parameters:
        regularization: L2 regularizations of weights and biases.
        dropout:        Dropout (fc layers): probability to keep hidden neurons. No dropout with 1.
        batch_size:     Batch size. Must divide evenly into the dataset sizes.
        eval_frequency: Number of steps between evaluations.

    Directories:
        dir_name: Name for directories (summaries and model parameters).
    """

    def __init__(self, L, F, K, p, M, _STACK_NUM=1, _nfilter=64, _nres_layer_count=4, filter='chebyshev5', brelu='b1relu', pool='mpool1',
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

        self.build_graph(M_0, np.sum(self.C_0))

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
        L = scipy.sparse.csr_matrix(L)
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
        L = scipy.sparse.csr_matrix(L)
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
        activation_funcs = {'brelu': lambda x:self.brelu(x),
                            'brelu2': lambda x:self.b2relu(x),
                            'tanh' : lambda x:tf.nn.tanh(x)}
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
                            #x1 = self.filter(x_arr[i], self.L[0], nfilter, self.K[0])
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

    # def prediction(self, logits):
    #         """Return the predicted classes."""
    #         with tf.name_scope('prediction'):
    #             prediction = self.# tf.nn.tanh(logits)  # tf.argmax(logits, axis=1)
    #             return prediction

def filter_in_fourier_conv(x, L, Fout, K, U, W):
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

def fourier_conv(x, L, Fout, K, W):
    # assert K == L.shape[0]  # artificial but useful to compute number of parameters
    K = L.shape[0]
    N, M, Fin = x.get_shape()
    N, M, Fin = int(N), int(M), int(Fin)
    # Fourier basis
    _, U = graph.fourier(L)
    U = tf.constant(U.T, dtype=tf.float32)
    # Weights
    # W = self._weight_variable([M, Fout, Fin], regularization=False)
    return filter_in_fourier_conv(x, L, Fout, K, U, W)

def cheby_conv(x, L, lmax, feat_out, K, W):
    '''
    x : [batch_size, N_node, feat_in] - input of each time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    feat_in : number of input feature
    feat_out : number of output feature
    L : laplacian
    lmax : ?
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''
    nSample, nNode, feat_in = x.get_shape()
    nSample, nNode, feat_in = int(nSample), int(nNode), int(feat_in)
    L = graph.rescale_L(L, lmax)  # What is this operation?? --> rescale Laplacian
    L = L.tocoo()

    indices = np.column_stack((L.row, L.col))
    L = tf.SparseTensor(indices, L.data, L.shape)
    L = tf.sparse_reorder(L)

    x0 = tf.transpose(x, perm=[1, 2, 0])  # change it to [nNode, feat_in, nSample]
    x0 = tf.reshape(x0, [nNode, feat_in * nSample])
    x = tf.expand_dims(x0, 0)  # make it [1, nNode, feat_in*nSample]

    def concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    if K > 1:
        x1 = tf.sparse_tensor_dense_matmul(L, x0)
        x = concat(x, x1)

    for k in range(2, K):
        x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0
        x = concat(x, x2)
        x0, x1 = x1, x2

    x = tf.reshape(x, [K, nNode, feat_in, nSample])
    x = tf.transpose(x, perm=[3, 1, 2, 0])
    x = tf.reshape(x, [nSample * nNode, feat_in * K])

    x = tf.matmul(x, W)  # No Bias term?? -> Yes
    out = tf.reshape(x, [nSample, nNode, feat_out])
    return out

_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ('c', 'h'))

class LSTMStateTuple(_LSTMStateTuple):
    __slots__ = ()  # What is this??

    @property
    def dtype(self):
        (c, h) = self
        if not c.dtype == h.dtype:
            raise TypeError("Inconsistent internal state")
        return c.dtype


class gconvLSTMCell(RNNCell):
    def __init__(self, num_units, forget_bias=1.0,
                 state_is_tuple=True, activation=None, reuse=None,
                 laplacian=None, lmax=None, K=None, feat_in=None, nNode=None):

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

            batch_size, nNode, feat_in = inputs.get_shape()

            # The inputs : [batch_size, nNode, feat_in, nTime?] size tensor
            if feat_in is None:
                # Take out the shape of input

                print("hey!")

            feat_out = self._num_units

            if K is None:
                K = 2

            scope = tf.get_variable_scope()
            with tf.variable_scope(scope) as scope:
                try:
                    # Need four diff Wconv weight + for Hidden weight
                    # Wzxt = tf.get_variable("Wzxt", [K * feat_in, feat_out], dtype=tf.float32,
                    #                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    # Wixt = tf.get_variable("Wixt", [K * feat_in, feat_out], dtype=tf.float32,
                    #                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    # Wfxt = tf.get_variable("Wfxt", [K * feat_in, feat_out], dtype=tf.float32,
                    #                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    # Woxt = tf.get_variable("Woxt", [K * feat_in, feat_out], dtype=tf.float32,
                    #                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    #
                    # Wzht = tf.get_variable("Wzht", [K * feat_out, feat_out], dtype=tf.float32,
                    #                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    # Wiht = tf.get_variable("Wiht", [K * feat_out, feat_out], dtype=tf.float32,
                    #                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    # Wfht = tf.get_variable("Wfht", [K * feat_out, feat_out], dtype=tf.float32,
                    #                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    # Woht = tf.get_variable("Woht", [K * feat_out, feat_out], dtype=tf.float32,
                    #                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
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
                    # Wzxt = tf.get_variable("Wzxt", [K * feat_in, feat_out], dtype=tf.float32,
                    #                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    # Wixt = tf.get_variable("Wixt", [K * feat_in, feat_out], dtype=tf.float32,
                    #                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    # Wfxt = tf.get_variable("Wfxt", [K * feat_in, feat_out], dtype=tf.float32,
                    #                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    # Woxt = tf.get_variable("Woxt", [K * feat_in, feat_out], dtype=tf.float32,
                    #                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    #
                    # Wzht = tf.get_variable("Wzht", [K * feat_out, feat_out], dtype=tf.float32,
                    #                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    # Wiht = tf.get_variable("Wiht", [K * feat_out, feat_out], dtype=tf.float32,
                    #                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    # Wfht = tf.get_variable("Wfht", [K * feat_out, feat_out], dtype=tf.float32,
                    #                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    # Woht = tf.get_variable("Woht", [K * feat_out, feat_out], dtype=tf.float32,
                    #                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

                    Wzxt = tf.get_variable("Wzxt", [nNode, feat_out, feat_in], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    Wixt = tf.get_variable("Wixt", [nNode, feat_out, feat_in], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    Wfxt = tf.get_variable("Wfxt", [nNode, feat_out, feat_in], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    Woxt = tf.get_variable("Woxt", [nNode, feat_out, feat_in], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

                    Wzht = tf.get_variable("Wzht", [nNode, feat_out, feat_in], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    Wiht = tf.get_variable("Wiht", [nNode, feat_out, feat_in], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    Wfht = tf.get_variable("Wfht", [nNode, feat_out, feat_in], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    Woht = tf.get_variable("Woht", [nNode, feat_out, feat_in], dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

                bzt = tf.get_variable("bzt", [feat_out])
                bit = tf.get_variable("bit", [feat_out])
                bft = tf.get_variable("bft", [feat_out])
                bot = tf.get_variable("bot", [feat_out])

                # gconv Calculation
                # zxt = cheby_conv(inputs, laplacian, lmax, feat_out, K, Wzxt)
                # zht = cheby_conv(h, laplacian, lmax, feat_out, K, Wzht)
                zxt = fourier_conv(inputs, laplacian, feat_out, K, Wzxt)
                zht = fourier_conv(h, laplacian, feat_out, K, Wzht)
                zt = zxt + zht + bzt
                zt = tf.tanh(zt)

                # ixt = cheby_conv(inputs, laplacian, lmax, feat_out, K, Wixt)
                # iht = cheby_conv(h, laplacian, lmax, feat_out, K, Wiht)
                ixt = fourier_conv(inputs, laplacian, feat_out, K, Wixt)
                iht = fourier_conv(h, laplacian, feat_out, K, Wiht)
                it = ixt + iht + bit
                it = tf.sigmoid(it)

                # fxt = cheby_conv(inputs, laplacian, lmax, feat_out, K, Wfxt)
                # fht = cheby_conv(h, laplacian, lmax, feat_out, K, Wfht)
                fxt = fourier_conv(inputs, laplacian, feat_out, K, Wfxt)
                fht = fourier_conv(h, laplacian, feat_out, K, Wfht)
                ft = fxt + fht + bft
                ft = tf.sigmoid(ft)

                # oxt = cheby_conv(inputs, laplacian, lmax, feat_out, K, Woxt)
                # oht = cheby_conv(h, laplacian, lmax, feat_out, K, Woht)
                oxt = fourier_conv(inputs, laplacian, feat_out, K, Woxt)
                oht = fourier_conv(h, laplacian, feat_out, K, Woht)
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


class GconvModel(base_model):
    """
    Defined:
        Placeholder
        Model architecture
        Train / Test function
    """

    def __init__(self, L, F, K, p, M, _STACK_NUM=1, _nfilter=64, _nres_layer_count=4, filter='chebyshev5', brelu='b1relu', pool='mpool1',
                 num_epochs=20, learning_rate=0.1, decay_rate=0.95, decay_steps=None, momentum=0.9,
                 regularization=0, dropout=0, batch_size=100, eval_frequency=200,
                 dir_name='', C_0=[6], model_name='ResGNN'):
        super().__init__()

        self.model_type = 'glstm'
        self.batch_size = batch_size

        self.feat_in = 2
        self.num_time_steps = C_0 // 2
        self.feat_out = 2
        ##Need to import laplacian, lmax
        self.laplacian = L
        self.lmax = graph.lmax(self.laplacian)

        self.num_hidden = 32
        self.num_kernel = 1
        self.L, self.F, self.K, self.p, self.M = L, F, K, p, M
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        # self.filter = getattr(self, filter)
        # self.brelu = getattr(self, brelu)
        # self.pool = getattr(self, pool)

        # Build the computational graph.
        self.C_0 = C_0
        self.reuse = None
        M_0 = L.shape[0]
        self.num_node = M_0
        self.build_graph(M_0, np.sum(self.C_0))

    '''
    def residual_layer(self, x, nfilter, activation, name_scope):
        flag = self.model_name == 'ResGNN'
        if flag is True:
            x_identity = x
            with tf.variable_scope(name_scope):
                with tf.variable_scope('sublayer0'):
                    with tf.name_scope('filter'):
                        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
                        x = cheby_conv(x, self.L, self.lmax, self.feat_out, self.K, output_variable['weight'])
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
    '''

    def _inference(self, x, dropout):
        A, B, C = x.get_shape()
        x = tf.reshape(x, [int(A), int(B), self.num_time_steps, 2])
        x = tf.transpose(x, [0, 1, 3, 2])
        self.rnn_input = x
        self.rnn_input_seq = tf.unstack(self.rnn_input, self.num_time_steps, 3)
        with tf.variable_scope("gconv_model", reuse=self.reuse) as sc:
            if self.model_type == 'lstm':
                cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_hidden, forget_bias=1.0)
                n_classes = self.num_node
                output_variable = {
                    'weight': tf.Variable(tf.random_normal([self.num_hidden, n_classes])),
                    'bias': tf.Variable(tf.random_normal([n_classes]))}
            elif self.model_type == 'glstm':
                cell = None
                cell2 = None
                with tf.variable_scope('filter'):
                    cell = gconvLSTMCell(num_units=self.num_hidden, forget_bias=1.0,
                                     laplacian=self.laplacian, lmax=self.lmax,
                                     feat_in=self.feat_in, K=self.num_kernel,
                                     nNode=self.num_node)
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.8)
                with tf.variable_scope('result'):
                    cell2 = gconvLSTMCell(num_units=self.num_hidden, forget_bias=1.0,
                                     laplacian=self.laplacian, lmax=self.lmax,
                                     feat_in=self.num_hidden, K=self.num_kernel,
                                     nNode=self.num_node)
                    cell2 = tf.nn.rnn_cell.DropoutWrapper(cell2, output_keep_prob=0.8)
                cell = tf.nn.rnn_cell.MultiRNNCell([cell, cell2], state_is_tuple=True)
                #  [K * feat_in, feat_out]
                output_variable = None
                with tf.variable_scope('output'):
                    output_variable = {
                    'weight': tf.Variable(tf.random_normal([self.num_node, self.feat_out, self.num_hidden])),
                    'bias': tf.Variable(tf.random_normal([self.feat_out]))}
                    print("test here")

            else:
                raise Exception("[!] Unkown model type: {}".format(self.model_type))



            outputs, states = tf.nn.static_rnn(cell, self.rnn_input_seq, dtype=tf.float32)

            # cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(cell, output_keep_prob=0.8)
            # Check the tf version here
            outputs = outputs[-1]# tf.stack(outputs, axis=3)
            outputs = tf.reshape(outputs, [-1, self.num_node, self.num_hidden])
            y = fourier_conv(outputs, self.L, self.feat_out, self.K, output_variable['weight'])


        return tf.nn.tanh(y)

