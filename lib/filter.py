#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# created by Xu Wang
# 2018-01-10

import tensorflow as tf
import graph
import numpy as np


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


def fourier_conv(x, L, lmax, Fout, K, W=None):
    # assert K == L.shape[0]  # artificial but useful to compute number of parameters
    K = L.shape[0]
    N, M, Fin = x.get_shape()
    N, M, Fin = int(N), int(M), int(Fin)
    # Fourier basis
    _, U = graph.fourier(L)
    U = tf.constant(U.T, dtype=tf.float32)
    if W is None:
        initial = tf.truncated_normal_initializer(0, 0.1)
        W = tf.get_variable('weights', [M, Fout, Fin], tf.float32, initializer=initial)

    return filter_in_fourier_conv(x, L, Fout, K, U, W)


def cheby_conv(x, L, lmax, feat_out, K, W=None):

    """
    x : [batch_size, N_node, feat_in] - input of each time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    feat_in : number of input feature
    feat_out : number of output feature
    L : laplacian
    lmax : ?
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    """


    nSample, nNode, feat_in = x.get_shape()
    nSample, nNode, feat_in = int(nSample), int(nNode), int(feat_in)
    if W is None:
        initial = tf.truncated_normal_initializer(0, 0.1)
        W = tf.get_variable('weights', [K * feat_in, feat_out], tf.float32, initializer=initial)
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
