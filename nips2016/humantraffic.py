#!/bin/bash python
# coding: utf-8
# Script Name: humantraffic.py
# Author     : Xu Wang
# Date       : Mar 15th, 2017
# Description: predicte human traffic using graph cnn

import numpy as np
from scipy.sparse.csr import csr_matrix
import scipy
import math
import pickle
import os
from scipy.io import loadmat
from stldecompose import decompose

class HumanTraffic:

    def __init__(self, data_set_path):
        self.max_val = 0.0
        self.in_matrix = None
        self.out_matrix = None
        self.dataset_path = data_set_path


    def load_ln_data_period(self, seq_num, seq_num_period=1, seq_num_trend=1, datafile='ln_data.mat'):
        ln_data = loadmat(os.path.join(self.dataset_path, datafile))
        edge_matrix = loadmat(os.path.join(self.dataset_path, 'edge_matrix.mat'))['edge_matrix']
        # edge_matrix = edge_matrix.multiply(edge_matrix>=400)
        # edge_matrix.eliminate_zeros()
        # edge_matrix.data = np.log10(edge_matrix.data)
        edge_matrix = csr_matrix(edge_matrix)
        in_matrix = ln_data['inmatrix']
        out_matrix = ln_data['outmatrix']
        in_matrix = in_matrix[:, 0:1344]
        out_matrix = out_matrix[:, 0:1344]
        in_matrix, out_matrix = self.normalize(in_matrix, out_matrix)
        # in_matrix, out_matrix = self.normalize_seasonal_decompose(in_matrix, out_matrix)


        # train data test data
        data_samples = []
        data_labels = []
        for i in range((48 * 7 - seq_num) + int(seq_num_trend / 2), in_matrix.shape[1] - seq_num):
            x1 = np.concatenate((in_matrix[:, i:i+seq_num], out_matrix[:, i:i + seq_num]), axis=1)
            x2 = np.concatenate((in_matrix[:, (i + seq_num - 48) - int(seq_num_period / 2): (i + seq_num - 48) + int(seq_num_period / 2) + (seq_num_period % 2)], out_matrix[:, (i + seq_num - 48) - int(seq_num_period / 2): (i + seq_num - 48) + int(seq_num_period / 2) + (seq_num_period % 2)]), axis=1)
            x3 = np.concatenate((in_matrix[:, (i + seq_num - 48 * 7) - int(seq_num_trend / 2): (i + seq_num - 48 * 7) + int(seq_num_trend / 2) + (seq_num_trend % 2)], out_matrix[:, (i + seq_num - 48 * 7) - int(seq_num_trend / 2): (i + seq_num - 48 * 7) + int(seq_num_trend / 2) + (seq_num_trend % 2)]), axis=1)
            data_samples.append(np.concatenate((x1, x2, x3), axis=1))
            data_labels.append(np.concatenate((in_matrix[:, i + seq_num:i + seq_num + 1], out_matrix[:, i + seq_num:i + seq_num + 1]), axis=1))

        # for i in range((48 * 7 - seq_num) + int(seq_num_trend / 2), in_matrix.shape[1] - seq_num):
        #     x1 = np.concatenate((in_matrix[:, i:i+seq_num], out_matrix[:, i:i + seq_num]), axis=1)
        #     x2 = np.concatenate((in_matrix[:, (i + seq_num - 48) - int(seq_num_period / 2): (i + seq_num - 48) + int(seq_num_period / 2) + (seq_num_period % 2)], out_matrix[:, (i + seq_num - 48) - int(seq_num_period / 2): (i + seq_num - 48) + int(seq_num_period / 2) + (seq_num_period % 2)]), axis=1)
        #     x3 = np.concatenate((in_matrix[:, (i + seq_num - 48 * 7) - int(seq_num_trend / 2): (i + seq_num - 48 * 7) + int(seq_num_trend / 2) + (seq_num_trend % 2)], out_matrix[:, (i + seq_num - 48 * 7) - int(seq_num_trend / 2): (i + seq_num - 48 * 7) + int(seq_num_trend / 2) + (seq_num_trend % 2)]), axis=1)
        #     data_samples.append(np.concatenate((x1, x2, x3), axis=1))
        #     data_labels.append(np.concatenate((in_matrix[:, i + seq_num:i + seq_num + 1], out_matrix[:, i + seq_num:i + seq_num + 1]), axis=1))

        # for i in range(48 - seq_num, in_matrix.shape[1] - seq_num):
        #     x1 = np.concatenate((in_matrix[:, i:i+seq_num], out_matrix[:, i:i + seq_num]), axis=1)
        #     x2 = np.concatenate((in_matrix[:, i+seq_num - 48: i + seq_num - 1: 48], out_matrix[:, i + seq_num - 48: i + seq_num - 1: 48]), axis=1)
        #     # x3 = np.concatenate((in_matrix[:, i + 3 - (48 * 7): i + 2: 48 * 7], out_matrix[:, i + 3 - 48 * 7: i + 2: 48 * 7]), axis=1)
        #     data_samples.append(np.concatenate((x1, x2), axis=1))
        #     data_labels.append(np.concatenate((in_matrix[:, i + seq_num:i + seq_num + 1], out_matrix[:, i + seq_num:i + seq_num + 1]), axis=1))
        #     # data_labels.append(in_matrix[:, i + seq_num])
        data_samples = np.array(data_samples)
        data_labels = np.array(data_labels)
        print('shape of data_samples: {0}'.format(data_samples.shape[0]))
        # shuffle_array = np.random.permutation(data_samples.shape[0])
        # shuffle_array = pickle.load(open(os.path.join(self.dataset_path, 'shuffle_array.pkl'), 'rb'))
        # data_samples = data_samples[shuffle_array]
        # data_labels = data_labels[shuffle_array]
        total_row = data_samples.shape[0]
        train_row = int(total_row * 0.75)
        validate_row = int(total_row * 0.125)

        train_data = data_samples[0:train_row, :]
        validate_data = data_samples[train_row: train_row + validate_row, :]
        test_data = data_samples[train_row + validate_row:, :]
        train_labels = data_labels[0: train_row, :]
        validate_labels = data_labels[train_row: train_row + validate_row, :]
        test_labels = data_labels[train_row + validate_row:, :]
        # normalized

        return train_data, validate_data, test_data, train_labels, validate_labels, test_labels, edge_matrix

    def load_split_ln_data_period(self, seq_num, seq_num_period=1, seq_num_trend=1, datafile='ln_data.mat'):
        ln_data = loadmat(os.path.join(self.dataset_path, datafile))
        edge_matrix = loadmat(os.path.join(self.dataset_path, 'edge_matrix.mat'))['edge_matrix']
        # edge_matrix = edge_matrix.multiply(edge_matrix>=400)
        # edge_matrix.eliminate_zeros()
        # edge_matrix.data = np.log10(edge_matrix.data)
        edge_matrix = csr_matrix(edge_matrix)
        split_in_matrix = ln_data['split_in_traffic']
        split_out_matrix = ln_data['split_out_traffic']
        target_in_matrix = ln_data['inmatrix']
        target_out_matrix = ln_data['outmatrix']

        target_in_matrix, target_out_matrix, split_in_matrix, split_out_matrix = self.split_normalize_seasonal_decompose(target_in_matrix, target_out_matrix, split_in_matrix, split_out_matrix)

        # train data test data
        data_samples = []
        data_labels = []
        for i in range((48 * 7 - seq_num) + int(seq_num_trend / 2), target_in_matrix.shape[1] - seq_num):
            x1 = np.concatenate((split_in_matrix[:, i:i+seq_num], split_out_matrix[:, i:i + seq_num]), axis=1)
            x2 = np.concatenate((split_in_matrix[:, (i + seq_num - 48) - int(seq_num_period / 2): (i + seq_num - 48) + int(seq_num_period / 2) + (seq_num_period % 2)], split_out_matrix[:, (i + seq_num - 48) - int(seq_num_period / 2): (i + seq_num - 48) + int(seq_num_period / 2) + (seq_num_period % 2)]), axis=1)
            x3 = np.concatenate((split_in_matrix[:, (i + seq_num - 48 * 7) - int(seq_num_trend / 2): (i + seq_num - 48 * 7) + int(seq_num_trend / 2) + (seq_num_trend % 2)], split_out_matrix[:, (i + seq_num - 48 * 7) - int(seq_num_trend / 2): (i + seq_num - 48 * 7) + int(seq_num_trend / 2) + (seq_num_trend % 2)]), axis=1)
            x1 = np.reshape(x1, newshape=(x1.shape[0], x1.shape[1] * x1.shape[2]))
            x2 = np.reshape(x2, newshape=(x2.shape[0], x2.shape[1] * x2.shape[2]))
            x2 = np.reshape(x3, newshape=(x3.shape[0], x3.shape[1] * x3.shape[2]))
            data_samples.append(np.concatenate((x1, x2, x3), axis=1))
            data_labels.append(np.concatenate((target_in_matrix[:, i+seq_num:i+seq_num + 1], target_out_matrix[:, i+seq_num:i+seq_num + 1]), axis=1))
        data_samples = np.array(data_samples)
        data_labels = np.array(data_labels)
        print('shape of data_samples: {0}'.format(data_samples.shape[0]))
        # shuffle_array = np.random.permutation(data_samples.shape[0])
        # shuffle_array = pickle.load(open(os.path.join(self.dataset_path, 'shuffle_array.pkl'), 'rb'))
        # shuffle_array = np.random.permutation(data_samples.shape[0])
        # data_samples = data_samples[shuffle_array]
        # data_labels = data_labels[shuffle_array]
        total_row = data_samples.shape[0]
        train_row = int(total_row * 0.75)
        validate_row = int(total_row * 0.125)

        train_data = data_samples[0:train_row, :]
        validate_data = data_samples[train_row: train_row + validate_row, :]
        test_data = data_samples[train_row + validate_row:, :]
        train_labels = data_labels[0: train_row, :]
        validate_labels = data_labels[train_row: train_row + validate_row, :]
        test_labels = data_labels[train_row + validate_row:, :]
        # normalized
        return train_data, validate_data, test_data, train_labels, validate_labels, test_labels, edge_matrix

    def load_split_ln_data(self, seq_num, datafile='split_lndata_street.mat'):
        # ln_data = loadmat(os.path.join(self.dataset_path, 'split_ln_data_2.mat'))
        ln_data = loadmat(os.path.join(self.dataset_path, datafile))
        edge_matrix = loadmat(os.path.join(self.dataset_path, 'edge_matrix.mat'))['edge_matrix']
        edge_matrix = csr_matrix(edge_matrix)
        # edge_matrix = edge_matrix.multiply(edge_matrix>=400)
        # edge_matrix.eliminate_zeros()
        # edge_matrix.data = np.log10(edge_matrix.data)
        split_in_matrix = ln_data['split_in_traffic']
        split_out_matrix = ln_data['split_out_traffic']

        target_in_matrix = ln_data['inmatrix']
        target_out_matrix = ln_data['outmatrix']


        target_in_matrix, target_out_matrix, split_in_matrix, split_out_matrix = self.split_normalize_seasonal_decompose(target_in_matrix, target_out_matrix, split_in_matrix, split_out_matrix)
        # target_in_matrix, target_out_matrix = self.normalize(target_in_matrix, target_out_matrix)
        # split_in_matrix = split_in_matrix * 1.0 / self.max_val
        # split_out_matrix = split_out_matrix * 1.0 / self.max_val

        # train data test data
        data_samples = []
        data_labels = []
        for i in range(target_in_matrix.shape[1] - seq_num):
            x1 = np.concatenate((split_in_matrix[:, i:i+seq_num], split_out_matrix[:, i:i + seq_num]), axis=1)
            x1 = np.reshape(x1, newshape=(x1.shape[0], x1.shape[1] * x1.shape[2]))
            data_samples.append(x1)
            data_labels.append(np.concatenate((target_in_matrix[:, i+seq_num:i+seq_num + 1], target_out_matrix[:, i+seq_num:i+seq_num + 1]), axis=1))
         # data_labels.append(in_matrix[:, i + seq_num])

        data_samples = np.array(data_samples)
        print(data_samples.shape)
        # data_samples = np.reshape(data_samples, newshape=(data_samples.shape[0], data_samples.shape[1], data_samples.shape[2] * data_samples.shape[3]))
        # data_samples = np.array(data_samples)
        data_labels = np.array(data_labels)
        print('shape of data_samples: {0}'.format(data_samples.shape[0]))
        total_row = data_samples.shape[0]
        train_row = int(total_row * 0.75)
        validate_row = int(total_row * 0.125)
        train_data = data_samples[0:train_row, :]
        validate_data = data_samples[train_row: train_row + validate_row, :]
        test_data = data_samples[train_row + validate_row:, :]
        train_labels = data_labels[0: train_row, :]
        validate_labels = data_labels[train_row: train_row + validate_row, :]
        test_labels = data_labels[train_row + validate_row:, :]
        # normalized

        return train_data, validate_data, test_data, train_labels, validate_labels, test_labels, edge_matrix


    def load_bj_data(self, seq_num):
        ln_data = loadmat(os.path.join(self.dataset_path, 'bj_data.mat'))
        # edge_matrix = loadmat(os.path.join(self.dataset_path, 'edge_matrix.mat'))['edge_matrix']
        # edge_matrix = edge_matrix.multiply(edge_matrix>=400)
        # edge_matrix.eliminate_zeros()
        # edge_matrix.data = np.log10(edge_matrix.data)
        in_matrix = ln_data['inmatrix']
        out_matrix = ln_data['outmatrix']
        # train data test data
        data_samples = []
        in_matrix, out_matrix = self.normalize(in_matrix, out_matrix)

        data_labels = []
        for i in range(in_matrix.shape[1] - seq_num):
            data_samples.append(np.concatenate((in_matrix[:, i:i+seq_num], out_matrix[:, i:i + seq_num]), axis=1))
            data_labels.append(np.concatenate((in_matrix[:, i+seq_num:i+seq_num + 1], out_matrix[:, i+seq_num:i+seq_num + 1]), axis=1))
            # data_labels.append(in_matrix[:, i + seq_num])
        data_samples = np.array(data_samples)
        data_labels = np.array(data_labels)
        print('shape of data_samples: {0}'.format(data_samples.shape[0]))
        # shuffle_array = np.random.permutation(data_samples.shape[0])
        # shuffle_array = pickle.load(open(os.path.join(self.dataset_path, 'shuffle_array.pkl'), 'rb'))
        # data_samples = data_samples[shuffle_array]
        # data_labels = data_labels[shuffle_array]
        total_row = data_samples.shape[0]
        train_row = int(total_row * 0.75)
        validate_row = int(total_row * 0.125)

        train_data = data_samples[0:train_row, :]
        validate_data = data_samples[train_row: train_row + validate_row, :]
        test_data = data_samples[train_row + validate_row:, :]
        train_labels = data_labels[0: train_row, :]
        validate_labels = data_labels[train_row: train_row + validate_row, :]
        test_labels = data_labels[train_row + validate_row:, :]
        # normalized

        return train_data, validate_data, test_data, train_labels, validate_labels, test_labels, None


    def load_bj_clus_data(self, seq_num):
        ln_data = loadmat(os.path.join(self.dataset_path, 'bj_clus.mat'))
        edge_matrix = loadmat(os.path.join(self.dataset_path, 'edge_matrix.mat'))['edge_matrix']
        edge_matrix = csr_matrix(edge_matrix)
        # edge_matrix = edge_matrix.multiply(edge_matrix>=400)
        # edge_matrix.eliminate_zeros()
        # edge_matrix.data = np.log10(edge_matrix.data)
        in_matrix = ln_data['inmatrix']
        out_matrix = ln_data['outmatrix']
        # train data test data
        data_samples = []
        in_matrix, out_matrix = self.normalize(in_matrix, out_matrix)

        data_labels = []
        for i in range(in_matrix.shape[1] - seq_num):
            x_input = np.concatenate((in_matrix[:, i:i+seq_num], out_matrix[:, i:i + seq_num]), axis=1)
            x_output = np.concatenate((in_matrix[:, i+seq_num:i+seq_num + 1], out_matrix[:, i+seq_num:i+seq_num + 1]), axis=1)
            nan_num = np.count_nonzero(x_input == -1) + np.count_nonzero(x_output == -1)
            if nan_num == 0:
                data_samples.append(x_input)
                data_labels.append(x_output)
            # data_labels.append(in_matrix[:, i + seq_num])
        data_samples = np.array(data_samples)
        data_labels = np.array(data_labels)
        print('shape of data_samples: {0}'.format(data_samples.shape[0]))
        shuffle_array = np.random.permutation(data_samples.shape[0])
        # shuffle_array = pickle.load(open(os.path.join(self.dataset_path, 'shuffle_array.pkl'), 'rb'))
        data_samples = data_samples[shuffle_array]
        data_labels = data_labels[shuffle_array]
        total_row = data_samples.shape[0]
        train_row = int(total_row * 0.75)
        validate_row = int(total_row * 0.125)

        train_data = data_samples[0:train_row, :]
        validate_data = data_samples[train_row: train_row + validate_row, :]
        test_data = data_samples[train_row + validate_row:, :]
        train_labels = data_labels[0: train_row, :]
        validate_labels = data_labels[train_row: train_row + validate_row, :]
        test_labels = data_labels[train_row + validate_row:, :]
        # normalized

        return train_data, validate_data, test_data, train_labels, validate_labels, test_labels, edge_matrix



    def load_bj_data_period_trend(self, seq_num, seq_num_period=1, seq_num_trend=1):
        ln_data = loadmat(os.path.join(self.dataset_path, 'bj_data.mat'))
        # edge_matrix = loadmat(os.path.join(self.dataset_path, 'edge_matrix.mat'))['edge_matrix']
        # edge_matrix = edge_matrix.multiply(edge_matrix>=400)
        # edge_matrix.eliminate_zeros()
        # edge_matrix.data = np.log10(edge_matrix.data)
        in_matrix = ln_data['inmatrix']
        out_matrix = ln_data['outmatrix']
        in_matrix, out_matrix = self.normalize(in_matrix, out_matrix)

        # train data test data
        data_samples = []
        data_labels = []
        for i in range((48 * 7 - seq_num) + int(seq_num_trend / 2), in_matrix.shape[1] - seq_num):
            x1 = np.concatenate((in_matrix[:, i:i+seq_num], out_matrix[:, i:i + seq_num]), axis=1)
            x2 = np.concatenate((in_matrix[:, (i + seq_num - 48) - int(seq_num_period / 2): (i + seq_num - 48) + int(seq_num_period / 2) + (seq_num_period % 2)], out_matrix[:, (i + seq_num - 48) - int(seq_num_period / 2): (i + seq_num - 48) + int(seq_num_period / 2) + (seq_num_period % 2)]), axis=1)
            x3 = np.concatenate((in_matrix[:, (i + seq_num - 48 * 7) - int(seq_num_trend / 2): (i + seq_num - 48 * 7) + int(seq_num_trend / 2) + (seq_num_trend % 2)], out_matrix[:, (i + seq_num - 48 * 7) - int(seq_num_trend / 2): (i + seq_num - 48 * 7) + int(seq_num_trend / 2) + (seq_num_trend % 2)]), axis=1)
            data_samples.append(np.concatenate((x1, x2, x3), axis=1))
            data_labels.append(np.concatenate((in_matrix[:, i + seq_num:i + seq_num + 1], out_matrix[:, i + seq_num:i + seq_num + 1]), axis=1))
            # data_labels.append(in_matrix[:, i + seq_num])
        data_samples = np.array(data_samples)
        data_labels = np.array(data_labels)
        print('shape of data_samples: {0}'.format(data_samples.shape[0]))
        # shuffle_array = np.random.permutation(data_samples.shape[0])
        # shuffle_array = pickle.load(open(os.path.join(self.dataset_path, 'shuffle_array.pkl'), 'rb'))
        # data_samples = data_samples[shuffle_array]
        # data_labels = data_labels[shuffle_array]
        total_row = data_samples.shape[0]
        train_row = int(total_row * 0.75)
        validate_row = int(total_row * 0.125)

        train_data = data_samples[0:train_row, :]
        validate_data = data_samples[train_row: train_row + validate_row, :]
        test_data = data_samples[train_row + validate_row:, :]
        train_labels = data_labels[0: train_row, :]
        validate_labels = data_labels[train_row: train_row + validate_row, :]
        test_labels = data_labels[train_row + validate_row:, :]
        # normalized
        return train_data, validate_data, test_data, train_labels, validate_labels, test_labels, None


    def load_bj_clus_period_trend(self, seq_num):
        ln_data = loadmat(os.path.join(self.dataset_path, 'bj_clus.mat'))
        edge_matrix = loadmat(os.path.join(self.dataset_path, 'edge_matrix.mat'))['edge_matrix']
        edge_matrix = csr_matrix(edge_matrix)
        # edge_matrix = edge_matrix.multiply(edge_matrix>=400)
        # edge_matrix.eliminate_zeros()
        # edge_matrix.data = np.log10(edge_matrix.data)
        in_matrix = ln_data['inmatrix']
        out_matrix = ln_data['outmatrix']
        in_matrix, out_matrix = self.normalize(in_matrix, out_matrix)

        # train data test data
        data_samples = []
        data_labels = []
        for i in range(48 * 2 - seq_num, in_matrix.shape[1] - seq_num):
            x1 = np.concatenate((in_matrix[:, i:i+seq_num], out_matrix[:, i:i + seq_num]), axis=1)
            x2 = np.concatenate((in_matrix[:, i+seq_num - 96: i + seq_num - 1: 48], out_matrix[:, i + seq_num - 96: i + seq_num - 1: 48]), axis=1)
            # x3 = np.concatenate((in_matrix[:, i + seq_num - (48 * 7): i + seq_num - 1: 48 * 7], out_matrix[:, i + seq_num - 48 * 7: i + seq_num - 1: 48 * 7]), axis=1)
            input_data = np.concatenate((x1, x2), axis=1)
            output_data = np.concatenate((in_matrix[:, i + seq_num:i + seq_num + 1], out_matrix[:, i + seq_num:i + seq_num + 1]), axis=1)
            nan_num = np.count_nonzero(input_data == -1) + np.count_nonzero(output_data == -1)
            if nan_num == 0:
                data_samples.append(input_data)
                data_labels.append(output_data)
            # data_labels.append(in_matrix[:, i + seq_num])
        data_samples = np.array(data_samples)
        data_labels = np.array(data_labels)
        print('shape of data_samples: {0}'.format(data_samples.shape[0]))
        shuffle_array = np.random.permutation(data_samples.shape[0])
        # shuffle_array = pickle.load(open(os.path.join(self.dataset_path, 'shuffle_array.pkl'), 'rb'))
        data_samples = data_samples[shuffle_array]
        data_labels = data_labels[shuffle_array]
        total_row = data_samples.shape[0]
        train_row = int(total_row * 0.75)
        validate_row = int(total_row * 0.125)

        train_data = data_samples[0:train_row, :]
        validate_data = data_samples[train_row: train_row + validate_row, :]
        test_data = data_samples[train_row + validate_row:, :]
        train_labels = data_labels[0: train_row, :]
        validate_labels = data_labels[train_row: train_row + validate_row, :]
        test_labels = data_labels[train_row + validate_row:, :]
        # normalized

        return train_data, validate_data, test_data, train_labels, validate_labels, test_labels, edge_matrix

    # only use nodes contains edges
    def load_unisolate_data(self, seq_num):
        ln_data = loadmat(os.path.join(self.dataset_path, 'ln_data.mat'))
        in_matrix = ln_data['inmatrix']
        out_matrix = ln_data['outmatrix']
        edge_matrix = loadmat(os.path.join(self.dataset_path, 'edge_matrix.mat'))['edge_matrix']
        edge_matrix = edge_matrix.todense()

        edge_sum = np.sum(edge_matrix, axis=1)
        node_list = np.where(edge_sum != 0)
        remove_node = np.where(edge_sum == 0)[0]
        edge_matrix = edge_matrix[np.ix_(node_list[0], node_list[0])]
        remove_in_matrix = in_matrix[remove_node, :]
        in_matrix = in_matrix[node_list[0], :]
        remove_out_matrix = out_matrix[remove_node, :]
        out_matrix = out_matrix[node_list[0], :]
        print('current nnz is {0}\n'.format(np.count_nonzero(edge_matrix)))
        # edge_matrix[edge_matrix < 400] = 0
        print('current nnz is {0}\n'.format(np.count_nonzero(edge_matrix)))
        # to symmetrical
        edge_matrix = edge_matrix + np.transpose(edge_matrix)
        node_list = np.where(np.sum(edge_matrix, axis=1) != 0)[0]
        remove_node = np.where(np.sum(edge_matrix, axis=1) == 0)[0]
        edge_matrix = edge_matrix[np.ix_(node_list, node_list)]
        remove_in_matrix = np.append(remove_in_matrix, in_matrix[remove_node, :], axis=0)
        in_matrix = in_matrix[node_list]
        remove_out_matrix = np.append(remove_out_matrix, out_matrix[remove_node, :], axis=0)

        out_matrix = out_matrix[node_list]
        # edge_matrix[edge_matrix > 0] = 1
        # edge_matrix = np.log10(edge_matrix + 1)
        edge_matrix = csr_matrix(edge_matrix)
        in_matrix, out_matrix = self.normalize(in_matrix, out_matrix)
        # train data test data
        data_samples = []
        data_labels = []
        remove_labels = []
        for i in range(in_matrix.shape[1] - seq_num):
            data_samples.append(np.concatenate((in_matrix[:, i:i+seq_num], out_matrix[:, i:i + seq_num]), axis=1))
            data_labels.append(np.concatenate((in_matrix[:, i+seq_num:i+seq_num + 1], out_matrix[:, i+seq_num:i+seq_num + 1]), axis=1))
            # data_labels.append(in_matrix[:, i + seq_num])
        for i in range(remove_in_matrix.shape[1] - seq_num):
           remove_labels.append(np.concatenate((remove_in_matrix[:, i+seq_num:i+seq_num + 1], remove_out_matrix[:, i+seq_num:i+seq_num + 1]), axis=1))
        data_samples = np.array(data_samples)
        data_labels = np.array(data_labels)
        remove_labels = np.array(remove_labels)
        print('shape of data_samples: {0}'.format(data_samples.shape[0]))
        # shuffle_array = np.random.permutation(data_samples.shape[0])
        shuffle_array = pickle.load(open(os.path.join(self.dataset_path, 'shuffle_array.pkl'), 'rb'))
        data_samples = data_samples[shuffle_array]
        data_labels = data_labels[shuffle_array]
        total_row = data_samples.shape[0]
        train_row = int(total_row * 0.75)
        validate_row = int(total_row * 0.125)

        train_data = data_samples[0:train_row, :]
        validate_data = data_samples[train_row: train_row + validate_row, :]
        test_data = data_samples[train_row + validate_row:, :]
        train_labels = data_labels[0: train_row, :]
        validate_labels = data_labels[train_row: train_row + validate_row, :]
        test_labels = data_labels[train_row + validate_row:, :]
        remove_labels = remove_labels[train_row+validate_row:, :]
        # normalized

        return train_data, validate_data, test_data, train_labels, validate_labels, test_labels, edge_matrix, remove_labels

    def load_data(self, seq_num):
        # load in_matrix
        # f1 = open(os.path.join(self.dataset_path, 'in_matrix.txt'), 'r')
        # in_matrix = []
        # for line in f1.readlines():
        #     in_matrix.append([int(v) for v in line[0:-1].split(' ')])
        # in_matrix = np.array(in_matrix)
        # in_matrix = in_matrix[1:, 1:]
        # f1.close()
        # f2 = open(os.path.join(self.dataset_path, 'out_matrix.txt'), 'r')
        # out_matrix = []
        # for line in f2.readlines():
        #     out_matrix.append([int(v) for v in line[0:-1].split(' ')])
        # out_matrix = np.array(out_matrix)
        # out_matrix = out_matrix[1:, 1:]
        #
        # f2.close()
        # edge_matrix = None
        # if os.path.isfile(os.path.join(self.dataset_path, 'edge_weight.pkl')):
        #     pkl_file = open(os.path.join(self.dataset_path, 'edge_weight.pkl'), 'rb')
        #     edge_matrix = pickle.load(pkl_file)
        #     pkl_file.close()
        # else:
        #     f3 = open(os.path.join(self.dataset_path, 'edge_weight.txt'), 'r')
        #     I = []
        #     J = []
        #     V = []

        #     for line in f3.readlines():
        #         split_items = line[0:-1].split(' ')
        #         in_node = int(split_items[0])
        #         out_node = int(split_items[1])
        #         weight_array = [int(v.strip()) for v in line[line.index('[') + 1: line.index(']')].split(',')]
        #         weight_val = np.sum(np.array(weight_array)) * 1.0
        #         if weight_val > 600:
        #             I.append(in_node)
        #             J.append(out_node)
        #             V.append(math.log(weight_val))
        #     print(len(V))
        #     edge_matrix = scipy.sparse.coo_matrix((V, (I, J)), shape=(in_matrix.shape[0], in_matrix.shape[0]))
        #     pkl_file = open(os.path.join(self.dataset_path, 'edge_weight.pkl'), 'wb')
        #     pickle.dump(edge_matrix, pkl_file)
        #     pkl_file.close()

        # replace zero columns
        # in_matrix[:, 196] = in_matrix[:, 196 - 48]
        # out_matrix[:, 196] = out_matrix[:, 196 - 48]
        # normalize

        # max_val = np.amax([np.amax(in_matrix), np.amax(out_matrix)])
        # max_val = 500.0
        # in_matrix = in_matrix / max_val  # (in_matrix - max_val / 2) / (max_val / 2)
        # out_matrix = out_matrix / max_val  # (out_matrix - max_val / 2) / (max_val /2)
        ln_data = loadmat(os.path.join(self.dataset_path, 'ln_data.mat'))
        edge_matrix = loadmat(os.path.join(self.dataset_path, 'edge_matrix.mat'))['edge_matrix']
        edge_matrix = edge_matrix + edge_matrix.transpose()
        edge_matrix = edge_matrix.todense()
        edge_matrix = np.multiply(np.ones((400,400)), (edge_matrix>=700))
        #edge_matrix = edge_matrix + np.eye(400)
        #print(edge_matrix[1,])
        edge_matrix = csr_matrix(edge_matrix)
        # edge_matrix.eliminate_zeros()
        # edge_matrix.data = np.log10(edge_matrix.data)

        in_matrix = ln_data['inmatrix']
        out_matrix = ln_data['outmatrix']
        in_matrix, out_matrix = self.normalize(in_matrix, out_matrix)

        # train data test data
        data_samples = []
        data_labels = []
        for i in range(in_matrix.shape[1] - seq_num):
            data_samples.append(np.concatenate((in_matrix[:, i:i+seq_num], out_matrix[:, i:i + seq_num]), axis=1))
            data_labels.append(np.concatenate((in_matrix[:, i+seq_num:i+seq_num + 1], out_matrix[:, i+seq_num:i+seq_num + 1]), axis=1))
            # data_labels.append(in_matrix[:, i + seq_num])
        data_samples = np.array(data_samples)
        data_labels = np.array(data_labels)
        print('shape of data_samples: {0}'.format(data_samples.shape[0]))
        shuffle_array = np.random.permutation(data_samples.shape[0])
        # shuffle_array = pickle.load(open(os.path.join(self.dataset_path, 'shuffle_array.pkl'), 'rb'))
        print(shuffle_array)
        # pickle.dump(shuffle_array, open(os.path.join(self.dataset_path, 'shuffle_array_forzy.pkl'), 'wb'), 2)
        data_samples = data_samples[shuffle_array]
        data_labels = data_labels[shuffle_array]
        total_row = data_samples.shape[0]
        train_row = int(total_row * 0.75)
        validate_row = int(total_row * 0.125)

        train_data = data_samples[0:train_row, :]
        validate_data = data_samples[train_row: train_row + validate_row, :]
        test_data = data_samples[train_row + validate_row:, :]
        train_labels = data_labels[0: train_row, :]
        validate_labels = data_labels[train_row: train_row + validate_row, :]
        test_labels = data_labels[train_row + validate_row:, :]
        # normalized

        return train_data, validate_data, test_data, train_labels, validate_labels, test_labels, edge_matrix


    def load_lstm_data(self, seq_num, neighbor_num, datafile):
        ln_data = loadmat(os.path.join(self.dataset_path, datafile))
        in_matrix = ln_data['inmatrix']
        out_matrix = ln_data['outmatrix']
        in_matrix, out_matrix = self.normalize(in_matrix, out_matrix)
        data_collection = []
        for kk in range(in_matrix.shape[0]):
            neighbor = np.array([kk])
            # train data test data
            data_samples = []
            data_labels = []
            for i in range(in_matrix.shape[1] - seq_num):
                data_samples.append(np.concatenate((in_matrix[neighbor, i:i+seq_num], out_matrix[neighbor, i:i + seq_num]), axis=0))
                yy = [in_matrix[kk, i + seq_num], out_matrix[kk, i + seq_num]]
                data_labels.append(yy)
                # data_labels.append(in_matrix[:, i + seq_num])
            data_samples = np.array(data_samples)
            data_labels = np.array(data_labels)
            print('shape of data_samples: {0}'.format(data_samples.shape[0]))
            total_row = data_samples.shape[0]
            train_row = int(total_row * 0.85)
            validate_row = 0
            train_data = data_samples[0:train_row, :]
            # validate_data = data_samples[train_row: train_row + validate_row, :]
            test_data = data_samples[train_row + validate_row:, :]
            train_labels = data_labels[0: train_row, :]
            # validate_labels = data_labels[train_row: train_row + validate_row, :]
            test_labels = data_labels[train_row + validate_row:, :]
            data_collection.append({'train_data':train_data, 'test_data':test_data, 'train_labels': train_labels, 'test_labels': test_labels})

    
        return data_collection

    def load_lndata_street(self, seq_num, datafile='lndata_street.mat'):
        ln_data = loadmat(os.path.join(self.dataset_path, datafile))
        edge_matrix = loadmat(os.path.join(self.dataset_path, 'edge_matrix.mat'))['edge_matrix']
        # edge_matrix = edge_matrix + edge_matrix.transpose()
        # edge_matrix = edge_matrix.todense()
        # edge_matrix = np.multiply(np.ones((400,400)), (edge_matrix>=700))
        #edge_matrix = edge_matrix + np.eye(400)
        #print(edge_matrix[1,])
        edge_matrix = csr_matrix(edge_matrix)
        # edge_matrix.eliminate_zeros()
        # edge_matrix.data = np.log10(edge_matrix.data)

        in_matrix = ln_data['inmatrix']
        out_matrix = ln_data['outmatrix']
        # in_matrix, out_matrix = self.normalize(in_matrix, out_matrix)
        in_matrix, out_matrix = self.normalize_seasonal_decompose(in_matrix, out_matrix)

        # train data test data
        data_samples = []
        data_labels = []
        for i in range(in_matrix.shape[1] - seq_num):
            data_samples.append(np.concatenate((in_matrix[:, i:i+seq_num], out_matrix[:, i:i + seq_num]), axis=1))
            data_labels.append(np.concatenate((in_matrix[:, i+seq_num:i+seq_num + 1], out_matrix[:, i+seq_num:i+seq_num + 1]), axis=1))
            # data_labels.append(in_matrix[:, i + seq_num])
        data_samples = np.array(data_samples)
        data_labels = np.array(data_labels)
        print('shape of data_samples: {0}'.format(data_samples.shape[0]))
        # shuffle_array = np.random.permutation(data_samples.shape[0])
        # shuffle_array = pickle.load(open(os.path.join(self.dataset_path, 'shuffle_array.pkl'), 'rb'))
        # shuffle_array = np.random.permutation(data_samples.shape[0])
        # data_samples = data_samples[shuffle_array]
        # data_labels = data_labels[shuffle_array]
        total_row = data_samples.shape[0]
        train_row = int(total_row * 0.75)
        validate_row = int(total_row * 0.125)

        train_data = data_samples[0:train_row, :]
        validate_data = data_samples[train_row: train_row + validate_row, :]
        test_data = data_samples[train_row + validate_row:, :]
        train_labels = data_labels[0: train_row, :]
        validate_labels = data_labels[train_row: train_row + validate_row, :]
        test_labels = data_labels[train_row + validate_row:, :]
        # normalized

        return train_data, validate_data, test_data, train_labels, validate_labels, test_labels, edge_matrix

    def split_normalize_seasonal_decompose(self, in_matrix, out_matrix, split_in_matrix, split_out_matrix):
        in_matrix = in_matrix.astype(np.float32)
        out_matrix = in_matrix.astype(np.float32)
        split_in_matrix = split_in_matrix.astype(np.float32)
        split_out_matrix = split_out_matrix.astype(np.float32)
        region_num = in_matrix.shape[0]
        for i in range(region_num):
            x_slow = split_in_matrix[i, :, 0]
            x_fast = split_in_matrix[i, :, 1]
            stl_slow = decompose(x_slow, period=48)
            split_in_matrix[i, :, 0] = stl_slow.resid
            stl_fast = decompose(x_fast, period=48)
            split_in_matrix[i, :, 1] = stl_fast.resid
            in_matrix[i, :] = in_matrix[i, :] - stl_slow.seasonal - stl_slow.trend - stl_fast.seasonal - stl_fast.trend

            x_slow = split_out_matrix[i, :, 0]
            x_fast = split_out_matrix[i, :, 1]
            stl_slow = decompose(x_slow, period=48)
            split_out_matrix[i, :, 0] = stl_slow.resid
            stl_fast = decompose(x_fast, period=48)
            split_out_matrix[i, :, 1] = stl_fast.resid
            out_matrix[i, :] = out_matrix[i, :] - stl_slow.seasonal - stl_slow.trend - stl_fast.seasonal - stl_fast.trend
        self.max_val = np.amax([np.amax(in_matrix), np.amax(out_matrix), np.amax(split_in_matrix), np.amax(split_out_matrix)])
        self.min_val = np.amin([np.amin(in_matrix), np.amin(out_matrix), np.amin(split_in_matrix), np.amin(split_out_matrix)])
        in_matrix = in_matrix * 1.0 / (self.max_val - self.min_val)
        out_matrix = out_matrix * 1.0 / (self.max_val - self.min_val)
        split_in_matrix = split_in_matrix * 1.0 / (self.max_val - self.min_val)
        split_out_matrix = split_out_matrix * 1.0 / (self.max_val - self.min_val)
        return in_matrix, out_matrix, split_in_matrix, split_out_matrix


    def normalize_seasonal_decompose(self, in_matrix, out_matrix):
        in_matrix = in_matrix.astype(np.float32)
        out_matrix = in_matrix.astype(np.float32)
        self.seasonal_in_matrix = np.zeros(in_matrix.shape)
        self.seasonal_out_matrix = np.zeros(out_matrix.shape)
        for i in range(in_matrix.shape[0]):
            x = in_matrix[i, :]
            stl = decompose(x, period=48)
            self.seasonal_in_matrix[i, :] = stl.seasonal
            in_matrix[i, :] = stl.resid
            x2 = out_matrix[i, :]
            stl = decompose(x2, period=48)
            self.seasonal_out_matrix[i, :] = stl.seasonal
            out_matrix[i, :] = stl.resid
        self.max_val = np.amax([np.amax(in_matrix), np.amax(out_matrix)])
        self.min_val = np.amin([np.amin(in_matrix), np.amin(out_matrix)])
        # self.max_val = 1000
        in_matrix = in_matrix * 1.0 / (self.max_val - self.min_val) # (in_matrix * 1.0 - self.max_val / 2) / (self.max_val / 2)
        out_matrix = out_matrix * 1.0 / (self.max_val - self.min_val) # (out_matrix * 1.0 - self.max_val / 2) / (self.max_val / 2)
        return in_matrix, out_matrix

    def normalize(self, in_matrix, out_matrix):
        self.min_val = 0
        self.max_val = np.amax([np.amax(in_matrix), np.amax(out_matrix)])
        # self.max_val = 1000
        in_matrix = in_matrix * 1.0 / self.max_val # (in_matrix * 1.0 - self.max_val / 2) / (self.max_val / 2)
        out_matrix = out_matrix * 1.0 / self.max_val # (out_matrix * 1.0 - self.max_val / 2) / (self.max_val / 2)

        # in_matrix[in_matrix>1] = 1
        # out_matrix[out_matrix>1] = 1
        # in_matrix = in_matrix * 2.0 - 1.0
        # out_matrix = out_matrix * 2.0 - 1.0
        # in_matrix = np.array(in_matrix) * 1.0
        # out_matrix = np.array(out_matrix) * 1.0
        # self.in_avg = np.mean(in_matrix, axis=0)
        # self.in_std = np.std(in_matrix, axis=0)
        # self.out_avg = np.mean(out_matrix, axis=0)
        # self.out_std = np.std(out_matrix, axis=0)
        # in_matrix -= self.in_avg
        # in_matrix /= self.in_std
        # out_matrix -= self.out_avg
        # ut_matrix /= self.out_std
        return in_matrix, out_matrix

    def reverse_normalize(self, data):
        # return data * self.in_std + self.in_avg
        print(self.min_val)
        return data * (self.max_val - self.min_val)
        # return (data + 1.0) / 2.0 * self.max_val


if __name__ == "__main__":
    h = HumanTraffic('../../data/lndata_filter')
    h.load_unisolate_data(3)

