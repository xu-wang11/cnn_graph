#!/bin/bash python
# coding: utf-8
# Script Name: humantraffic.py
# Author     : Xu Wang
# Date       : Mar 15th, 2017
# Description: predicte human traffic using graph cnn

import numpy as np
from scipy.sparse.csr import csr_matrix

class HumanTraffic:

    def __init__(self):
        self.max_val = 0.0
        self.in_matrix = None
        self.out_matrix = None


    def load_data(self, seq_num):
        # load in_matrix
        f1 = open('../data/humanflow/in_matrix.txt', 'r')
        in_matrix = []
        for line in f1.readlines():
            in_matrix.append([int(v) for v in line[0:-1].split(' ')])
        in_matrix = np.array(in_matrix)
        in_matrix = in_matrix[1:, 1:]
        f1.close()
        f2 = open('../data/humanflow/out_matrix.txt', 'r')
        out_matrix = []
        for line in f2.readlines():
            out_matrix.append([int(v) for v in line[0:-1].split(' ')])
        out_matrix = np.array(out_matrix)
        out_matrix = out_matrix[1:, 1:]

        f2.close()
        '''
        f3 = open('edge_weight.txt', 'r')
        edge_matrix = []
        for line in f3.readlines():
            edge_matrix.append([int(v) for v in line[0:-1].split(' ')])
        edge_matrix = np.array(edge_matrix)
        edge_matrix = edge_matrix[edge_matrix[:, 0] != 0]
        edge_matrix = edge_matrix[edge_matrix[:, 1] != 0]

        # to sparse matrix
        edge_matrix = csr_matrix(edge_matrix[:, 2], (edge_matrix[:, 0], edge_matrix[:, 1]))
        '''
        # replace zero columns
        in_matrix[:, 196] = in_matrix[:, 196 - 48]
        out_matrix[:, 196] = out_matrix[:, 196 - 48]
        # normalize

        # max_val = np.amax([np.amax(in_matrix), np.amax(out_matrix)])
        # max_val = 500.0
        # in_matrix = in_matrix / max_val  # (in_matrix - max_val / 2) / (max_val / 2)
        # out_matrix = out_matrix / max_val  # (out_matrix - max_val / 2) / (max_val /2)

        in_matrix, out_matrix = self.normalize(in_matrix, out_matrix)

        # train data test data
        data_samples = []
        data_labels = []
        for i in range(in_matrix.shape[1] - seq_num):
            data_samples.append(np.concatenate((in_matrix[:, i:i+seq_num], out_matrix[:, i:i + seq_num]), axis=1))
            # data_labels.append(np.concatenate((in_matrix[:, i+3:i+4], out_matrix[:, i+3:i+4]), axis=1))
            data_labels.append(in_matrix[:, i + seq_num])
        data_samples = np.array(data_samples)
        data_labels = np.array(data_labels)
        total_row = data_samples.shape[0]
        train_row = int(total_row * 0.6)
        validate_row = int(total_row * 0.2)

        train_data = data_samples[0:train_row, :]
        validate_data = data_samples[train_row: train_row + validate_row, :]
        test_data = data_samples[train_row + validate_row:, :]
        train_labels = data_labels[0: train_row, :]
        validate_labels = data_labels[train_row: train_row + validate_row, :]
        test_labels = data_labels[train_row + validate_row:, :]
        # normalized

        return train_data, validate_data, test_data, train_labels, validate_labels, test_labels

    def normalize(self, in_matrix, out_matrix):
        self.max_val = np.amax([np.amax(in_matrix), np.amax(out_matrix)])
        # self.max_val = 1000
        in_matrix = in_matrix / self.max_val # (in_matrix * 1.0 - self.max_val / 2) / (self.max_val / 2)
        out_matrix = out_matrix / self.max_val # (out_matrix * 1.0 - self.max_val / 2) / (self.max_val / 2)
        # in_matrix = in_matrix * 2 - 1
        # out_matrix = out_matrix * 2 - 1
        # in_matrix[in_matrix>1] = 1
        # out_matrix[out_matrix>1] = 1

        return in_matrix, out_matrix

    def reverse_normalize(self, data):

        return self.max_val * data


if __name__ == "__main__":
    print("hehe")

