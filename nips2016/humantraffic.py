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

cell_rows = 20
cell_cols = 20

class HumanTraffic:

    def __init__(self):
        self.max_val = 0.0
        self.in_matrix = None
        self.out_matrix = None


    def load_data(self, seq_num):
        # load in_matrix
        f1 = open('../../data/in_matrix.txt', 'r')
        in_matrix = []
        for line in f1.readlines():
            in_matrix.append([int(v) for v in line[0:-1].split(' ')])
        in_matrix = np.array(in_matrix)
        in_matrix = in_matrix[1:, 1:]
        f1.close()
        f2 = open('../../data/out_matrix.txt', 'r')
        out_matrix = []
        for line in f2.readlines():
            out_matrix.append([int(v) for v in line[0:-1].split(' ')])
        out_matrix = np.array(out_matrix)
        out_matrix = out_matrix[1:, 1:]

        f2.close()
        edge_matrix = None
        if os.path.isfile('../../data/edge_weight.pkl'):
            pkl_file = open('../../data/edge_weight.pkl', 'rb')
            edge_matrix = pickle.load(pkl_file)
            pkl_file.close()
        else:
            f3 = open('../../data/edge_weight.txt', 'r')
            I = []
            J = []
            V = []

            for line in f3.readlines():
                split_items = line[0:-1].split(' ')
                in_node = int(split_items[0])
                out_node = int(split_items[1])
                weight_array = [int(v.strip()) for v in line[line.index('[') + 1: line.index(']')].split(',')]
                weight_val = np.sum(np.array(weight_array)) * 1.0
                if weight_val > 600:
                    I.append(in_node)
                    J.append(out_node)
                    V.append(math.log(weight_val))
            print(len(V))
            edge_matrix = scipy.sparse.coo_matrix((V, (I, J)), shape=(cell_rows * cell_cols, cell_rows * cell_cols))
            pkl_file = open('../../data/edge_weight.pkl', 'wb')
            pickle.dump(edge_matrix, pkl_file)
            pkl_file.close()

        # replace zero columns
        # in_matrix[:, 196] = in_matrix[:, 196 - 48]
        # out_matrix[:, 196] = out_matrix[:, 196 - 48]
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
            data_labels.append(np.concatenate((in_matrix[:, i+3:i+4], out_matrix[:, i+3:i+4]), axis=1))
            # data_labels.append(in_matrix[:, i + seq_num])
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

        return train_data, validate_data, test_data, train_labels, validate_labels, test_labels, edge_matrix

    def normalize(self, in_matrix, out_matrix):
        # self.max_val = np.amax([np.amax(in_matrix), np.amax(out_matrix)])
        self.max_val = 1000
        in_matrix = in_matrix * 1.0 / self.max_val # (in_matrix * 1.0 - self.max_val / 2) / (self.max_val / 2)
        out_matrix = out_matrix * 1.0 / self.max_val # (out_matrix * 1.0 - self.max_val / 2) / (self.max_val / 2)

        in_matrix[in_matrix>1] = 1
        out_matrix[out_matrix>1] = 1
        # in_matrix = in_matrix * 2 - 1
        # out_matrix = out_matrix * 2 - 1
        #in_matrix = np.array(in_matrix) * 1.0
        #out_matrix = np.array(out_matrix) * 1.0
        #self.in_avg = np.mean(in_matrix, axis=0)
        #self.in_std = np.std(in_matrix, axis=0)
        #self.out_avg = np.mean(out_matrix, axis=0)
        #self.out_std = np.std(out_matrix, axis=0)
        #in_matrix -= self.in_avg
        #in_matrix /= self.in_std
        #out_matrix -= self.out_avg
        #ut_matrix /= self.out_std
        return in_matrix, out_matrix

    def reverse_normalize(self, data):
        # return data * self.in_std + self.in_avg
        return data * self.max_val
        # return (data + 1) /2 * self.max_val


if __name__ == "__main__":
    print("hehe")

