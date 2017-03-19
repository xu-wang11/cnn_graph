#!/bin/bash python
# coding: utf-8
# Script Name: humantraffic.py
# Author     : Xu Wang
# Date       : Mar 15th, 2017
# Description: predicte human traffic using graph cnn

import numpy as np
from scipy.sparse.csr import csr_matrix

def load_data():
    # load in_matrix
    f1 = open('in_matrix.txt', 'r')
    in_matrix = []
    for line in f1.readlines():
        in_matrix.append([int(v) for v in line[0:-1].split(' ')])
    in_matrix = np.array(in_matrix)
    in_matrix = in_matrix[1:, 1:]
    f1.close()
    f2 = open('out_matrix.txt', 'r')
    out_matrix = []
    for line in f2.readlines():
        out_matrix.append([int(v) for v in line[0:-1].split(' ')])
    out_matrix = np.array(out_matrix)
    out_matrix = out_matrix[1:, 1:]

    f2.close()
    f3 = open('edge_weight.txt', 'r')
    edge_matrix = []
    for line in f3.readlines():
        edge_matrix.append([int(v) for v in line[0:-1].split(' ')])
    edge_matrix = np.array(edge_matrix)
    edge_matrix = edge_matrix[edge_matrix[:, 0] != 0]
    edge_matrix = edge_matrix[edge_matrix[:, 1] != 0]

    # to sparse matrix
    edge_matrix = csr_matrix(edge_matrix[:, 2], (edge_matrix[:, 0], edge_matrix[:, 1]))


    return None



if __name__ == "__main__":

