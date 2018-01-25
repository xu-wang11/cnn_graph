# coding: utf-8

# In[1]:


# In[ ]:
import sys, os

sys.path.insert(0, '..')
sys.path.insert(0, '/home/csi/Git/HumanFlowPrediction/cnn_graph/lib/')
from lib import graph, coarsening, utils, gconv_lstm

import numpy as np
import time
from nips2016 import humantraffic
from tensorflow.python import debug as tf_debug
# get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import itertools
import traceback


# notification
sys.path.insert(0, '../../')

np.random.seed(2017)

# 配置显存大小
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# config = tf.ConfigProto(
#         device_count = {'GPU': 0}
#     )
# sess = tf.Session(config=config)

# In[ ]:

flags = tf.app.flags
FLAGS = flags.FLAGS
print("come here")
# Graphs.
flags.DEFINE_integer('number_edges', 8, 'Graph: minimum number of edges per vertex.')
flags.DEFINE_string('metric', 'euclidean', 'Graph: similarity measure (between features).')
# TODO: change cgcnn for combinatorial Laplacians.
flags.DEFINE_bool('normalized_laplacian', True, 'Graph Laplacian: normalized.')
flags.DEFINE_integer('coarsening_levels', 4, 'Number of coarsened graphs.')

# Directories.
flags.DEFINE_string('dir_data', os.path.join('..', 'data', 'mnist'), 'Directory to store data.')

print('current path is {0}'.format(os.getcwd()))
def sparse_matrix_element_wise_max(A, B):
    BisBigger = A - B
    BisBigger.data = np.where(BisBigger.data < 0, 1, 0)
    return A - A.multiply(BisBigger) + B.multiply(BisBigger)


def grid_graph(m, corners=False):
    z = graph.grid(m)
    dist, idx = graph.distance_sklearn_metrics(z, k=FLAGS.number_edges, metric=FLAGS.metric)
    A = graph.adjacency(dist, idx)

    # Connections are only vertical or horizontal on the grid.
    # Corner vertices are connected to 2 neightbors only.
    if corners:
        import scipy.sparse
        A = A.toarray()
        A[A < A.max() / 1.5] = 0
        A = scipy.sparse.csr_matrix(A)
        print('{} edges'.format(A.nnz))

    print("{} > {} edges".format(A.nnz // 2, FLAGS.number_edges * m ** 2 // 2))
    return A

A = grid_graph(32, corners=False)
L = graph.laplacian(A, normalized=True)  # [graph.laplacian(A, normalized=True) for A in graphs]

seq_num_closeness_choice = [3, 4, 5]  #seq_num
seq_num_period_choice = [3]  #seq_num
seq_num_trend_choice = [3]  #seq_num
learning_rate_choice = [0.005, 0.01, 0.03]  # learning_rate
filter_num_choice = [32]  # filter_num
kernel_num_choice = [2]  # kernel_num
layer_count_choice = [1, 2, 3, 4] # layer count
lstm_layer_count_choice = [1, 2]
filter_choice = ['fourier_conv']  # filter
infer_methods = ['inference_glstm', 'inference_gconv', 'inference_gconv_period_no_expand', 'inference_gconv_period_expand', 'inference_glstm_gconv', 'inference_glstm_period_expand',
                'inference_glstm_period_expand_gconv1', 'inference_glstm_period_expand_gconv2', 'inference_glstm_period_expand_gconv3']


# seq_num_closeness_choice = [5]  #seq_num
# seq_num_period_choice = [3]  #seq_num
# seq_num_trend_choice = [3]  #seq_num
# learning_rate_choice = [0.03]  # learning_rate
# filter_num_choice = [32]  # filter_num
# kernel_num_choice = [2]  # kernel_num
# layer_count_choice = [4] # layer count
# lstm_layer_count_choice = [3]
# filter_choice = ['fourier_conv']  # filter
# infer_methods = ['inference_glstm_period_expand_gconv3']

for params_instance in itertools.product(seq_num_closeness_choice, seq_num_period_choice, seq_num_trend_choice, learning_rate_choice, filter_num_choice, kernel_num_choice, layer_count_choice, filter_choice, lstm_layer_count_choice, infer_methods):
    print(params_instance)
    try:
        # sess = tf.Session(config=config)
        DATA_SET_PATH = '../../data/bjtaxi'

        seq_num_closeness = params_instance[0]
        seq_num_period = seq_num_closeness  # params_instance[1]
        seq_num_trend = seq_num_closeness  # params_instance[2]
        ht = humantraffic.HumanTraffic(DATA_SET_PATH)
        if "expand" in params_instance[9]:
            train_data, val_data, test_data, train_labels, val_labels, test_labels, A1 = ht.load_bj_data_period_trend(seq_num_closeness, seq_num_period, seq_num_trend)
        else:
            train_data, val_data, test_data, train_labels, val_labels, test_labels, A1 = ht.load_bj_data(seq_num_closeness)
        print('here')
        train_data_ = np.zeros((train_data.shape[0], train_data.shape[1], train_data.shape[2]))
        print('train_data shape: ', train_data.shape)
        model_perf = utils.model_perf()
        start_time = time.time()
        name = 'fgconv_softmax'

        params = {}
        params['num_epochs'] = 20
        params['batch_size'] = 100
        params['decay_steps'] = train_data.shape[0] / params['batch_size']
        params['eval_frequency'] = 100
        params['filter'] = params_instance[7]
        params['regularization'] = 0
        params['dropout'] = 1
        params['learning_rate'] = params_instance[3]
        params['decay_rate'] = 0.9
        params['momentum'] = 0.9
        # params['filter'] = 'fourier_conv'
        params['filter_num'] = params_instance[4]
        params['kernel_num'] = params_instance[5]
        params['feature_num'] = train_data.shape[2]
        params['conv_layer_num'] = params_instance[6]
        params['seq_num_closeness'] = seq_num_closeness
        params['seq_num_period'] = seq_num_period
        params['seq_num_trend'] = seq_num_trend
        params['infer_func'] = params_instance[9] # 'inference_glstm_period_expand_gconv2'
        params['lstm_layer_count'] = params_instance[8]
        model = gconv_lstm.GconvModel(L, **params)
        # params['model_name'] = 'GNN'
        train_pred, test_pred = model_perf.test(model, name, params,
                                                train_data, train_labels, val_data, val_labels, test_data, test_labels)

        train_pred, test_pred = ht.reverse_normalize(train_pred), ht.reverse_normalize(test_pred)
        train_target, test_target = ht.reverse_normalize(train_labels), ht.reverse_normalize(test_labels)
        end_time = time.time()
        print("total time {0} elapse...\n".format(end_time - start_time))
        train_err = math.sqrt(
            np.sum((train_target - train_pred) ** 2) / (train_pred.shape[0] * train_pred.shape[1] * train_pred.shape[2]))
        test_err = math.sqrt(
            np.sum((test_target - test_pred) ** 2) / (test_pred.shape[0] * test_pred.shape[1] * test_pred.shape[2]))
        print('{0}| {1}| {2}|'.format(model.to_string(), train_err, test_err))
    except Exception as e:
        print(e)
        traceback.print_exc()


