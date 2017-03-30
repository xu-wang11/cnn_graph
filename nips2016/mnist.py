
# coding: utf-8

# In[ ]:
from IPython import get_ipython

from lib.models import lgcnn2_1



import sys, os
sys.path.insert(0, '..')
from lib import models, graph, coarsening, utils
import numpy as np
import time
from nips2016 import humantraffic

import tensorflow as tf
import math

# In[ ]:
if __name__ == "__main__":
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    # Graphs.
    flags.DEFINE_integer('number_edges', 8, 'Graph: minimum number of edges per vertex.')
    flags.DEFINE_string('metric', 'euclidean', 'Graph: similarity measure (between features).')
    # TODO: change cgcnn for combinatorial Laplacians.
    flags.DEFINE_bool('normalized_laplacian', True, 'Graph Laplacian: normalized.')
    flags.DEFINE_integer('coarsening_levels', 4, 'Number of coarsened graphs.')

    # Directories.
    flags.DEFINE_string('dir_data', os.path.join('..', 'data', 'mnist'), 'Directory to store data.')

    # In[ ]:

    def grid_graph(m, corners=False):
        z = graph.grid(m)
        dist, idx = graph.distance_sklearn_metrics(z, k=FLAGS.number_edges, metric=FLAGS.metric)
        A = graph.adjacency(dist, idx)

        # Connections are only vertical or horizontal on the grid.
        # Corner vertices are connected to 2 neightbors only.
        if corners:
            import scipy.sparse
            A = A.toarray()
            A[A < A.max()/1.5] = 0
            A = scipy.sparse.csr_matrix(A)
            print('{} edges'.format(A.nnz))

        print("{} > {} edges".format(A.nnz//2, FLAGS.number_edges*m**2//2))
        return A

    t_start = time.process_time()
    A = grid_graph(32, corners=False)
    # A = graph.replace_random_edges(A, 0)
    # graphs, perm = coarsening.coarsen(A, levels=FLAGS.coarsening_levels, self_connections=False)
    L = [graph.laplacian(A, normalized=True)] # [graph.laplacian(A, normalized=True) for A in graphs]
    print('Execution time: {:.2f}s'.format(time.process_time() - t_start))
    # graph.plot_spectrum(L)
    del A


    # # Data

    # In[ ]:

    # from tensorflow.examples.tutorials.mnist import input_data
    # mnist = input_data.read_data_sets(FLAGS.dir_data, one_hot=False)
    #
    # train_data = mnist.train.images.astype(np.float32)
    # val_data = mnist.validation.images.astype(np.float32)
    # test_data = mnist.test.images.astype(np.float32)
    #
    # train_labels = mnist.train.labels
    # val_labels = mnist.validation.labels
    # test_labels = mnist.test.labels
    node_count = 32 * 32
    t_start = time.process_time()
    ht = humantraffic.HumanTraffic()
    seq_num = 3
    train_data, val_data, test_data, train_labels, val_labels, test_labels = ht.load_data(seq_num)
    train_data_ = np.zeros((train_data.shape[0], train_data.shape[1], train_data.shape[2]))



    # train_data = coarsening.perm_data(train_data, perm)
    # val_data = coarsening.perm_data(val_data, perm)
    # test_data = coarsening.perm_data(test_data, perm)
    # train_data_ = []
    # val_data_ = []
    # test_data_ = []
    # for row in train_data:
    #     train_data_.append(np.transpose([row, row, row]))
    # for row in val_data:
    #     val_data_.append(np.transpose([row, row, row]))
    # for row in test_data:
    #     test_data_.append(np.transpose([row, row, row]))
    # train_data = np.array(train_data_)
    # val_data = np.array(val_data_)
    # test_data = np.array(test_data_)

    print('Execution time: {:.2f}s'.format(time.process_time() - t_start))
    # del perm


    # # Neural networks

    # In[ ]:

    #model = fc1()
    #model = fc2(nhiddens=100)
    #model = cnn2(K=5, F=10)  # K=28 is equivalent to filtering with fgcnn.
    #model = fcnn2(F=10)
    #model = fgcnn2(L[0], F=10)
    #model = lgcnn2_2(L[0], F=10, K=10)
    #model = cgcnn2_3(L[0], F=10, K=5)
    #model = cgcnn2_4(L[0], F=10, K=5)
    #model = cgcnn2_5(L[0], F=10, K=5)

    # if False:
    #     K = 5  # 5 or 5^2
    #     t_start = time.process_time()
    #     mnist.test._images = graph.lanczos(L, mnist.test._images.T, K).T
    #     mnist.train._images = graph.lanczos(L, mnist.train._images.T, K).T
    #     model = lgcnn2_1(L, F=10, K=K)
    #     print('Execution time: {:.2f}s'.format(time.process_time() - t_start))
    #     ph_data = tf.placeholder(tf.float32, (FLAGS.batch_size, mnist.train.images.shape[1], K), 'data')


    # In[ ]:

    common = {}
    common['dir_name']       = 'mnist/'
    common['num_epochs']     = 200
    common['batch_size']     = 20
    common['decay_steps']    = train_data.shape[0] / common['batch_size']
    common['eval_frequency'] = 60
    common['brelu']          = 'b1relu'
    common['pool']           = 'mpool1'
    # C = max(mnist.train.labels) + 1  # number of classes

    model_perf = utils.model_perf()
    print(train_data.shape)


    # In[ ]:

    if True:
        name = 'softmax'
        params = common.copy()
        params['dir_name'] += name
        params['regularization'] = 5e-4
        params['dropout']        = 1
        params['learning_rate']  = 0.02
        params['decay_rate']     = 0.95
        params['momentum']       = 0.9
        params['F']              = []
        params['K']              = []
        params['p']              = []
        params['M']              = [node_count]
        #model_perf.test(models.cgcnn(L, **params), name, params,
        #                train_data, train_labels, val_data, val_labels, test_data, test_labels)


    # In[ ]:

    # Common hyper-parameters for networks with one convolutional layer.
    common['regularization'] = 0
    common['dropout']        = 1
    common['learning_rate']  = 0.02
    common['decay_rate']     = 0.95
    common['momentum']       = 0.9
    common['F']              = [10]
    common['K']              = [20]
    common['p']              = [1]
    common['M']              = [node_count]

    # In[ ]:

    if True:
        name = 'fgconv_softmax'
        params = common.copy()
        params['dir_name'] += name
        params['filter'] = 'chebyshev5'
        params['K'] = [5]
        params['C_0'] = seq_num * 2
        train_pred, test_pred =  model_perf.test(models.cgcnn(L, **params), name, params,
                        train_data, train_labels, val_data, val_labels, test_data, test_labels)

        train_pred, test_pred = ht.reverse_normalize(train_pred), ht.reverse_normalize(test_pred)
        train_target, test_target = ht.reverse_normalize(train_labels), ht.reverse_normalize(test_labels)
        #
        # real_data = np.concatenate(train_labels, test_labels)
        # pred_data = np.concatenate(train_pred, test_pred)
        print(str(math.sqrt(np.sum((train_target - train_pred) ** 2) / (train_pred.shape[0] * train_pred.shape[1]))))
        print(str(math.sqrt(np.sum((test_target - test_pred) ** 2) / (test_pred.shape[0] * test_pred.shape[1]))))

        # plt.figure()
        # plt.plot(train_target[:, 41])
        # plt.plot(train_pred[:, 41])
        # plt.show()
        print("train finish~")


    # In[ ]:
'''
    if True:
        name = 'sgconv_softmax'
        params = common.copy()
        params['dir_name'] += name
        params['filter'] = 'spline'
        # model_perf.test(models.cgcnn(L, **params), name, params,
        #                train_data, train_labels, val_data, val_labels, test_data, test_labels)


    # In[ ]:

    # With 'chebyshev2' and 'b2relu', it corresponds to cgcnn2_2(L[0], F=10, K=20).
    if True:
        name = 'cgconv_softmax'
        params = common.copy()
        params['dir_name'] += name
        params['filter'] = 'chebyshev5'
    #    params['filter'] = 'chebyshev2'
    #    params['brelu'] = 'b2relu'
        model_perf.test(models.cgcnn(L, **params), name, params,
                         train_data, train_labels, val_data, val_labels, test_data, test_labels)


    # In[ ]:

    # Common hyper-parameters for LeNet5-like networks.
    common['regularization'] = 5e-4
    common['dropout']        = 0.5
    common['learning_rate']  = 0.02  # 0.03 in the paper but sgconv_sgconv_fc_softmax has difficulty to converge
    common['decay_rate']     = 0.95
    common['momentum']       = 0.9
    common['F']              = [32, 64]
    common['K']              = [25, 25]
    common['p']              = [4, 4]
    # common['M']              = [512, C]


    # In[ ]:

    # Architecture of TF MNIST conv model (LeNet-5-like).
    # Changes: regularization, dropout, decaying learning rate, momentum optimizer, stopping condition, size of biases.
    # Differences: training data randomization, init conv1 biases at 0.
    if True:
        name = 'fgconv_fgconv_fc_softmax' #  'Non-Param'
        params = common.copy()
        params['dir_name'] += name
        params['filter'] = 'fourier'
        L.append(L[0])
        params['K'] = [L[0].shape[0], L[0].shape[0]]
        model_perf.test(models.cgcnn(L, **params), name, params,
                        train_data, train_labels, val_data, val_labels, test_data, test_labels)


    # In[ ]:

    if True:
        name = 'sgconv_sgconv_fc_softmax'  # 'Spline'
        params = common.copy()
        params['dir_name'] += name
        params['filter'] = 'spline'
        model_perf.test(models.cgcnn(L, **params), name, params,
                        train_data, train_labels, val_data, val_labels, test_data, test_labels)


    # In[ ]:

    if True:
        name = 'cgconv_cgconv_fc_softmax'  # 'Chebyshev'
        params = common.copy()
        params['dir_name'] += name
        params['filter'] = 'chebyshev5'
        model_perf.test(models.cgcnn(L, **params), name, params,
                        train_data, train_labels, val_data, val_labels, test_data, test_labels)


    # In[ ]:

    model_perf.show()


    # In[ ]:

    if False:
        grid_params = {}
        data = (train_data, train_labels, val_data, val_labels, test_data, test_labels)
        utils.grid_search(params, grid_params, *data, model=lambda x: models.cgcnn(L,**x))

'''