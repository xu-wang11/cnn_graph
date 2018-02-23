# -*- coding: utf-8 -*-
# Create by Xu Wang
# 2018-01-09

import numpy as np
import time
import os
import collections
import shutil

# tensorflow
import tensorflow as tf


class GraphModel(object):
    def __init__(self):
        self.is_train = None
        self.reuse = None
        self.regularizers = []
        self.convFuncs = []
        self.activeFuncs = []
        self.nets = {}
        self.x = None
        self.y = None
        self.op_dropout = None
        self.graph = None
        self.op_loss = None
        self.op_loss_average = None
        self.op_train = None
        self.op_prediction = None
        self.op_init = None
        self.op_summary = None
        self.op_saver = None
        self.sess = None

    # create tensor flow graph
    def build_graph(self, node_num, feature_num, output_num):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(2017)
            # Inputs.
            with tf.name_scope('inputs'):
                self.is_train = tf.placeholder(tf.bool, name='phase_train')
                self.x = tf.placeholder(tf.float32, (self.batch_size, node_num, feature_num), 'data')
                self.y = tf.placeholder(tf.float32, (self.batch_size, node_num, output_num), 'labels')
                self.op_dropout = tf.placeholder(tf.float32, (), 'dropout')
            # Model.
            op_pred = self.inference(self.x, self.dropout)
            self.op_loss, self.op_loss_average = self.loss(op_pred, self.y, self.regularization)
            self.op_train = self.training(self.op_loss, self.learning_rate,
                                          self.decay_steps, self.decay_rate, self.momentum)
            self.op_prediction = self.prediction(op_pred)
            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()
            # Summaries for TensorBoard and Save for model parameters.
            # self.op_summary = tf.summary.merge_all()
            # self.op_saver = tf.train.Saver(max_to_keep=5)
        self.graph.finalize()
        # writer = tf.summary.FileWriter(self._get_path("checkpoints") + "/graph_def", tf.Session().graph)
        # writer.close()

    # High-level interface which runs the constructed computational graph.
    def predict(self, data, labels=None, sess=None):
        loss = 0
        size = data.shape[0]
        predictions = np.empty((size, data.shape[1], 2))
        sess = self._get_session(sess)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])
            batch_data = np.zeros((self.batch_size, data.shape[1], data.shape[2]))
            tmp_data = data[begin:end, :]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end - begin] = tmp_data
            feed_dict = {self.x: batch_data, self.op_dropout: 1, self.is_train: False}
            # Compute loss if labels are given.
            if labels is not None:
                batch_labels = np.zeros((self.batch_size, data.shape[1], 2))
                batch_labels[:end - begin] = labels[begin:end]
                feed_dict[self.y] = batch_labels
                batch_pred, batch_loss = sess.run(
                    [self.op_prediction, self.op_loss], feed_dict)
                loss += batch_loss
            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)

            predictions[begin:end] = batch_pred[:end - begin]

        if labels is not None:
            return predictions, loss * self.batch_size / size
        else:
            return predictions

    def evaluate(self, data, labels, sess=None):
        """
        Runs one evaluation against the full epoch of data.
        Return the precision and the number of correct predictions.
        Batch evaluation saves memory and enables this to run on smaller GPUs.

        sess: the session in which the model has been trained.
        op: the Tensor that returns the number of correct predictions.
        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        labels: size N
            N: number of signals (samples)
        """
        t_process, t_wall = time.process_time(), time.time()
        predictions, loss = self.predict(data, labels, sess)
        # print(predictions)
        # ncorrects = sum(predictions == labels)
        # accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
        # mse = sklearn.metrics.mean_squared_error(labels, predictions)
        mse = np.sum((labels - predictions) ** 2) / (predictions.shape[0] * predictions.shape[1] * predictions.shape[2])
        # f1 = 100 * sklearn.metrics.f1_score(labels, predictions, average='weighted')
        string = 'mse: {:.5f} ( {:d}), f1 (weighted), loss: {:.2e}'.format(
            mse, len(labels), loss)
        if sess is None:
            string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(time.process_time() - t_process, time.time() - t_wall)
        return string, mse, 0, loss, predictions

    def fit(self, train_data, train_labels, val_data, val_labels):
        t_process, t_wall = time.process_time(), time.time()
        sess = tf.Session(graph=self.graph)
        self.sess= sess
        # shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
        # writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        # shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
        # os.makedirs(self._get_path('checkpoints'))
        # path = os.path.join(self._get_path('checkpoints'), 'model')
        self.sess.run(self.op_init)
        self.sess.graph.finalize()

        # Training.
        accuracies = []
        losses = []
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_data.shape[0] / self.batch_size)
        start_time = time.time()
        for step in range(1, num_steps + 1):
            # print(train_data.shape)
            # Be sure to have used all the samples before using one a second time.
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(train_data.shape[0]))
            idx = [indices.popleft() for i in range(self.batch_size)]

            batch_data, batch_labels = train_data[idx, :], train_labels[idx]
            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()  # convert sparse matrices
            feed_dict = {self.x: batch_data, self.y: batch_labels, self.op_dropout: self.dropout,
                         self.is_train: True}
            query_list = [self.op_train, self.op_loss, self.op_loss_average]
            for k, v in self.nets.items():
                query_list.append(v)
            x = sess.run(query_list, feed_dict)
            learning_rate, loss, loss_average = x[0], x[1], x[2]  # x = sess.run(query_list, feed_dict)
            parse_result = {}
            i = 1
            for k, v in self.nets.items():
                parse_result[k] = x[2 + i]
                i += 1

            # print("come to deu")
            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0 or step == num_steps:
                epoch = step * self.batch_size / train_data.shape[0]
                print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                print('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))
                string, accuracy, f1, loss, no_use = self.evaluate(val_data, val_labels, sess)
                accuracies.append(accuracy)
                losses.append(loss)
                print('  validation {}'.format(string))
                print('  time: {:.0f}s (wall {:.0f}s)'.format(time.process_time() - t_process, time.time() - t_wall))

                end_time = time.time()

                print("time spend {0}...:\n".format(end_time - start_time))
                start_time = time.time()
                # Summaries for TensorBoard.
                # summary = tf.Summary()
                # summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                # summary.value.add(tag='validation/accuracy', simple_value=accuracy)
                # summary.value.add(tag='validation/f1', simple_value=f1)
                # summary.value.add(tag='validation/loss', simple_value=loss)
                # writer.add_summary(summary, step)

                # # Save model parameters (for evaluation).
                # self.op_saver.save(sess, path, global_step=step)

        print('validation accuracy: peak = {:.2f}, mean = {:.2f}'.format(max(accuracies), np.mean(accuracies[-10:])))
        # writer.close()
        # sess.close()

        t_step = (time.time() - t_wall) / num_steps
        return accuracies, losses, t_step

    def get_var(self, name):
        sess = self._get_session()
        var = self.graph.get_tensor_by_name(name + ':0')
        val = sess.run(var)
        # sess.close()
        return val

    # Methods to construct the computational graph.
    # S_0 is stack number


    def inference(self, data, dropout):
        """
        It builds the model, i.e. the computational graph, as far as
        is required for running the network forward to make predictions,
        i.e. return logits given raw data.

        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        training: we may want to discriminate the two, e.g. for dropout.
            True: the model is built for training.
            False: the model is built for evaluation.
        """
        # TODO: optimizations for sparse data
        logits = self._inference(data, dropout)
        return logits

    def probabilities(self, logits):
        """Return the probability of a sample to belong to each class."""
        with tf.name_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            return probabilities

    def prediction(self, x):
        """Return the predicted classes."""
        with tf.name_scope('prediction'):
            # N, M = x.get_shape()
            # b = self._bias_variable([1, 1], regularization=False)
            # prediction = tf.nn.tanh(x + b)
            # prediction = x
            # prediction = x
            prediction = tf.nn.relu(x)  # tf.argmax(logits, axis=1)
            # prediction = tf.clip_by_value(prediction, 0, 1)
            self.nets[prediction.name] = prediction
        return prediction

    def loss(self, logits, labels, regularization):
        """Adds to the inference model the layers required to generate loss."""
        with tf.name_scope('loss'):
            # with tf.name_scope('cross_entropy'):
            #     labels = tf.to_int64(labels)
            #     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            #     cross_entropy = tf.reduce_mean(cross_entropy)
            with tf.name_scope('mse'):
                # cross_entropy = tf.nn.l2_loss(labels - logits)
                cross_entropy = tf.reduce_mean(tf.square(tf.subtract(labels, logits)))
            # with tf.name_scope('regularization'):
            #    regularization *= tf.add_n(self.regularizers)
            # loss = cross_entropy + regularization
            loss = cross_entropy
            self.nets[loss.name] = loss
            # Summaries for TensorBoard.
            # tf.summary.scalar('loss/cross_entropy', cross_entropy)
            # tf.summary.scalar('loss/regularization', regularization)
            # tf.summary.scalar('loss/total', loss)
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                # op_averages = loss
                op_averages = averages.apply([cross_entropy])
                # tf.summary.scalar('loss/avg/cross_entropy', averages.average(cross_entropy))
                # # tf.summary.scalar('loss/avg/regularization', averages.average(regularization))
                # tf.summary.scalar('loss/avg/total', averages.average(loss))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')

            return loss, loss_average

    def training(self, loss, learning_rate, decay_steps, decay_rate=0.95, momentum=0.9):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            # Learning rate.
            global_step = tf.Variable(0, name='global_step', trainable=False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(
                    learning_rate, global_step, decay_steps, decay_rate, staircase=True)
            # tf.summary.scalar('learning_rate', learning_rate)
            # Optimizer.
            # if momentum == 0:
            #    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            # optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            # else:
            #    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            # optimizer = tf.train.AdagradOptimizer(learning_rate)
            # optimizer = tf.train.AdadeltaOptimizer(learning_rate)
            grads = optimizer.compute_gradients(loss, aggregation_method=2)
            # self.nets[grads.name] = grads
            op_gradients = optimizer.apply_gradients(grads, global_step=global_step)

            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    # tf.summary.histogram(var.op.name + '/gradients', grad)
                    self.nets[var.op.name] = grad
            # The op return the learning rate.

            with tf.control_dependencies([op_gradients]):
                op_train = tf.identity(learning_rate, name='control')
            return op_train

    # Helper methods.

    def _get_path(self, folder):
        path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(path, '..', folder, self.dir_name)

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            sess = self.sess
            # filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            # self.op_saver.restore(sess, filename)
        return sess

    def _weight_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        # tf.summary.histogram(var.op.name, var)
        self.nets[var.name] = var
        return var

    def _bias_variable(self, shape, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        # tf.summary.histogram(var.op.name, var)
        self.nets[var.name] = var
        return var

    def _conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
