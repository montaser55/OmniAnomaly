# -*- coding: utf-8 -*-
import logging

import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
import tensorflow_probability as tfp
# from tfsnippet.distributions import Distribution

class Distribution:

    def __init__(self):
        self._is_continuous = True
        self._is_reparameterized = False

    @property
    def is_continuous(self):
        return self._is_continuous

    @property
    def is_reparameterized(self):
        return self._is_reparameterized

    @property
    def value_shape(self):
        return self.get_value_shape()

    def get_value_shape(self):
        raise NotImplementedError

    @property
    def batch_shape(self):
        return self.get_batch_shape()

    def get_batch_shape(self):
        raise NotImplementedError

    def sample(self, n_samples=None, is_reparameterized=None, group_ndims=0, compute_density=False, name=None):
        raise NotImplementedError

    def log_prob(self, given, group_ndims=0, name=None):
        raise NotImplementedError


class TfpDistribution(Distribution):
    """
    A wrapper class for `tfp.distributions.Distribution`
    """

    @property
    def is_continuous(self):
        return self._is_continuous

    def __init__(self, distribution):
        if not isinstance(distribution, tfp.distributions.Distribution):
            raise TypeError('`distribution` is not an instance of `tfp.'
                            'distributions.Distribution`')
        super(TfpDistribution, self).__init__()
        self._distribution = distribution
        self._is_continuous = True
        self._is_reparameterized = self._distribution.reparameterization_type is tfp.distributions.FULLY_REPARAMETERIZED

    def __repr__(self):
        return 'Distribution({!r})'.format(self._distribution)

    @property
    def dtype(self):
        return self._distribution.dtype

    @property
    def is_reparameterized(self):
        return self._is_reparameterized

    @property
    def value_shape(self):
        return self._distribution.event_shape

    def get_value_shape(self):
        return self._distribution.event_shape

    @property
    def batch_shape(self):
        return self._distribution.batch_shape

    def get_batch_shape(self):
        return self._distribution.batch_shape()

    def sample(self, n_samples=None, is_reparameterized=None, group_ndims=0, compute_density=False,
               name=None):
        class SimpleStochasticTensor:
            def __init__(self, tensor, distribution, n_samples, group_ndims, is_reparameterized):
                self.tensor = tensor
                self.distribution = distribution
                self.n_samples = n_samples
                self.group_ndims = group_ndims
                self.is_reparameterized = is_reparameterized
                self._self_prob = None

            def log_prob(self):
                return self.distribution.log_prob(self.tensor)
        if n_samples is None or n_samples < 2:
            n_samples = 2
        with tf.name_scope(name=name, default_name='sample'):
            samples = self._distribution.sample(n_samples)
            samples = tf.reduce_mean(samples, axis=0)
            t = SimpleStochasticTensor(
                distribution=self,
                tensor=samples,
                n_samples=n_samples,
                group_ndims=group_ndims,
                is_reparameterized=self.is_reparameterized
            )
            if compute_density:
                with tf.name_scope('compute_prob_and_log_prob'):
                    log_p = t.log_prob()
                    t._self_prob = tf.exp(log_p)
            return t

    def log_prob(self, given, group_ndims=0, name=None):
        with tf.name_scope(name=name, default_name='log_prob'):
            return self._distribution.log_prob(given)


def softplus_std(inputs, units, epsilon, name):
    return tf.nn.softplus(tf.compat.v1.layers.dense(inputs, units, name=name, reuse=tf.compat.v1.AUTO_REUSE)) + epsilon


def rnn(x,
        window_length,
        rnn_num_hidden,
        rnn_cell='GRU',
        hidden_dense=2,
        dense_dim=200,
        time_axis=1,
        name='rnn'):
    # Use compat.v1 for RNN components
    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        if len(x.shape) == 4:
            x = tf.reduce_mean(x, axis=0)
        elif len(x.shape) != 3:
            logging.error("rnn input shape error")
        x = tf.unstack(x, window_length, time_axis)

        if rnn_cell == 'LSTM':
            # Define lstm cells with TensorFlow
            # Forward direction cell
            fw_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(rnn_num_hidden,
                                        forget_bias=1.0)
        elif rnn_cell == "GRU":
            fw_cell = tf.compat.v1.nn.rnn_cell.GRUCell(rnn_num_hidden)
        elif rnn_cell == 'Basic':
            fw_cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(rnn_num_hidden)
        else:
            raise ValueError("rnn_cell must be LSTM or GRU")

        # Get lstm cell output
        try:
            outputs, _ = tf.compat.v1.nn.static_rnn(fw_cell, x, dtype=tf.float32)
        except Exception:  # Old TensorFlow version only returns outputs not states
            outputs = tf.compat.v1.nn.static_rnn(fw_cell, x, dtype=tf.float32)
        outputs = tf.stack(outputs, axis=time_axis)
        for i in range(hidden_dense):
            outputs = tf.compat.v1.layers.dense(outputs, dense_dim)
        return outputs
    # return size: (batch_size, window_length, rnn_num_hidden)


def wrap_params_net(inputs, h_for_dist, mean_layer, std_layer):
    with tf.compat.v1.variable_scope('hidden', reuse=tf.compat.v1.AUTO_REUSE):
        h = h_for_dist(inputs)
    return {
        'mean': mean_layer(h),
        'std': std_layer(h),
    }


def wrap_params_net_srnn(inputs, h_for_dist):
    with tf.compat.v1.variable_scope('hidden', reuse=tf.compat.v1.AUTO_REUSE):
        h = h_for_dist(inputs)
    return {
        'input_q': h
    }