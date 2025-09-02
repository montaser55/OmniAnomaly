# -*- coding: utf-8 -*-
import logging

import tensorflow.compat.v1 as tf

tf.compat.v1.disable_v2_behavior()
import tensorflow_probability as tfp

tfd = tfp.distributions

# Remove the custom Distribution class and use TFP's Distribution directly
# from tfsnippet.distributions import Distribution

# Remove the entire custom Distribution class since we'll use TFP's directly



class TfpDistribution:
    """
    A simplified wrapper class for `tfp.distributions.Distribution`
    that provides compatibility interface
    """

    def __init__(self, distribution):
        if not isinstance(distribution, tfd.Distribution):
            raise TypeError('`distribution` is not an instance of `tfp.distributions.Distribution`')
        self._distribution = distribution

    @property
    def is_continuous(self):
        return True  # Most TFP distributions are continuous

    @property
    def is_reparameterized(self):
        return (hasattr(self._distribution, 'reparameterization_type') and
                self._distribution.reparameterization_type == tfd.FULLY_REPARAMETERIZED)

    @property
    def value_shape(self):
        return self._distribution.event_shape

    def get_value_shape(self):
        return self._distribution.event_shape

    @property
    def batch_shape(self):
        return self._distribution.batch_shape

    def get_batch_shape(self):
        return self._distribution.batch_shape

    def sample(self, n_samples=None, is_reparameterized=None, group_ndims=0, compute_density=False, name=None):
        with tf.name_scope(name or 'sample'):
            if n_samples is not None:
                samples = self._distribution.sample(n_samples)
            else:
                samples = self._distribution.sample()

            # For compatibility, return an object with log_prob method
            class SampleResult:
                def __init__(self, tensor, distribution):
                    self.tensor = tensor
                    self.distribution = distribution

                def log_prob(self):
                    return self.distribution.log_prob(self.tensor)

            result = SampleResult(samples, self)

            if compute_density:
                # Precompute log probability if requested
                result._log_prob = self.log_prob(samples)

            return result

    def log_prob(self, given, group_ndims=0, name=None):
        with tf.name_scope(name or 'log_prob'):
            return self._distribution.log_prob(given)

    # Delegate other methods to the underlying distribution
    def __getattr__(self, name):
        return getattr(self._distribution, name)

    def __repr__(self):
        return f'TfpDistribution({repr(self._distribution)})'


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
            fw_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(rnn_num_hidden, forget_bias=1.0)
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