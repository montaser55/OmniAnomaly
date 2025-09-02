# -*- coding: utf-8 -*-
from functools import partial

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import tensorflow_probability as tfp
# import tfsnippet as spt
from tensorflow.python.ops.linalg.linear_operator_identity import LinearOperatorIdentity
from tensorflow_probability.python.distributions import LinearGaussianStateSpaceModel, MultivariateNormalDiag
# from tfsnippet.distributions import Normal
# from tfsnippet.utils import VarScopeObject, reopen_variable_scope
# from tfsnippet.variational import VariationalInference

from omni_anomaly.recurrent_distribution import RecurrentDistribution
from omni_anomaly.vae import Lambda, VAE
from omni_anomaly.wrapper import TfpDistribution, softplus_std, rnn, wrap_params_net


class Normal:
    """Minimal replacement for tfsnippet.distributions.Normal"""

    def __init__(self, mean, std, is_reparameterized=True, **kwargs):
        self.mean = mean
        self.std = std
        self._is_reparameterized = is_reparameterized
        self._tfp_dist = tfp.distributions.Normal(loc=mean, scale=std, **kwargs)

    @property
    def is_reparameterized(self):
        return self._is_reparameterized

    @property
    def is_continuous(self):
        return True

    def sample(self, n_samples=None, **kwargs):
        if n_samples is not None:
            samples = self._tfp_dist.sample(n_samples, **kwargs)
        else:
            samples = self._tfp_dist.sample(**kwargs)
        return samples

    def log_prob(self, value, **kwargs):
        return self._tfp_dist.log_prob(value, **kwargs)

    def prob(self, value, **kwargs):
        return tf.exp(self.log_prob(value, **kwargs))


# ==================== Utils (already defined, but included for completeness) ====================
class VarScopeObject:
    """Minimal replacement for tfsnippet.utils.VarScopeObject"""

    def __init__(self, name=None, scope=None):
        self._name = name
        self._scope = scope or tf.compat.v1.get_variable_scope().name

    @property
    def name(self):
        return self._name

    @property
    def variable_scope(self):
        return self._scope


def reopen_variable_scope(scope):
    """Reopen a variable scope for reuse"""
    return tf.compat.v1.variable_scope(scope, reuse=True)


# ==================== Variational Inference Replacement ====================
class VariationalInference:
    """Minimal replacement for tfsnippet.variational.VariationalInference"""

    def __init__(self, latent_log_joint, latent_log_probs, axis=None):
        self.latent_log_joint = latent_log_joint
        self.latent_log_probs = latent_log_probs
        self.axis = axis

        # Compute ELBO (Evidence Lower Bound)
        self.log_likelihood = latent_log_joint
        self.kl_divergence = self._compute_kl_divergence()
        self.elbo = self.log_likelihood - self.kl_divergence
        self.loss = -self.elbo  # Negative ELBO for minimization

    def _compute_kl_divergence(self):
        """Compute KL divergence between variational and prior distributions"""
        if isinstance(self.latent_log_probs, dict):
            # Sum KL divergences for multiple latent variables
            kl_terms = []
            for log_q, log_p in self.latent_log_probs.values():
                kl_terms.append(log_q - log_p)
            return tf.add_n(kl_terms)
        else:
            # Single latent variable case
            log_q, log_p = self.latent_log_probs
            return log_q - log_p

    @property
    def training_loss(self):
        """Get the training loss (negative ELBO)"""
        return self.loss

    def reparameterized(self, *args, **kwargs):
        """Stub for reparameterized training - returns the loss directly"""
        return self.loss

    def sgvb(self, *args, **kwargs):
        """Stub for SGVB training - returns the loss directly"""
        return self.loss

    def reinforce(self, *args, **kwargs):
        """Stub for REINFORCE training - returns the loss directly"""
        return self.loss

    def add_summary(self, name, value, *args, **kwargs):
        """Simple summary addition"""
        tf.summary.scalar(name, value, *args, **kwargs)

    def __repr__(self):
        return f"VariationalInference(elbo={self.elbo}, loss={self.loss})"


# ==================== Helper functions for variational inference ====================
def vi_objective(latent_log_joint, latent_log_probs, axis=None):
    """Helper function to create VariationalInference objective"""
    return VariationalInference(latent_log_joint, latent_log_probs, axis)


def build_vi_objective(latent_log_joint, latent_log_probs, axis=None):
    """Alternative helper function for VI objective"""
    vi = VariationalInference(latent_log_joint, latent_log_probs, axis)
    return vi.training_loss


def planar_normalizing_flows_tfp(n_layers, name='planar_flow'):
    """
    Alternative implementation using TensorFlow Probability's AffineScalar bijector
    """
    from tensorflow_probability import bijectors as tfb

    # Create a chain of simple affine transformations (simplified version)
    bijectors = []
    for i in range(n_layers):
        with tf.compat.v1.variable_scope(f'{name}_layer_{i}'):
            # Trainable parameters for each flow layer
            scale = tf.compat.v1.get_variable('scale', shape=[1],
                                              initializer=tf.initializers.constant(1.0))
            shift = tf.compat.v1.get_variable('shift', shape=[1],
                                              initializer=tf.initializers.zeros())

            # Create affine transformation
            bijector = tfb.AffineScalar(shift=shift, scale=scale)
            bijectors.append(bijector)

    # Chain all bijectors together
    flow_bijector = tfb.Chain(bijectors)

    return flow_bijector


# ==================== Usage in your code ====================
# In your model's __init__ method, replace:


class OmniAnomaly(VarScopeObject):
    def __init__(self, config, name=None, scope=None):
        self.config = config
        super(OmniAnomaly, self).__init__(name=name, scope=scope)
        with reopen_variable_scope(self.variable_scope):
            if config.posterior_flow_type == 'nf':
                # self._posterior_flow = spt.layers.planar_normalizing_flows(
                #     config.nf_layers, name='posterior_flow')
                self._posterior_flow = planar_normalizing_flows_tfp(config.nf_layers, name='posterior_flow')

            else:
                self._posterior_flow = None
            self._window_length = config.window_length
            self._x_dims = config.x_dim
            self._z_dims = config.z_dim
            self._vae = VAE(
                p_z=TfpDistribution(
                    LinearGaussianStateSpaceModel(
                        num_timesteps=config.window_length,
                        transition_matrix=LinearOperatorIdentity(config.z_dim),
                        transition_noise=MultivariateNormalDiag(
                            scale_diag=tf.ones([config.z_dim])),
                        observation_matrix=LinearOperatorIdentity(config.z_dim),
                        observation_noise=MultivariateNormalDiag(
                            scale_diag=tf.ones([config.z_dim])),
                        initial_state_prior=MultivariateNormalDiag(
                            scale_diag=tf.ones([config.z_dim]))
                    )
                ) if config.use_connected_z_p else Normal(mean=tf.zeros([config.z_dim]), std=tf.ones([config.z_dim])),
                p_x_given_z=Normal,
                q_z_given_x=partial(RecurrentDistribution,
                                    mean_q_mlp=partial(tf.layers.dense, units=config.z_dim, name='z_mean', reuse=tf.AUTO_REUSE),
                                    std_q_mlp=partial(softplus_std, units=config.z_dim, epsilon=config.std_epsilon,
                                                      name='z_std'),
                                    z_dim=config.z_dim, window_length=config.window_length) if config.use_connected_z_q else Normal,
                h_for_p_x=Lambda(
                    partial(
                        wrap_params_net,
                        h_for_dist=lambda x: rnn(x=x,
                                                 window_length=config.window_length,
                                                 rnn_num_hidden=config.rnn_num_hidden,
                                                 hidden_dense=2,
                                                 dense_dim=config.dense_dim,
                                                 name='rnn_p_x'),
                        mean_layer=partial(
                            tf.layers.dense, units=config.x_dim, name='x_mean', reuse=tf.AUTO_REUSE
                        ),
                        std_layer=partial(
                            softplus_std, units=config.x_dim, epsilon=config.std_epsilon,
                            name='x_std'
                        )
                    ),
                    name='p_x_given_z'
                ),
                h_for_q_z=Lambda(
                    lambda x: {'input_q': rnn(x=x,
                                              window_length=config.window_length,
                                              rnn_num_hidden=config.rnn_num_hidden,
                                              hidden_dense=2,
                                              dense_dim=config.dense_dim,
                                              name="rnn_q_z")},
                    name='q_z_given_x'
                ) if config.use_connected_z_q else Lambda(
                    partial(
                        wrap_params_net,
                        h_for_dist=lambda x: rnn(x=x,
                                                 window_length=config.window_length,
                                                 rnn_num_hidden=config.rnn_num_hidden,
                                                 hidden_dense=2,
                                                 dense_dim=config.dense_dim,
                                                 name="rnn_q_z"),
                        mean_layer=partial(
                            tf.layers.dense, units=config.z_dim, name='z_mean', reuse=tf.AUTO_REUSE
                        ),
                        std_layer=partial(
                            softplus_std, units=config.z_dim, epsilon=config.std_epsilon,
                            name='z_std'
                        )
                    ),
                    name='q_z_given_x'
                )
            )

    @property
    def x_dims(self):
        """Get the number of `x` dimensions."""
        return self._x_dims

    @property
    def z_dims(self):
        """Get the number of `z` dimensions."""
        return self._z_dims

    @property
    def vae(self):
        """
        Get the VAE object of this :class:`OmniAnomaly` model.

        Returns:
            VAE: The VAE object of this model.
        """
        return self._vae

    @property
    def window_length(self):
        return self._window_length

    def get_training_loss(self, x, n_z=None):
        """
        Get the training loss for `x`.

        Args:
            x (tf.Tensor): 2-D `float32` :class:`tf.Tensor`, the windows of
                KPI observations in a mini-batch.
            n_z (int or None): Number of `z` samples to take for each `x`.
                (default :obj:`None`, one sample without explicit sampling
                dimension)

        Returns:
            tf.Tensor: 0-d tensor, the training loss, which can be optimized
                by gradient descent algorithms.
        """
        with tf.name_scope('training_loss'):
            chain = self.vae.chain(x, n_z=n_z, posterior_flow=self._posterior_flow)
            x_log_prob = chain.model['x'].log_prob(group_ndims=0)
            log_joint = tf.reduce_sum(x_log_prob, -1)
            chain.vi.training.sgvb()
            vi = VariationalInference(
                log_joint=log_joint,
                latent_log_probs=chain.vi.latent_log_probs,
                axis=chain.vi.axis
            )
            loss = tf.reduce_mean(vi.training.sgvb())
            return loss

    def get_score(self, x, n_z=None,
                  last_point_only=True):
        """
        Get the reconstruction probability for `x`.

        The larger `reconstruction probability`, the less likely a point
        is anomaly.  You may take the negative of the score, if you want
        something to directly indicate the severity of anomaly.

        Args:
            x (tf.Tensor): 2-D `float32` :class:`tf.Tensor`, the windows of
                KPI observations in a mini-batch.
            n_z (int or None): Number of `z` samples to take for each `x`.
                (default :obj:`None`, one sample without explicit sampling
                dimension)
            last_point_only (bool): Whether to obtain the reconstruction
                probability of only the last point in each window?
                (default :obj:`True`)

        Returns:
            tf.Tensor: The reconstruction probability, with the shape
                ``(len(x) - self.x_dims + 1,)`` if `last_point_only` is
                :obj:`True`, or ``(len(x) - self.x_dims + 1, self.x_dims)``
                if `last_point_only` is :obj:`False`.  This is because the
                first ``self.x_dims - 1`` points are not the last point of
                any window.
        """
        with tf.name_scope('get_score'):
            x_r = x

            # get the reconstruction probability
            print('-' * 30, 'testing', '-' * 30)
            q_net = self.vae.variational(x=x_r, n_z=n_z, posterior_flow=self._posterior_flow)  # notice: x=x_r
            p_net = self.vae.model(z=q_net['z'], x=x, n_z=n_z)  # notice: x=x
            z_samples = q_net['z'].tensor
            z_mean = tf.reduce_mean(z_samples, axis=0) if n_z is not None else z_samples
            z_std = tf.sqrt(tf.reduce_sum(tf.square(z_samples - z_mean), axis=0) / (n_z - 1)) \
                if n_z is not None and n_z > 1 else tf.zeros_like(z_mean)
            z = tf.concat((z_mean, z_std), axis=-1)

            r_prob = p_net['x'].log_prob(group_ndims=int(not self.config.get_score_on_dim))

            if last_point_only:
                r_prob = r_prob[:, -1]
            return r_prob, z
