# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_v2_behavior()
import tensorflow_probability as tfp

tfd = tfp.distributions

class BayesianNet:
    def __init__(self):
        self._nodes = {}

    def add(self, name, distribution, **kwargs):
        self._nodes[name] = {'distribution': distribution, 'kwargs': kwargs}
        return self

    def __getitem__(self, name):
        return self._nodes[name]

    def output(self, names=None):
        if names is None:
            return self._nodes
        return {name: self._nodes[name] for name in names}


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

    def get_value_shape(self):
        raise NotImplementedError

    def get_batch_shape(self):
        raise NotImplementedError

    def sample(self, n_samples=None, **kwargs):
        raise NotImplementedError

    def log_prob(self, given, **kwargs):
        raise NotImplementedError


class TfpDistribution(Distribution):
    """
    A wrapper for TFP distributions that inherits from the custom Distribution class
    """

    def __init__(self, distribution):
        super().__init__()
        if not isinstance(distribution, tfd.Distribution):
            raise TypeError('`distribution` is not an instance of `tfp.distributions.Distribution`')
        self._distribution = distribution

        # Set the reparameterization flag based on TFP distribution
        self._is_reparameterized = (
                hasattr(distribution, 'reparameterization_type') and
                distribution.reparameterization_type == tfd.FULLY_REPARAMETERIZED
        )

    def get_value_shape(self):
        return self._distribution.event_shape

    def get_batch_shape(self):
        return self._distribution.batch_shape

    def sample(self, n_samples=None, **kwargs):
        if n_samples is not None:
            return self._distribution.sample(n_samples)
        return self._distribution.sample()

    def log_prob(self, given, **kwargs):
        return self._distribution.log_prob(given)

    # Delegate other methods to the underlying distribution
    def __getattr__(self, name):
        return getattr(self._distribution, name)

    def __repr__(self):
        return f'TfpDistribution({repr(self._distribution)})'


class StochasticTensor:
    def __init__(self, distribution, tensor, n_samples=1, group_ndims=0, is_reparameterized=None):
        self.distribution = distribution
        self.tensor = tensor
        self.n_samples = n_samples
        self.group_ndims = group_ndims
        self.is_reparameterized = is_reparameterized if is_reparameterized is not None else distribution.is_reparameterized
        self._self_prob = None

    def log_prob(self):
        return self.distribution.log_prob(self.tensor)

    def prob(self):
        if self._self_prob is None:
            self._self_prob = tf.exp(self.log_prob())
        return self._self_prob


def validate_n_samples_arg(n_samples, name='n_samples'):
    if n_samples is not None and n_samples < 1:
        raise ValueError('{} must be None or a positive integer.'.format(name))
    return n_samples


def instance_reuse(func):
    """Minimal decorator replacement for instance_reuse"""
    return func


def is_tensor_object(obj):
    """Check if object is a TensorFlow tensor"""
    return tf.is_tensor(obj)


def reopen_variable_scope(scope):
    """Minimal variable scope reopening"""
    return tf.compat.v1.variable_scope(scope, reuse=True)


class VarScopeObject:
    """Minimal variable scope object base"""

    def __init__(self, name=None, scope=None):
        self._name = name
        self._scope = scope

    @property
    def name(self):
        return self._name

    @property
    def variable_scope(self):
        return self._scope


class VAE(VarScopeObject):
    def __init__(self, p_z, p_x_given_z, q_z_given_x, h_for_p_x, h_for_q_z,
                 z_group_ndims=1, x_group_ndims=1, is_reparameterized=None,
                 name=None, scope=None):
        # Check if p_z is either a custom Distribution or a TFP distribution
        is_custom_dist = isinstance(p_z, Distribution)
        is_tfp_dist = hasattr(p_z, '_distribution') and isinstance(p_z._distribution, tfd.Distribution)

        if not (is_custom_dist or is_tfp_dist):
            raise TypeError('`p_z` must be an instance of `Distribution` or a TFP distribution wrapper')

        if not callable(h_for_p_x):
            raise TypeError('`h_for_p_x` must be an instance of `Module` or '
                            'a callable object')
        if not callable(h_for_q_z):
            raise TypeError('`h_for_q_z` must be an instance of `Module` or '
                            'a callable object')
        super(VAE, self).__init__(name=name, scope=scope)

        # Defensive coding: wrap `h_for_p_x` and `h_for_q_z` in reused scope.
        if not isinstance(h_for_p_x, VarScopeObject):
            with reopen_variable_scope(self.variable_scope):
                h_for_p_x = Lambda(h_for_p_x, name='h_for_p_x')
        if not isinstance(h_for_q_z, VarScopeObject):
            with reopen_variable_scope(self.variable_scope):
                h_for_q_z = Lambda(h_for_q_z, name='h_for_q_z')

        self._p_z = p_z
        self._p_x_given_z = p_x_given_z
        self._q_z_given_x = q_z_given_x
        self._h_for_p_x = h_for_p_x
        self._h_for_q_z = h_for_q_z
        self._z_group_ndims = z_group_ndims
        self._x_group_ndims = x_group_ndims
        self._is_reparameterized = is_reparameterized

    def __call__(self, inputs, **kwargs):
        with reopen_variable_scope(self.variable_scope):
            with tf.name_scope('forward'):
                return self._forward(inputs, **kwargs)

    @property
    def p_z(self):
        return self._p_z

    @property
    def p_x_given_z(self):
        return self._p_x_given_z

    @property
    def q_z_given_x(self):
        return self._q_z_given_x

    @property
    def h_for_p_x(self):
        return self._h_for_p_x

    @property
    def h_for_q_z(self):
        return self._h_for_q_z

    @property
    def z_group_ndims(self):
        return self._z_group_ndims

    @property
    def x_group_ndims(self):
        return self._x_group_ndims

    @property
    def is_reparameterized(self):
        return self._is_reparameterized

    @instance_reuse
    def variational(self, x, z=None, n_z=None, posterior_flow=None):
        observed = {}
        if z is not None:
            observed['z'] = z
        net = BayesianNet(observed=observed)
        with tf.compat.v1.variable_scope('h_for_q_z'):
            z_params = self.h_for_q_z(x)
        with tf.compat.v1.variable_scope('q_z_given_x'):
            q_z_given_x = self.q_z_given_x(**z_params)
            # Check if it's either a custom Distribution or TFP distribution
            is_custom = isinstance(q_z_given_x, Distribution)
            is_tfp = hasattr(q_z_given_x, '_distribution') and isinstance(q_z_given_x._distribution, tfd.Distribution)
            if not (is_custom or is_tfp):
                raise AssertionError('q_z_given_x must be an instance of Distribution or TFP distribution wrapper')
        with tf.name_scope('z'):
            z = net.add('z', q_z_given_x, n_samples=n_z,
                        group_ndims=self.z_group_ndims,
                        is_reparameterized=self.is_reparameterized,
                        flow=posterior_flow)
        return net

    @instance_reuse
    def model(self, z=None, x=None, n_z=None, n_x=None):
        observed = {k: v for k, v in [('z', z), ('x', x)] if v is not None}
        net = BayesianNet(observed=observed)
        with tf.name_scope('z'):
            z = net.add('z', self.p_z, n_samples=n_z,
                        group_ndims=self.z_group_ndims,
                        is_reparameterized=self.is_reparameterized)
        with tf.compat.v1.variable_scope('h_for_p_x'):
            x_params = self.h_for_p_x(z)
        with tf.compat.v1.variable_scope('p_x_given_z'):
            p_x_given_z = self.p_x_given_z(**x_params)
            # Check if it's either a custom Distribution or TFP distribution
            is_custom = isinstance(p_x_given_z, Distribution)
            is_tfp = hasattr(p_x_given_z, '_distribution') and isinstance(p_x_given_z._distribution, tfd.Distribution)
            if not (is_custom or is_tfp):
                raise AssertionError('p_x_given_z must be an instance of Distribution or TFP distribution wrapper')
        with tf.name_scope('x'):
            x = net.add('x', p_x_given_z, n_samples=n_x,
                        group_ndims=self.x_group_ndims)
        return net

    def chain(self, x, n_z=None, posterior_flow=None):
        with tf.name_scope('VAE.chain'):
            q_net = self.variational(x, n_z=n_z, posterior_flow=posterior_flow)

            if n_z is not None:
                latent_axis = 0
            else:
                latent_axis = None

            chain = q_net.variational_chain(
                lambda observed: self.model(n_z=n_z, n_x=None, **observed),
                latent_axis=latent_axis,
                observed={'x': x}
            )
        return chain

    def get_training_loss(self, x, n_z=None):
        with tf.name_scope('VAE.get_training_loss'):
            if n_z is not None:
                if is_tensor_object(n_z):
                    raise TypeError('Cannot choose the variational solver '
                                    'automatically for dynamic `n_z`')
                n_z = validate_n_samples_arg(n_z, 'n_z')

            chain = self.chain(x, n_z)
            z = chain.variational['z']

            if n_z is not None and n_z > 1:
                if z.is_reparameterized:
                    solver = chain.vi.training.iwae
                else:
                    solver = chain.vi.training.vimco
            else:
                if z.is_reparameterized:
                    solver = chain.vi.training.sgvb
                else:
                    solver = chain.vi.training.reinforce

            return tf.reduce_mean(solver())

    def reconstruct(self, x, n_z=None, n_x=None, posterior_flow=None):
        with tf.name_scope('VAE.reconstruct'):
            q_net = self.variational(x, n_z=n_z, posterior_flow=posterior_flow)
            model = self.model(z=q_net['z'], n_z=n_z, n_x=n_x)
            return model['x']

    def _forward(self, inputs, n_z=None, **kwargs):
        q_net = self.variational(inputs, z=None, n_z=n_z, **kwargs)
        return q_net['z']


class Lambda(VarScopeObject):
    def __init__(self, f, name=None, scope=None):
        super(Lambda, self).__init__(name=name, scope=scope)
        self._factory = f

    def _forward(self, inputs, **kwargs):
        return self._factory(inputs, **kwargs)

    def __call__(self, inputs, **kwargs):
        with reopen_variable_scope(self.variable_scope):
            with tf.name_scope('forward'):
                return self._forward(inputs, **kwargs)