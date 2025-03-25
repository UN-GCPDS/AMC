import os
import numpy as np
import tensorflow as tf
import math

from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils.conv_utils import normalize_data_format
import tensorflow.keras.backend as K

def sqrt_init(shape, dtype=None):
    value = (1 / np.sqrt(2)) * K.ones(shape)
    return value

def sanitizedInitGet(init):
    if init in ["sqrt_init"]:
        return sqrt_init
    else:
        return initializers.get(init)

def sanitizedInitSer(init):
    if init in [sqrt_init]:
        return "sqrt_init"
    else:
        return initializers.serialize(init)

def complex_standardization(input_centred, Vrr, Vii, Vri, layernorm=False, axis=-1):
    ndim = K.ndim(input_centred)
    input_dim = K.shape(input_centred)[axis] // 2
    variances_broadcast = [1] * ndim
    variances_broadcast[axis] = input_dim
    if layernorm:
        variances_broadcast[0] = K.shape(input_centred)[0]

    tau = Vrr + Vii
    delta = (Vrr * Vii) - (Vri ** 2)

    s = K.sqrt(delta)
    t = K.sqrt(tau + 2 * s)

    inverse_st = 1.0 / (s * t)
    Wrr = (Vii + s) * inverse_st
    Wii = (Vrr + s) * inverse_st
    Wri = -Vri * inverse_st

    broadcast_Wrr = K.reshape(Wrr, variances_broadcast)
    broadcast_Wri = K.reshape(Wri, variances_broadcast)
    broadcast_Wii = K.reshape(Wii, variances_broadcast)

    cat_W_4_real = K.concatenate([broadcast_Wrr, broadcast_Wii], axis=axis)
    cat_W_4_imag = K.concatenate([broadcast_Wri, broadcast_Wri], axis=axis)

    if (axis == 1 and ndim != 3) or ndim == 2:
        centred_real = input_centred[:, :input_dim]
        centred_imag = input_centred[:, input_dim:]
    elif ndim == 3:
        centred_real = input_centred[:, :, :input_dim]
        centred_imag = input_centred[:, :, input_dim:]
    elif axis == -1 and ndim == 4:
        centred_real = input_centred[:, :, :, :input_dim]
        centred_imag = input_centred[:, :, :, input_dim:]
    elif axis == -1 and ndim == 5:
        centred_real = input_centred[:, :, :, :, :input_dim]
        centred_imag = input_centred[:, :, :, :, input_dim:]
    else:
        raise ValueError(
            'Incorrect Batchnorm combination of axis and dimensions. axis '
            'should be either 1 or -1. '
            'axis: ' + str(axis) + '; ndim: ' + str(ndim) + '.'
        )
    rolled_input = K.concatenate([centred_imag, centred_real], axis=axis)

    output = cat_W_4_real * input_centred + cat_W_4_imag * rolled_input

    return output

def ComplexBN(input_centred, Vrr, Vii, Vri, beta, gamma_rr, gamma_ri, gamma_ii, scale=True, center=True, layernorm=False, axis=-1):
    ndim = K.ndim(input_centred)
    input_dim = K.shape(input_centred)[axis] // 2
    if scale:
        gamma_broadcast_shape = [1] * ndim
        gamma_broadcast_shape[axis] = input_dim
    if center:
        broadcast_beta_shape = [1] * ndim
        broadcast_beta_shape[axis] = input_dim * 2

    if scale:
        standardized_output = complex_standardization(input_centred, Vrr, Vii, Vri, layernorm, axis=axis)

        broadcast_gamma_rr = K.reshape(gamma_rr, gamma_broadcast_shape)
        broadcast_gamma_ri = K.reshape(gamma_ri, gamma_broadcast_shape)
        broadcast_gamma_ii = K.reshape(gamma_ii, gamma_broadcast_shape)

        cat_gamma_4_real = K.concatenate([broadcast_gamma_rr, broadcast_gamma_ii], axis=axis)
        cat_gamma_4_imag = K.concatenate([broadcast_gamma_ri, broadcast_gamma_ri], axis=axis)
        if (axis == 1 and ndim != 3) or ndim == 2:
            centred_real = standardized_output[:, :input_dim]
            centred_imag = standardized_output[:, input_dim:]
        elif ndim == 3:
            centred_real = standardized_output[:, :, :input_dim]
            centred_imag = standardized_output[:, :, input_dim:]
        elif axis == -1 and ndim == 4:
            centred_real = standardized_output[:, :, :, :input_dim]
            centred_imag = standardized_output[:, :, :, input_dim:]
        elif axis == -1 and ndim == 5:
            centred_real = standardized_output[:, :, :, :, :input_dim]
            centred_imag = standardized_output[:, :, :, :, input_dim:]
        else:
            raise ValueError(
                'Incorrect Batchnorm combination of axis and dimensions. axis'
                ' should be either 1 or -1. '
                'axis: ' + str(axis) + '; ndim: ' + str(ndim) + '.'
            )
        rolled_standardized_output = K.concatenate([centred_imag, centred_real], axis=axis)
        if center:
            broadcast_beta = K.reshape(beta, broadcast_beta_shape)
            return cat_gamma_4_real * standardized_output + cat_gamma_4_imag * rolled_standardized_output + broadcast_beta
        else:
            return cat_gamma_4_real * standardized_output + cat_gamma_4_imag * rolled_standardized_output
    else:
        if center:
            broadcast_beta = K.reshape(beta, broadcast_beta_shape)
            return input_centred + broadcast_beta
        else:
            return input_centred

class ComplexBatchNormalization(Layer):
    def __init__(self, axis=-1, momentum=0.9, epsilon=1e-4, center=True, scale=True, beta_initializer='zeros', gamma_diag_initializer='sqrt_init', gamma_off_initializer='zeros', moving_mean_initializer='zeros', moving_variance_initializer='sqrt_init', moving_covariance_initializer='zeros', beta_regularizer=None, gamma_diag_regularizer=None, gamma_off_regularizer=None, beta_constraint=None, gamma_diag_constraint=None, gamma_off_constraint=None, **kwargs):
        super(ComplexBatchNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = sanitizedInitGet(beta_initializer)
        self.gamma_diag_initializer = sanitizedInitGet(gamma_diag_initializer)
        self.gamma_off_initializer = sanitizedInitGet(gamma_off_initializer)
        self.moving_mean_initializer = sanitizedInitGet(moving_mean_initializer)
        self.moving_variance_initializer = sanitizedInitGet(moving_variance_initializer)
        self.moving_covariance_initializer = sanitizedInitGet(moving_covariance_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_diag_regularizer = regularizers.get(gamma_diag_regularizer)
        self.gamma_off_regularizer = regularizers.get(gamma_off_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_diag_constraint = constraints.get(gamma_diag_constraint)
        self.gamma_off_constraint = constraints.get(gamma_off_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of input tensor should have a defined dimension but the layer received an input with shape ' + str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape), axes={self.axis: dim})

        param_shape = (input_shape[self.axis] // 2,)

        if self.scale:
            self.gamma_rr = self.add_weight(shape=param_shape, name='gamma_rr', initializer=self.gamma_diag_initializer, regularizer=self.gamma_diag_regularizer, constraint=self.gamma_diag_constraint)
            self.gamma_ii = self.add_weight(shape=param_shape, name='gamma_ii', initializer=self.gamma_diag_initializer, regularizer=self.gamma_diag_regularizer, constraint=self.gamma_diag_constraint)
            self.gamma_ri = self.add_weight(shape=param_shape, name='gamma_ri', initializer=self.gamma_off_initializer, regularizer=self.gamma_off_regularizer, constraint=self.gamma_off_constraint)
            self.moving_Vrr = self.add_weight(shape=param_shape, initializer=self.moving_variance_initializer, name='moving_Vrr', trainable=False)
            self.moving_Vii = self.add_weight(shape=param_shape, initializer=self.moving_variance_initializer, name='moving_Vii', trainable=False)
            self.moving_Vri = self.add_weight(shape=param_shape, initializer=self.moving_covariance_initializer, name='moving_Vri', trainable=False)
        else:
            self.gamma_rr = None
            self.gamma_ii = None
            self.gamma_ri = None
            self.moving_Vrr = None
            self.moving_Vii = None
            self.moving_Vri = None

        if self.center:
            self.beta = self.add_weight(shape=(input_shape[self.axis],), name='beta', initializer=self.beta_initializer, regularizer=self.beta_regularizer, constraint=self.beta_constraint)
            self.moving_mean = self.add_weight(shape=(input_shape[self.axis],), initializer=self.moving_mean_initializer, name='moving_mean', trainable=False)
        else:
            self.beta = None
            self.moving_mean = None

        self.built = True
    
    def call(self, inputs, training=None):
        # Handle training=None by inferring from K.learning_phase()
        if training is None:
            training = False
    
        input_shape = K.int_shape(inputs)
        ndim = len(input_shape)
        reduction_axes = list(range(ndim))
        del reduction_axes[self.axis]
        input_dim = input_shape[self.axis] // 2
    
        # Compute mean and broadcast it
        mu = K.mean(inputs, axis=reduction_axes)
        broadcast_mu_shape = [1] * len(input_shape)
        broadcast_mu_shape[self.axis] = input_shape[self.axis]
        broadcast_mu = K.reshape(mu, broadcast_mu_shape)
    
        # Center the input if required
        if self.center:
            input_centred = inputs - broadcast_mu
        else:
            input_centred = inputs
    
        # Compute squared values for variance
        centred_squared = input_centred ** 2
        if (self.axis == 1 and ndim != 3) or ndim == 2:
            centred_squared_real = centred_squared[:, :input_dim]
            centred_squared_imag = centred_squared[:, input_dim:]
            centred_real = input_centred[:, :input_dim]
            centred_imag = input_centred[:, input_dim:]
        elif ndim == 3:
            centred_squared_real = centred_squared[:, :, :input_dim]
            centred_squared_imag = centred_squared[:, :, input_dim:]
            centred_real = input_centred[:, :, :input_dim]
            centred_imag = input_centred[:, :, input_dim:]
        elif self.axis == -1 and ndim == 4:
            centred_squared_real = centred_squared[:, :, :, :input_dim]
            centred_squared_imag = centred_squared[:, :, :, input_dim:]
            centred_real = input_centred[:, :, :, :input_dim]
            centred_imag = input_centred[:, :, :, input_dim:]
        elif self.axis == -1 and ndim == 5:
            centred_squared_real = centred_squared[:, :, :, :, :input_dim]
            centred_squared_imag = centred_squared[:, :, :, :, input_dim:]
            centred_real = input_centred[:, :, :, :, :input_dim]
            centred_imag = input_centred[:, :, :, :, input_dim:]
        else:
            raise ValueError(
                'Incorrect Batchnorm combination of axis and dimensions. '
                'axis should be either 1 or -1. '
                f'axis: {self.axis}; ndim: {ndim}.'
            )
    
        # Compute variances and covariances
        if self.scale:
            Vrr = K.mean(centred_squared_real, axis=reduction_axes) + self.epsilon
            Vii = K.mean(centred_squared_imag, axis=reduction_axes) + self.epsilon
            Vri = K.mean(centred_real * centred_imag, axis=reduction_axes) + self.epsilon
        elif self.center:
            Vrr = None
            Vii = None
            Vri = None
        else:
            raise ValueError('Error. Both scale and center in batchnorm are set to False.')
    
        # Normalize the input
        input_bn = ComplexBN(
            input_centred, Vrr, Vii, Vri,
            self.beta, self.gamma_rr, self.gamma_ri, self.gamma_ii,
            self.scale, self.center, axis=self.axis
        )
    
        # Update moving averages during training
        if training:
            if self.center:
                self.moving_mean.assign(self.momentum * self.moving_mean + (1 - self.momentum) * mu)
            if self.scale:
                self.moving_Vrr.assign(self.momentum * self.moving_Vrr + (1 - self.momentum) * Vrr)
                self.moving_Vii.assign(self.momentum * self.moving_Vii + (1 - self.momentum) * Vii)
                self.moving_Vri.assign(self.momentum * self.moving_Vri + (1 - self.momentum) * Vri)
            return input_bn
        else:
            # Use moving averages during inference
            if self.center:
                inference_centred = inputs - K.reshape(self.moving_mean, broadcast_mu_shape)
            else:
                inference_centred = inputs
            return ComplexBN(
                inference_centred, self.moving_Vrr, self.moving_Vii, self.moving_Vri,
                self.beta, self.gamma_rr, self.gamma_ri, self.gamma_ii,
                self.scale, self.center, axis=self.axis
            )

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': sanitizedInitSer(self.beta_initializer),
            'gamma_diag_initializer': sanitizedInitSer(self.gamma_diag_initializer),
            'gamma_off_initializer': sanitizedInitSer(self.gamma_off_initializer),
            'moving_mean_initializer': sanitizedInitSer(self.moving_mean_initializer),
            'moving_variance_initializer': sanitizedInitSer(self.moving_variance_initializer),
            'moving_covariance_initializer': sanitizedInitSer(self.moving_covariance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_diag_regularizer': regularizers.serialize(self.gamma_diag_regularizer),
            'gamma_off_regularizer': regularizers.serialize(self.gamma_off_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_diag_constraint': constraints.serialize(self.gamma_diag_constraint),
            'gamma_off_constraint': constraints.serialize(self.gamma_off_constraint),
        }
        base_config = super(ComplexBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def compute_fans(shape):
    """
    Computes the number of input and output units for a weight shape.
    This is required for Glorot (Xavier) and He initialization.
    """
    if len(shape) < 1:  # Scalar case
        fan_in = fan_out = 1
    elif len(shape) == 1:  # Bias case
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:  # Dense layer case
        fan_in, fan_out = shape
    else:  # Convolutional layers case
        receptive_field_size = np.prod(shape[:-2])
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out

@register_keras_serializable()
class IndependentFilters(Initializer):
    """
    Initializes real-valued kernels that are as independent as possible
    while respecting the Glorot or He criterion.
    """
    def __init__(self, kernel_size, input_dim, weight_dim, nb_filters=None, criterion='glorot', seed=None):
        assert len(kernel_size) == weight_dim and weight_dim in {0, 1, 2, 3}
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        self.criterion = criterion
        self.seed = 1337 if seed is None else seed

    def __call__(self, shape, dtype=None):
        if self.nb_filters is not None:
            num_rows = self.nb_filters * self.input_dim
            num_cols = np.prod(self.kernel_size)
        else:
            num_rows = self.input_dim
            num_cols = self.kernel_size[-1]

        flat_shape = (num_rows, num_cols)
        rng = np.random.default_rng(self.seed)  # Updated random number generator
        x = rng.uniform(size=flat_shape)
        u, _, v = np.linalg.svd(x)
        orthogonal_x = np.dot(u, np.dot(np.eye(num_rows, num_cols), v.T))

        if self.nb_filters is not None:
            independent_filters = np.reshape(orthogonal_x, (num_rows,) + tuple(self.kernel_size))
            fan_in, fan_out = compute_fans(
                tuple(self.kernel_size) + (self.input_dim, self.nb_filters))
        else:
            independent_filters = orthogonal_x
            fan_in, fan_out = (self.input_dim, self.kernel_size[-1])

        if self.criterion == 'glorot':
            desired_var = 2. / (fan_in + fan_out)
        elif self.criterion == 'he':
            desired_var = 2. / fan_in
        else:
            raise ValueError('Invalid criterion: ' + self.criterion)

        multip_constant = np.sqrt(desired_var / np.var(independent_filters))
        scaled_indep = multip_constant * independent_filters

        if self.weight_dim == 2 and self.nb_filters is None:
            weight = scaled_indep
        else:
            kernel_shape = tuple(self.kernel_size) + (self.input_dim, self.nb_filters)
            if self.weight_dim == 1:
                transpose_shape = (1, 0)
            elif self.weight_dim == 2 and self.nb_filters is not None:
                transpose_shape = (1, 2, 0)
            elif self.weight_dim == 3 and self.nb_filters is not None:
                transpose_shape = (1, 2, 3, 0)
            weight = np.transpose(scaled_indep, transpose_shape)
            weight = np.reshape(weight, kernel_shape)

        return tf.convert_to_tensor(weight, dtype=dtype)

    def get_config(self):
        return {
            'nb_filters': self.nb_filters,
            'kernel_size': self.kernel_size,
            'input_dim': self.input_dim,
            'weight_dim': self.weight_dim,
            'criterion': self.criterion,
            'seed': self.seed
        }


@register_keras_serializable()
class ComplexIndependentFilters(Initializer):
    """
    Initializes complex-valued kernels that are as independent as possible
    while respecting the Glorot or He criterion.
    """
    def __init__(self, kernel_size, input_dim, weight_dim, nb_filters=None, criterion='glorot', seed=None):
        assert len(kernel_size) == weight_dim and weight_dim in {0, 1, 2, 3}
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        self.criterion = criterion
        self.seed = 1337 if seed is None else seed

    def __call__(self, shape, dtype=None):
        if self.nb_filters is not None:
            num_rows = self.nb_filters * self.input_dim
            num_cols = np.prod(self.kernel_size)
        else:
            num_rows = self.input_dim
            num_cols = self.kernel_size[-1]

        flat_shape = (int(num_rows), int(num_cols))
        rng = np.random.default_rng(self.seed)  # Updated random number generator
        r = rng.uniform(size=flat_shape)
        i = rng.uniform(size=flat_shape)
        z = r + 1j * i
        u, _, v = np.linalg.svd(z)
        unitary_z = np.dot(u, np.dot(np.eye(int(num_rows), int(num_cols)), np.conjugate(v).T))
        real_unitary = unitary_z.real
        imag_unitary = unitary_z.imag

        if self.nb_filters is not None:
            indep_real = np.reshape(real_unitary, (num_rows,) + tuple(self.kernel_size))
            indep_imag = np.reshape(imag_unitary, (num_rows,) + tuple(self.kernel_size))
            fan_in, fan_out = compute_fans(
                tuple(self.kernel_size) + (int(self.input_dim), self.nb_filters)
            )
        else:
            indep_real = real_unitary
            indep_imag = imag_unitary
            fan_in, fan_out = (int(self.input_dim), self.kernel_size[-1])

        if self.criterion == 'glorot':
            desired_var = 1. / (fan_in + fan_out)
        elif self.criterion == 'he':
            desired_var = 1. / fan_in
        else:
            raise ValueError('Invalid criterion: ' + self.criterion)

        multip_real = np.sqrt(desired_var / np.var(indep_real))
        multip_imag = np.sqrt(desired_var / np.var(indep_imag))
        scaled_real = multip_real * indep_real
        scaled_imag = multip_imag * indep_imag

        if self.weight_dim == 2 and self.nb_filters is None:
            weight_real = scaled_real
            weight_imag = scaled_imag
        else:
            kernel_shape = tuple(self.kernel_size) + (int(self.input_dim), self.nb_filters)
            if self.weight_dim == 1:
                transpose_shape = (1, 0)
            elif self.weight_dim == 2 and self.nb_filters is not None:
                transpose_shape = (1, 2, 0)
            elif self.weight_dim == 3 and self.nb_filters is not None:
                transpose_shape = (1, 2, 3, 0)
            weight_real = np.transpose(scaled_real, transpose_shape)
            weight_imag = np.transpose(scaled_imag, transpose_shape)
            weight_real = np.reshape(weight_real, kernel_shape)
            weight_imag = np.reshape(weight_imag, kernel_shape)

        weight = np.concatenate([weight_real, weight_imag], axis=-1)
        return tf.convert_to_tensor(weight, dtype=dtype)

    def get_config(self):
        return {
            'nb_filters': self.nb_filters,
            'kernel_size': self.kernel_size,
            'input_dim': self.input_dim,
            'weight_dim': self.weight_dim,
            'criterion': self.criterion,
            'seed': self.seed
        }


@register_keras_serializable()
class ComplexInit(Initializer):
    """
    Standard complex initialization using either the Glorot or He criterion.
    """
    def __init__(self, kernel_size, input_dim, weight_dim, nb_filters=None, criterion='glorot', seed=None):
        assert len(kernel_size) == weight_dim and weight_dim in {0, 1, 2, 3}
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        self.criterion = criterion
        self.seed = 1337 if seed is None else seed

    def __call__(self, shape, dtype=None):
        if self.nb_filters is not None:
            kernel_shape = shape
        else:
            kernel_shape = (int(self.input_dim), self.kernel_size[-1])

        fan_in, fan_out = compute_fans(kernel_shape)

        if self.criterion == 'glorot':
            s = 1. / (fan_in + fan_out)
        elif self.criterion == 'he':
            s = 1. / fan_in
        else:
            raise ValueError('Invalid criterion: ' + self.criterion)

        rng = np.random.default_rng(self.seed)  # Updated random number generator
        modulus = rng.rayleigh(scale=s, size=kernel_shape)
        phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
        weight_real = modulus * np.cos(phase)
        weight_imag = modulus * np.sin(phase)
        weight = np.concatenate([weight_real, weight_imag], axis=-1)
        return tf.convert_to_tensor(weight, dtype=dtype)

    def get_config(self):
        return {
            'nb_filters': self.nb_filters,
            'kernel_size': self.kernel_size,
            'input_dim': self.input_dim,
            'weight_dim': self.weight_dim,
            'criterion': self.criterion,
            'seed': self.seed
        }


@register_keras_serializable()
class SqrtInit(Initializer):
    """
    Initializer that generates tensors with values scaled by 1/sqrt(2).
    """
    def __call__(self, shape, dtype=None):
        return tf.constant(1 / tf.sqrt(2.0), shape=shape, dtype=dtype)


# Aliases:
sqrt_init = SqrtInit
independent_filters = IndependentFilters
complex_init = ComplexInit


def map_data_format(f):
    if f == 'channels_last':
        return 'NWC'
    elif f == 'channels_first':
        return 'NCW'
def conv2d_transpose(
        inputs,
        filters,  # Renamed from 'filter' to avoid built-in conflict
        kernel_size=None,
        strides=(1, 1),
        padding="SAME",
        output_padding=None,
        data_format="channels_last"):
    """Compatibility layer for K.conv2d_transpose."""
    input_shape = tf.shape(inputs)
    batch_size = input_shape[0]
    if data_format == 'channels_first':
        h_axis, w_axis = 2, 3
    else:
        h_axis, w_axis = 1, 2

    height, width = input_shape[h_axis], input_shape[w_axis]
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = strides

    # Infer the dynamic output shape:
    out_height = conv_utils.deconv_length(height, stride_h, kernel_h, padding, output_padding)
    out_width = conv_utils.deconv_length(width, stride_w, kernel_w, padding, output_padding)
    if data_format == 'channels_first':
        output_shape = (batch_size, filters, out_height, out_width)
    else:
        output_shape = (batch_size, out_height, out_width, filters)

    filters = tf.transpose(filters, (0, 1, 3, 2))  # Permute dimensions
    return tf.nn.conv2d_transpose(
        inputs, filters, output_shape, strides, padding=padding, data_format=data_format
    )


def conv_transpose_output_length(
        input_length, filter_size, padding, stride, dilation=1, output_padding=None):
    """Rearrange arguments for compatibility with conv_output_length."""
    if dilation != 1:
        raise ValueError("Dilation must be 1 for transposed convolution. Got dilation = {}".format(dilation))
    return conv_utils.deconv_length(input_length, stride, filter_size, padding, output_padding)


def sanitizedInitGet(init):
    """Get the appropriate initializer."""
    if init in ["sqrt_init"]:
        return sqrt_init
    elif init in ["complex", "complex_independent", "glorot_complex", "he_complex"]:
        return init
    else:
        return initializers.get(init)


def sanitizedInitSer(init):
    """Serialize the initializer."""
    if init in [sqrt_init]:
        return "sqrt_init"
    elif init == "complex" or isinstance(init, ComplexInit):
        return "complex"
    elif init == "complex_independent" or isinstance(init, ComplexIndependentFilters):
        return "complex_independent"
    else:
        return initializers.serialize(init)


@register_keras_serializable()
class ComplexConv(Layer):
    """Abstract nD complex convolution layer."""

    def __init__(
            self,
            rank,
            filters,
            kernel_size,
            strides=1,
            padding="valid",
            data_format=None,
            dilation_rate=1,
            activation=None,
            use_bias=True,
            normalize_weight=False,
            kernel_initializer="complex",
            bias_initializer="zeros",
            gamma_diag_initializer=sqrt_init,
            gamma_off_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            gamma_diag_regularizer=None,
            gamma_off_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            gamma_diag_constraint=None,
            gamma_off_constraint=None,
            init_criterion="he",
            seed=None,
            spectral_parametrization=False,
            transposed=False,
            epsilon=1e-7,
            **kwargs):
        super(ComplexConv, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, "kernel_size")
        self.strides = conv_utils.normalize_tuple(strides, rank, "strides")
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, "dilation_rate")
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.normalize_weight = normalize_weight
        self.init_criterion = init_criterion
        self.spectral_parametrization = spectral_parametrization
        self.transposed = transposed
        self.epsilon = epsilon
        self.kernel_initializer = sanitizedInitGet(kernel_initializer)
        self.bias_initializer = sanitizedInitGet(bias_initializer)
        self.gamma_diag_initializer = sanitizedInitGet(gamma_diag_initializer)
        self.gamma_off_initializer = sanitizedInitGet(gamma_off_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.gamma_diag_regularizer = regularizers.get(gamma_diag_regularizer)
        self.gamma_off_regularizer = regularizers.get(gamma_off_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.gamma_diag_constraint = constraints.get(gamma_diag_constraint)
        self.gamma_off_constraint = constraints.get(gamma_off_constraint)
        self.seed = seed if seed is not None else np.random.randint(1, 10e6)
        self.input_spec = InputSpec(ndim=self.rank + 2)

        # The following are initialized later
        self.kernel_shape = None
        self.kernel = None
        self.gamma_rr = None
        self.gamma_ii = None
        self.gamma_ri = None
        self.bias = None

    def build(self, input_shape):
        """Build the layer."""
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError("The channel dimension of the inputs should be defined. Found `None`.")
        input_dim = input_shape[channel_axis] // 2  # Divide by 2 for real and complex input.

        if self.transposed:
            self.kernel_shape = self.kernel_size + (self.filters, input_dim)
        else:
            self.kernel_shape = self.kernel_size + (input_dim, self.filters)

        if self.kernel_initializer in {"complex", "complex_independent"}:
            kls = {
                "complex": ComplexInit,
                "complex_independent": ComplexIndependentFilters,
            }[self.kernel_initializer]
            kern_init = kls(
                kernel_size=self.kernel_size,
                input_dim=input_dim,
                weight_dim=self.rank,
                nb_filters=self.filters,
                criterion=self.init_criterion,
            )
        else:
            kern_init = self.kernel_initializer

        self.kernel = self.add_weight(
            name="kernel",
            shape=self.kernel_shape,
            initializer=kern_init,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        if self.normalize_weight:
            gamma_shape = (input_dim * self.filters,)
            self.gamma_rr = self.add_weight(
                shape=gamma_shape,
                name="gamma_rr",
                initializer=self.gamma_diag_initializer,
                regularizer=self.gamma_diag_regularizer,
                constraint=self.gamma_diag_constraint,
            )
            self.gamma_ii = self.add_weight(
                shape=gamma_shape,
                name="gamma_ii",
                initializer=self.gamma_diag_initializer,
                regularizer=self.gamma_diag_regularizer,
                constraint=self.gamma_diag_constraint,
            )
            self.gamma_ri = self.add_weight(
                shape=gamma_shape,
                name="gamma_ri",
                initializer=self.gamma_off_initializer,
                regularizer=self.gamma_off_regularizer,
                constraint=self.gamma_off_constraint,
            )
        else:
            self.gamma_rr = None
            self.gamma_ii = None
            self.gamma_ri = None

        if self.use_bias:
            bias_shape = (2 * self.filters,)
            self.bias = self.add_weight(
                name="bias",
                shape=bias_shape,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None

        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim * 2})
        self.built = True

    def call(self, inputs, **kwargs):
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1
        input_dim = K.shape(inputs)[channel_axis] // 2
    
        if self.transposed:
            if self.rank == 1:
                f_real = self.kernel[:, :self.filters, :]
                f_imag = self.kernel[:, self.filters:, :]
            elif self.rank == 2:
                f_real = self.kernel[:, :, :self.filters, :]
                f_imag = self.kernel[:, :, self.filters:, :]
            elif self.rank == 3:
                f_real = self.kernel[:, :, :, :self.filters, :]
                f_imag = self.kernel[:, :, :, self.filters:, :]
        else:
            if self.rank == 1:
                f_real = self.kernel[:, :, :self.filters]
                f_imag = self.kernel[:, :, self.filters:]
            elif self.rank == 2:
                f_real = self.kernel[:, :, :, :self.filters]
                f_imag = self.kernel[:, :, :, self.filters:]
            elif self.rank == 3:
                f_real = self.kernel[:, :, :, :, :self.filters]
                f_imag = self.kernel[:, :, :, :, self.filters:]
    
        # Prepare arguments for convolution
        if self.rank == 1:
            strides = self.strides[0]  # Use the first element for 1D convolution
            padding = self.padding.upper()  # Ensure uppercase for TensorFlow
            data_format = self.data_format
            dilation_rate = self.dilation_rate[0] if self.dilation_rate else 1
    
            # Perform complex convolution
            cat_kernels_4_real = tf.concat([f_real, -f_imag], axis=-2)
            cat_kernels_4_imag = tf.concat([f_imag, f_real], axis=-2)
            cat_kernels_4_complex = tf.concat([cat_kernels_4_real, cat_kernels_4_imag], axis=-1)

            
            # Call tf.nn.conv1d with the correct arguments
            output = tf.nn.conv1d(
                input=inputs,
                filters=cat_kernels_4_complex,
                stride=strides,
                padding=padding,
                data_format=map_data_format(data_format),
                dilations=dilation_rate,
            )
        else:
            raise ValueError("Only 1D convolution is supported in this example.")
    
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias, data_format=map_data_format(self.data_format))
    
        if self.activation is not None:
            output = self.activation(output)
    
        return output
    def compute_output_shape(self, input_shape):
        """Compute the output shape."""
        if self.transposed:
            outputLengthFunc = conv_transpose_output_length
        else:
            outputLengthFunc = conv_utils.conv_output_length

        if self.data_format == "channels_last":
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = outputLengthFunc(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i],
                )
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (2 * self.filters,)
        else:
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = outputLengthFunc(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i],
                )
                new_space.append(new_dim)
            return (input_shape[0],) + (2 * self.filters,) + tuple(new_space)

    def get_config(self):
        """Get the layer configuration."""
        config = {
            "rank": self.rank,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "normalize_weight": self.normalize_weight,
            "kernel_initializer": sanitizedInitSer(self.kernel_initializer),
            "bias_initializer": sanitizedInitSer(self.bias_initializer),
            "gamma_diag_initializer": sanitizedInitSer(self.gamma_diag_initializer),
            "gamma_off_initializer": sanitizedInitSer(self.gamma_off_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "gamma_diag_regularizer": regularizers.serialize(self.gamma_diag_regularizer),
            "gamma_off_regularizer": regularizers.serialize(self.gamma_off_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "gamma_diag_constraint": constraints.serialize(self.gamma_diag_constraint),
            "gamma_off_constraint": constraints.serialize(self.gamma_off_constraint),
            "init_criterion": self.init_criterion,
            "spectral_parametrization": self.spectral_parametrization,
            "transposed": self.transposed,
            "epsilon": self.epsilon,
        }
        base_config = super(ComplexConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable()
class ComplexConv1D(ComplexConv):
    """1D complex convolution layer."""

    def __init__(
            self,
            filters,
            kernel_size,
            strides=1,
            padding="valid",
            data_format=None,
            dilation_rate=1,
            activation=None,
            use_bias=True,
            kernel_initializer="complex",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            seed=None,
            init_criterion="he",
            spectral_parametrization=False,
            transposed=False,
            **kwargs,
    ):
        super(ComplexConv1D, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format="channels_last",
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            init_criterion=init_criterion,
            spectral_parametrization=spectral_parametrization,
            transposed=transposed,
            **kwargs,
        )

    def get_config(self):
        """Get the layer configuration."""
        config = super(ComplexConv1D, self).get_config()
        config.pop("rank")
        return config


# Aliases
ComplexConvolution1D = ComplexConv1D

# -*-coding:utf-8-*-
# @Time : 2024/3/7 15:59



def channel_shuffle(x):
    in_channels, D = x.shape[1], x.shape[2]
    channels_per_group = in_channels // 2
    pre_shape = (-1, 2, channels_per_group, D)
    dim = (0, 2, 1, 3)
    later_shape = (-1, in_channels, D)

    x = tf.reshape(x, pre_shape)
    x = tf.transpose(x, perm=dim)
    x = tf.reshape(x, later_shape)
    return x

class layer_channel_shuffle(Layer):
    def call(self, x):
        return channel_shuffle(x)

def dwconv_mobile(x, neurons, ks=5):
    x = SeparableConv1D(int(2 * neurons), ks, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = layer_channel_shuffle()(x)
    return x

def DSECA(inputs, b=1, gamma=2, channel=32):
    # in_channel = inputs.shape[-1]  # Get the number of input feature channels
    in_channel = channel*2
    print(f'channel: {channel} - in_channel: {in_channel}')
    kernel_size = int(abs((math.log(int(in_channel), 2) + b) / gamma))
    if kernel_size % 2 == 0:
        kernel_size += 1  

    x_GAP = GlobalAveragePooling1D()(inputs)
    print(f"After GlobalAveragePooling1D - x_GAP.shape: {x_GAP.shape}")
    
    x_GAP = Reshape(target_shape=(in_channel, 1))(x_GAP)
    print(f"After Reshape to (in_channel, 1) - x_GAP.shape: {x_GAP.shape}")
    
    x_GAP = Conv1D(filters=1, kernel_size=kernel_size, padding="same", strides=2, use_bias=False)(x_GAP)
    print(f"After Conv1D - x_GAP.shape: {x_GAP.shape}")
    
    x_GAP = Activation('sigmoid')(x_GAP)
    print(f"After Activation - x_GAP.shape: {x_GAP.shape}")
    
    x_GAP = Reshape(target_shape=(1, channel))(x_GAP)
    print(f"After final Reshape to (1, channel) - x_GAP.shape: {x_GAP.shape}")


    x_GMP = GlobalMaxPooling1D()(inputs)
    x_GMP = Reshape(target_shape=(in_channel, 1))(x_GMP)
    x_GMP = Conv1D(filters=1, kernel_size=kernel_size, padding="same", strides=2, use_bias=False)(x_GMP)
    x_GMP = Activation('sigmoid')(x_GMP)
    x_GMP = Reshape(target_shape=(1, channel))(x_GMP)

    x_Mask = concatenate([x_GAP, x_GMP], axis=2)
    x_Mask = Activation('sigmoid')(x_Mask)
    
    print(f"DSECA - inputs shape: {inputs.shape}, x_Mask shape: {x_Mask.shape}")
    
    x = Multiply()([inputs, x_Mask])
    return x


def UlNN(weights=None, input_shape=[128, 2], deepth=6, n_neuron=16, ks=5):
    input = Input(input_shape, name='input1')
    
    print("Model input shape:", input.shape)
    
    # Complex Convolution
    x1 = ComplexConv1D(n_neuron, ks, padding='same')(input)
    x1 = ComplexBatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    print("Shape after initial complex conv:", x1.shape)

    for i in range(deepth):
        x1 = dwconv_mobile(x1, neurons=n_neuron, ks=5)
        x1 = DSECA(x1, b=1, gamma=2, channel=n_neuron)
        
        print(f"After layer {i+1} - x1.shape: {x1.shape}")
        
        if i == 3:
            f3_ave = GlobalAveragePooling1D()(x1)
            f3_max = GlobalMaxPooling1D()(x1)
            print(f"f3_ave.shape: {f3_ave.shape}, f3_max.shape: {f3_max.shape}")
        if i == 4:
            f4_ave = GlobalAveragePooling1D()(x1)
            f4_max = GlobalMaxPooling1D()(x1)
            print(f"f4_ave.shape: {f4_ave.shape}, f4_max.shape: {f4_max.shape}")
        if i == 5:
            f5_ave = GlobalAveragePooling1D()(x1)
            f5_max = GlobalMaxPooling1D()(x1)
            print(f"f5_ave.shape: {f5_ave.shape}, f5_max.shape: {f5_max.shape}")

    print("Pooling features collected")

    f1 = Add()([f3_ave, f4_ave])
    f1 = Add()([f1, f5_ave])
    print(f"f1.shape: {f1.shape}")

    f2 = Add()([f3_max, f4_max])
    f2 = Add()([f2, f5_max])
    print(f"f2.shape: {f2.shape}")

    f11 = Reshape(target_shape=(f1.shape[-1], 1))(f1)
    f22 = Reshape(target_shape=(f1.shape[-1], 1))(f2)
    
    print(f"After reshape - f11.shape: {f11.shape}, f22.shape: {f22.shape}")

    f11 = Conv1D(filters=1, kernel_size=5, padding="same", strides=2, use_bias=False)(f11)
    f11 = Activation('relu')(f11)
    f22 = Conv1D(filters=1, kernel_size=5, padding="same", strides=2, use_bias=False)(f22)
    f22 = Activation('relu')(f22)

    print(f"After Conv1D - f11.shape: {f11.shape}, f22.shape: {f22.shape}")

    f = concatenate([f11, f22], axis=1)
    
    print(f"After concatenate - f.shape: {f.shape}")

    f = Reshape(target_shape=(-1, f.shape[-2]))(f)

    print(f"After final reshape - f.shape: {f.shape}")

    f = Dense(11)(f)
    c = Activation('softmax', name='modulation')(f)
    print(c.shape)
    model = Model(inputs=input, outputs=c)
    total_params = model.count_params()
    print("Total number of parameters:", total_params)

    if weights is not None:
        model.load_weights(weights)

    return model


if __name__ == '__main__':
    """0. Set Hyperparameters"""
    filename = 'ULNN'
    model = UlNN(input_shape=[128, 2], deepth=6, n_neuron=16, ks=5)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
    model.summary()

