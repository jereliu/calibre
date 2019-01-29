"""Utility and helper functions for building calibre models."""

import tensorflow as tf
import numpy as np

""" Link functions. """


def sparse_softmax(logits, temp, name='weight'):
    """Defines the sparse softmax function (i.e. normalized exp with temperature).

    That is,
        softmax = tf.exp(logits/temp) / tf.reduce_sum(tf.exp(logits/temp), axis)

    Args:
        logits: (tf.Tensor of float32) M base logits for N observations
            to be normalized over. It has dimension
            (batch_size, num_obs, num_model-1).
        temp: (tf.Tensor of float32) temperature parameter, it has size
            (batch_size, ).
        name: (str) Name of the output weights.
    Returns:
        A `Tensor`. Has the same type as `logits`. It has shape
            (batch_size, num_obs, num_model).

    Raises:
        ValueError: If dimension of logits is less than 1
    """
    # TODO(jereliu): identify efficient method for multivariate expit

    logits = tf.convert_to_tensor(logits)
    temp = tf.convert_to_tensor(temp)

    batch_sample_size = logits.get_shape().as_list()[:-1]

    # check dimension
    if logits.get_shape().ndims < 1:
        raise ValueError("Dimension of logits must be more than 1.")

    if logits.get_shape().ndims >= 2:
        # if dimension is 2 or more, adjust dimension for broadcasting
        dim_diff = logits.get_shape().ndims - temp.get_shape().ndims
        temp = tf.reshape(temp, shape=temp.get_shape().as_list() + [1] * dim_diff)

    # compute denominator
    log_exp_list = -logits / temp
    log_expits = log_exp_list - tf.reduce_logsumexp(log_exp_list, -1, keepdims=True)

    return tf.exp(log_expits, name=name)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
