"""Utility functions for posterior inference"""
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed

tfd = tfp.distributions

def make_value_setter(**model_kwargs):
    """Creates a value-setting interceptor for VI under Edward2."""

    def set_values(f, *args, **kwargs):
        """Sets random variable values to its aligned value."""
        name = kwargs.get("name")
        if name in model_kwargs:
            kwargs["value"] = model_kwargs[name]
        return ed.interceptable(f)(*args, **kwargs)

    return set_values


def scalar_gaussian_variational(name):
    """
    Creates a scalar Gaussian random variable for variational approximation.

    Args:
        name: (str) name of the output random variable.

    Returns:
        (ed.RandomVariable of float32) A normal scalar random variable.
    """
    mean = tf.get_variable(shape=[], name='{}_mean'.format(name))
    sdev = tf.exp(tf.get_variable(shape=[], name='{}_sdev'.format(name)))

    scalar_gaussian_rv = ed.Normal(loc=mean, scale=sdev, name=name)
    return scalar_gaussian_rv, mean, sdev


def scalar_gaussian_variational_sample(n_sample, mean, sdev):
    """Generates samples from GPR scalar Gaussian random variable.

    Args:
        n_sample: (int) number of samples to draw
        qf_mean: (tf.Tensor of float32) mean parameters for variational family
        qf_sdev: (tf.Tensor of float32) standard deviation for variational family

    Returns:
         (np.ndarray) sampled values.
    """

    """Generates f samples from GPR mean-field variational family."""
    scalar_gaussian_rv = tfd.Normal(loc=mean, scale=sdev)
    return scalar_gaussian_rv.sample(n_sample)
