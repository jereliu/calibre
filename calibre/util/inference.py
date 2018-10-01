"""Utility functions for posterior inference"""
import tensorflow as tf
from tensorflow_probability import edward2 as ed


def make_value_setter(**model_kwargs):
    """Creates a value-setting interceptor for VI under Edward2."""

    def set_values(f, *args, **kwargs):
        """Sets random variable values to its aligned value."""
        name = kwargs.get("name")
        if name in model_kwargs:
            kwargs["value"] = model_kwargs[name]
        return ed.interceptable(f)(*args, **kwargs)

    return set_values


def scalar_gaussian_vi_rv(name):
    """
    Creates a scalar Gaussian random variable for variational approximation.

    Args:
        name: (str) name of the output random variable.

    Returns:
        (ed.RandomVariable of float32) A normal scalar random variable.
    """
    mean = tf.get_variable(shape=[], name='{}_mean'.format(name))
    sdev = tf.exp(tf.get_variable(shape=[], name='{}_sdev'.format(name)))

    return ed.Normal(loc=mean, scale=sdev, name=name)
