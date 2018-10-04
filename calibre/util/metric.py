"""Utility functions for calibration calibration."""
import tensorflow as tf
import numpy as np


def ecdf_eval(Y_obs, Y_sample, axis=-1):
    """Computes empirical cdf calibration (i.e. P(Y<y) ) for M samples.

    Args:
        Y_obs: (np.ndarray or tf.Tensor) N observations of dim (N, 1), dtype float32
        Y_sample: (np.ndarray or tf.Tensor) Samples of size M corresponding
        to the N observations. dim (N, M), dtype float32
        axis: (int) axis average over.

    Returns:
        (np.ndarray of float32) empirical cdf evaluations for Y_obs
            based on Y_sample. dim (N,)
    """
    if isinstance(Y_obs, np.ndarray) and isinstance(Y_sample, np.ndarray):
        if Y_obs.ndim == 1:
            Y_obs = np.expand_dims(Y_obs, axis)

        return np.mean(Y_sample < Y_obs, axis)

    elif isinstance(Y_obs, tf.Tensor) and isinstance(Y_sample, tf.Tensor):
        return tf.reduce_mean(tf.cast(Y_sample < Y_obs, tf.float32), axis)
    else:
        raise ValueError("'Y_obs' and 'Y_sample' must both be np.ndarray or tf.Tensor")


def make_empirical_cdf_1d(sample, reduce_mean=True):
    """Creates a 1D empirical cdf.

    Args:
        sample: (np.ndarray or tf.Tensor) Observed samples of dimension (M, )
        reduce_mean: (bool) Whether to reduce empirical cdf evaluation to scalar
            by mean averaging.

    Returns:
        (function) The empirical cdf function based on Y_sample
    """
    if isinstance(sample, np.ndarray):
        def ecdf_func(val):
            if val == 0.:
                return 0.
            else:
                ecdf_val = (sample <= val)
                if reduce_mean:
                    ecdf_val = np.mean(ecdf_val)
                return ecdf_val

        return np.vectorize(ecdf_func)

    elif isinstance(sample, tf.Tensor):
        def ecdf_func(val):
            if val == 0.:
                return 0.
            else:
                ecdf_val = tf.cast(sample <= val, tf.float32)
                if reduce_mean:
                    ecdf_val = tf.reduce_mean(ecdf_val)
                return ecdf_val

        return ecdf_func
    else:
        raise ValueError("'sample' must both be either np.ndarray or tf.Tensor")
