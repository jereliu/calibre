"""Utility functions for calibration evaluation."""
import numpy as np


def ecdf_eval(Y_obs, Y_sample):
    """Computes empirical cdf evaluation (i.e. P(Y<y) ) for M samples.

    Args:
        Y_obs: (np.ndarray of float32) N observations of dim (N, 1)
        Y_sample: (np.ndarray of float32) Samples of size M corresponding
        to the N observations. dim (N, M)
    Returns:
        (np.ndarray of float32) empirical cdf evaluations for Y_obs
            based on Y_sample. dim (N,)
    """
    if Y_obs.ndim == 1:
        Y_obs = np.expand_dims(Y_obs, 1)

    return np.mean(Y_sample < Y_obs, 1)


def make_empirical_cdf_1d(sample):
    """Creates a 1D empirical cdf.

    Args:
        Y_sample: (np.ndarray) Observed samples of dimension (M, )

    Returns:
        (function) The empirical cdf function based on Y_sample
    """
    def ecdf_func(val):
        if val == 0.:
            return 0.
        else:
            return np.mean(sample <= val)

    return np.vectorize(ecdf_func)