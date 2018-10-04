"""Functions computing mis-calibration metrics in probabilistic ensemble.

#### References

[1]:    Delle Monache L, Hacker JP, Zhou Y, Deng X, Stull RB. Probabilistic aspects of
        meteorological and ozone regional ensemble forecasts. _J. Geophys. Res. 111: D24307_
        2006. https://doi.org/10.1029/2005JD006917.
[2]:    Hamill TM, Colucci SJ. Verification of Eta‐RSM short‐range ensemble forecasts.
        _Mon. Weather Rev. 125: 1312–1327_. 1997.
[3]:    Daniel S. Wilks. Enforcing calibration in ensemble postprocessing.
        _Quarterly Journal of the Royal Meteorological Society, 144(710), 76-84_. 2018
"""
import functools

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import calibre.util.metric as metric_util

__all__ = ["calibration_score", "make_calibration_loss"]


def calibration_score(Y_sample, Y_obs, n_cdf_eval=500):
    """Computes the quality of probabilistic calibration.

    Denote f = F^{-1}(y_obs) the inverse CDF of posterior predictive evaluated at y_obs.
    If y_obs ~ F (i.e. probabilistically calibrated), then f should be
        uniformly distributed, and P(f < x) = x.
    The metric computed here is the Total Variation Distance
        (i.e. absolute difference) between P(F^{-1}(y_obs)<x) v.s. x.

    Args:
        Y_sample: (tf.Tensor of float32) Samples of size M corresponding
        to the N observations. dim (N, M).
        Y_obs: (tf.Tensor of float32) N observations of dim (N, 1)
        n_cdf_eval: (int) Number of values for CDF evaluations between [0, 1].

    Returns:
        (tf.Tensor) MSE between P(F^{-1}(y_obs)<x) and x.
    """
    Y_sample = tf.convert_to_tensor(Y_sample)
    Y_obs = tf.convert_to_tensor(Y_obs)

    if tf.size(Y_obs) in Y_sample.shape.as_list():
        raise ValueError(
            "Number of samples in Y_obs must match at least one "
            "dimension in Y_sample.")
    if tf.size(Y_obs) != Y_sample.shape.as_list()[0]:
        Y_sample = tf.transpose(Y_sample)

    # evaluate each of sampled Ys at ecdf made from Y_obs
    ecdf_sample = metric_util.ecdf_eval(Y_sample, Y_obs, axis=0)
    ecdf_func = metric_util.make_empirical_cdf_1d(ecdf_sample,
                                                  reduce_mean=False)

    ecdf_eval = np.linspace(0., 1., n_cdf_eval, dtype=np.float32)
    ecdf_value = tf.map_fn(ecdf_func, ecdf_eval)

    abs_diff = tf.subtract(np.expand_dims(ecdf_eval,-1),
                           ecdf_value)

    return tf.abs(tf.reduce_mean(abs_diff, 0))


def make_calibration_loss(Y_sample, Y_obs, log_prob,
                          axis=0, keep_dims=False, name=None):
    """Produces Calibration Loss Op using Monte Carlo Expectation.

    Args:
        Y_sample: (tf.Tensor of float32) Samples of size M corresponding
        to the N observations. dim (N, M).
        Y_obs: (tf.Tensor of float32) N observations of dim (N, 1)
        log_prob: (function) Python callable which can return
          `log_prob(samples)`. Must correspond to the natural-logarithm
          of the pdf/pmf of each sample. Only required/used if
          `use_reparametrization=False`.
          Default value: `None`.
        axis: (int or None) The dimensions to average. If `None`, averages all
          dimensions. Default value: `0` (the left-most dimension).
        keep_dims: (bool) If True, retains averaged dimensions using size
         `1`. Default value: `False`.
        name: (str) A `name_scope` for operations created by this function.
          Default value: `None` (which implies "expectation").

    Returns:
        approx_expectation: (tf.Tensor) corresponding to the Monte-Carlo
            approximation of `E_p[f(X)]`.

    """
    f = functools.partial(calibration_score, Y_obs=Y_obs)
    return tfp.monte_carlo.expectation(
        f=f, samples=Y_sample, log_prob=log_prob,
        use_reparametrization=False,
        axis=axis, keep_dims=keep_dims, name=name
    )
