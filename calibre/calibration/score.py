"""Functions computing mis-calibration metrics in probabilistic ensemble.

#### References

[1]:    Delle Monache L, Hacker JP, Zhou Y, Deng X, Stull RB. Probabilistic aspects of
        meteorological and ozone regional ensemble forecasts. _J. Geophys. Res. 111: D24307_
        2006. https://doi.org/10.1029/2005JD006917.
[2]:    Hamill TM, Colucci SJ. Verification of Eta‐RSM short‐range ensemble forecasts.
        _Mon. Weather Rev. 125: 1312–1327_. 1997.
[3]:    Daniel S. Wilks. Enforcing calibration in ensemble postprocessing.
        _Quarterly Journal of the Royal Meteorological Society, 144(710), 76-84_. 2018
[4]:    Gneiting, T., Raftery, A.E.: Strictly proper scoring rules, prediction, and estimation.
        _J. Am. Stat. Assoc. 102, 359–378_. 2007.
[5]:    Gneiting, T., Balabdaoui, F., and Raftery, A. E. Probabilistic Forecasts,
        Calibration and Sharpness, _Journal of the Royal Statistical Society,Ser.B_. 2007.
"""
import functools

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import calibre.util.metric as metric_util

__all__ = ["calibration_score",
           "make_calibration_loss",
           "make_kernel_score_loss"]


def empirical_cdf(Y_sample, cdf_eval, normalize=True):
    """Computes the value of empirical cdf for a given scalar value.

    Args:
        Y_sample: (tf.Tensor of float32) Samples of size M corresponding
        to the N observations. dim (n_sample, n_obs).
        cdf_eval: (tf.Tensor of float32) values to be evaluated at the empirical cdf.
            dimension (n_eval, ).
        normalize: (bool) Whether to average over Y_sample to produce a (n_obs, n_eval)
            vector of ecdf evaluations.
            Otherwise output a (n_sample, n_obs, n_eval) vector of unnormalized evaluations.

    Returns:
        (tf.Tensor) Empirical CDF evaluations at cdf_eval.
            If 'normalize'=True, a (n_sample, n_obs, n_eval) vector of unnormalized ecdf evals,
            If 'normalize'=False,  a (n_obs, n_eval) vector of normalized ecdf evals.
    """
    Y_sample = tf.squeeze(tf.convert_to_tensor(Y_sample, dtype=tf.float32))
    cdf_eval = tf.squeeze(tf.convert_to_tensor(cdf_eval, dtype=tf.float32))

    # check dimension
    if len(Y_sample.get_shape().as_list()) != 2:
        raise ValueError(
            "Dimension of Y_sample must be (n_obs, n_sample), observed ({})".format(
                Y_sample.get_shape().as_list()))
    if len(cdf_eval.get_shape().as_list()) != 1:
        raise ValueError(
            "Dimension of cdf_eval must be (n_eval, ), observed ({})".format(
                cdf_eval.get_shape().as_list()))

    # reshape Y_sample into (n_sample, n_obs, 1) and cdf_eval into (1, 1, n_eval)
    Y_sample = tf.expand_dims(Y_sample, -1)
    cdf_eval = tf.reshape(cdf_eval, shape=[1, 1, tf.size(cdf_eval)])

    # compute ecdf eval of dim (n_sample, n_obs, n_eval) using broadcasting
    ecdf_val = tf.cast(Y_sample < cdf_eval, tf.float32)

    if normalize:
        ecdf_val = tf.reduce_mean(ecdf_val, axis=0)

    return ecdf_val


def energy_distance(X_sample, Y_sample,
                    dist_func=tf.abs, normalize_over_observation=True):
    """Computes the energy distance E(dist(X - Y)) between X_sample and Y_sample.

    Args:
        X_sample: (tf.Tensor of float32) Samples of size (n_sample_1, n_obs).
        Y_sample: (tf.Tensor of float32) Samples of size (n_sample_2, n_obs).
        dist_func: (function) A positive definite function
        normalize_over_observation: (bool) Whether to average over
            Y_sample to produce a (n_sample_1, n_sample_2) vector of ecdf evaluations.
            Otherwise output a (n_sample_1, n_sample_2, n_obs) vector of unnormalized evaluations.

    Returns:
        (tf.Tensor) Empirical CDF evaluations at cdf_eval.
            If 'normalize'=True, a (n_sample_1, n_sample_2) vector of unnormalized ecdf evals,
            If 'normalize'=False,  a (n_sample_1, n_sample_2, n_obs) vector of normalized ecdf evals.
    """
    X_sample = tf.convert_to_tensor(X_sample, dtype=tf.float32)
    Y_sample = tf.convert_to_tensor(Y_sample, dtype=tf.float32)

    # check dimension
    if (len(X_sample.get_shape()) != 2) or (len(Y_sample.get_shape()) != 2):
        raise ValueError(
            "Dimension of X_sample/Y_sample must be 2 dim (n_sample, n_obs)")

    n_sample_1, n_obs_1 = X_sample.get_shape().as_list()
    n_sample_2, n_obs_2 = Y_sample.get_shape().as_list()

    if n_obs_1 != n_obs_2:
        raise ValueError("Expected last dimension of X_sample and Y_sample "
                         "to be equal and correspond to n_obs. "
                         "However observed {} and {}".format(n_obs_1, n_obs_2))

    # reshape X_sample and Y_sample for broadcasting
    X_sample = tf.expand_dims(X_sample, axis=1)  # (n_sample_1, 1, n_obs)
    Y_sample = tf.expand_dims(Y_sample, axis=0)  # (1, n_sample_2, n_obs)

    # compute ecdf eval of dim (n_sample, n_obs, n_eval) using broadcasting
    energy_val = dist_func(X_sample - Y_sample)

    if normalize_over_observation:
        energy_val = tf.reduce_mean(energy_val, axis=-1)

    return energy_val


def calibration_score(Y_sample, Y_obs, n_cdf_eval=500):
    """Computes the quality of probabilistic calibration.

    Denote f = F^{-1}(y_obs) the inverse CDF of posterior predictive evaluated at y_obs.
    If y_obs ~ F (i.e. probabilistically calibrated), then f should be
        uniformly distributed, and P(f < x) = x.
    The calibration computed here is the Total Variation Distance
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

    abs_diff = tf.subtract(np.expand_dims(ecdf_eval, -1),
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
          of the pdf/pmf of each sample.
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


def make_kernel_score_loss(X_sample, Y_sample, Y_obs, log_prob,
                           dist_func=tf.abs,
                           keep_dims=False, name=None):
    """Computes monte carlo average of energy score.

    Energy score is defined as E(g(Y-Y_obs)) - 0.5*E(g(X, Y)) as in [4].

    Args:
        X_sample: (tf.Tensor of float32) Samples of size n_sample_1 corresponding
        to the N observations. dim (n_sample_1, n_obs).
        Y_sample: (tf.Tensor of float32) Samples of size n_sample_2 corresponding
        to the N observations. dim (n_sample_2, n_obs).
        Y_obs: (tf.Tensor of float32) Observations. dim (n_obs).
        dist_func: (function) distance function for computing g(X, Y)
        log_prob: (function) Python callable which can return
          `log_prob(samples)`. Must correspond to the natural-logarithm
          of the pdf/pmf of each sample.
        keep_dims: (bool) If True, retains averaged dimensions using size
         `1`. Default value: `False`.
        name: (str) A `name_scope` for operations created by this function.
          Default value: `None` (which implies "expectation").

    Returns:
        (tf.Tensor) energy score averaged over observations.

    """
    # reshape Y_obs
    Y_obs = tf.convert_to_tensor(Y_obs)
    Y_obs = tf.reshape(Y_obs, [1, tf.size(Y_obs)])

    # prepare evaluation function for energy distance
    f_obsv = functools.partial(energy_distance,
                               Y_sample=Y_obs, dist_func=dist_func,
                               normalize_over_observation=True)
    f_pair = functools.partial(energy_distance, dist_func=dist_func,
                               normalize_over_observation=True)

    # produce monte carlo estimators averaged over observations
    obsv_distance = tfp.monte_carlo.expectation(
        f=f_obsv, samples=X_sample, log_prob=log_prob,
        use_reparametrization=False,
        axis=None, keep_dims=keep_dims, name="{}_obs".format(name))

    pair_distance = metric_util.monte_carlo_dual_expectation(
        f=f_pair, samples_1=X_sample, samples_2=Y_sample, log_prob=log_prob,
        axis=None, name="{}_pair".format(name))

    return obsv_distance - 0.5 * pair_distance
