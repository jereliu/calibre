"""Model and Sampling functions for Parametric Ensemble with Residual Process.

    y           ~   sum{ f_k(x) * w_k } + delta(x) + epsilon
    w_k         ~   LogisticNormal  ( 0, sigma_k )
    delta(x)    ~   GaussianProcess ( 0, k(x) )
    epsilon     ~   Normal          ( 0, sigma_e )

"""

import functools

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed

import calibre.util.inference as inference_util

from calibre.model import gaussian_process as gp
from calibre.model import tailfree_process as tail_free

from calibre.util.model import sparse_softmax

tfd = tfp.distributions

_WEIGHT_PRIOR_MEAN = np.array(0.).astype(np.float32)
_WEIGHT_PRIOR_SDEV = np.array(1.).astype(np.float32)

_TEMP_PRIOR_MEAN = np.array(-4.).astype(np.float32)
_TEMP_PRIOR_SDEV = np.array(1.).astype(np.float32)

_LS_PRIOR_MEAN = np.array(-5.).astype(np.float32)
_LS_PRIOR_SDEV = np.array(1.).astype(np.float32)

_NOISE_PRIOR_MEAN = np.array(-5.).astype(np.float32)
_NOISE_PRIOR_SDEV = np.array(1.).astype(np.float32)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Main model definition """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def sparse_logistic_weight(base_pred, temp,
                           link_func=sparse_softmax,
                           name="ensemble_weight"):
    r"""Defines the Logistic Normal prior for p(model).

    Defines the conditional distribution of model given feature as:

        w( model | x ) = link_func( w_raw(x) )
        w_raw( x ) ~ Normal( 0, sigma_k )

    Args:
        X: (np.ndarray) Input features of dimension (N, D)
        base_pred: (dict of np.ndarray) A dictionary of out-of-sample prediction
            from base models. For detail, see calibre.
        temp: (ed.RandomVariable of float32) list of unnormalized
            temperature parameter for sparse ensemble (to pass to link_func).
            It's dimension must be (len(family_tree), ) (i.e. one temp parameter for each node
            in the model family tree).
        link_func: (function) a link function that transforms the unnormalized
            base ensemble weights to a K-dimension simplex.
            This function has args (logits, temp)
        name: (str) name of the ensemble weight node on the computation graph.

    Returns:
        (tf.Tensor of float32) normalized ensemble weights, dimension (M, 1).
    """
    # specify un-normalized weights
    base_names = list(base_pred.keys())

    W_raw = tf.stack([
        ed.Normal(loc=_WEIGHT_PRIOR_MEAN,
                  scale=_WEIGHT_PRIOR_SDEV,
                  name='base_weight_{}'.format(base_names))
        for base_names in base_names])

    # specify normalized weights
    W_model = link_func(W_raw, tf.exp(temp))
    W_model = tf.expand_dims(W_model, -1, name=name)

    return W_model


def model(X, base_pred,
          add_resid=True, log_ls_resid=None):
    r"""Defines the sparse adaptive ensemble model.

    y           ~   sum{ f_k(x) * w_k } + delta(x) + epsilon
    w_k         ~   LogisticNormal  ( 0, sigma_k )
    delta(x)    ~   GaussianProcess ( 0, k(x) )
    epsilon     ~   Normal          ( 0, sigma_e )

    where the LogisticNormal is sparse_softmax transformed Normals.

    Args:
        X: (np.ndarray) Input features of dimension (N, D)
        base_pred: (dict of np.ndarray) A dictionary of out-of-sample prediction
            from base models. For each item in the dictionary,
            key is the model name, and value is the model prediction with
            dimension (N, ).
        add_resid: (bool) Whether to add residual process to model.
        log_ls_resid: (float32) length-scale parameter for residual GP.
            If None then will estimate with normal prior.

    Returns:
        (tf.Tensors of float32) model parameters.
    """
    # convert data type
    F = np.asarray(list(base_pred.values())).T
    F = tf.convert_to_tensor(F, dtype=tf.float32)
    X = tf.convert_to_tensor(X, dtype=tf.float32)

    # check dimension
    N, D = X.shape
    for key, value in base_pred.items():
        if not value.shape == (N,):
            raise ValueError(
                "All base-model predictions should have shape ({},), but"
                "observed {} for '{}'".format(N, value.shape, key))

    # specify prior for lengthscale and observation noise
    if log_ls_resid is None:
        log_ls_resid = ed.Normal(loc=_LS_PRIOR_MEAN,
                                 scale=_LS_PRIOR_SDEV, name="ls_resid")

    # specify logistic normal priors for ensemble weight
    temp = ed.Normal(loc=_TEMP_PRIOR_MEAN,
                     scale=_TEMP_PRIOR_SDEV, name='temp')
    W = sparse_logistic_weight(base_pred, temp,
                               name="ensemble_weight")

    # specify ensemble prediction
    FW = tf.matmul(F, W)
    ensemble_mean = tf.reduce_sum(FW, axis=1, name="ensemble_mean")

    # specify residual process
    if add_resid:
        ensemble_resid = gp.prior(X,
                                  ls=tf.exp(log_ls_resid),
                                  kernel_func=gp.rbf,
                                  name="ensemble_resid")
    else:
        ensemble_resid = 0.

    # specify observation noise
    sigma = ed.Normal(loc=_NOISE_PRIOR_MEAN,
                      scale=_NOISE_PRIOR_SDEV, name="sigma")

    # specify observation
    y = ed.MultivariateNormalDiag(loc=ensemble_mean + ensemble_resid,
                                  scale_identity_multiplier=tf.exp(sigma),
                                  name="y")
    return y


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Sampling functions for intermediate random variables """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def sample_posterior_weight(weight_sample, temp_sample,
                            link_func=sparse_softmax):
    """Computes posterior sample for f_ensemble functions.

    Args:
        weight_sample: (np.ndarray or list) list of un-normalized GP weights
            for sparse ensemble, each with dimension (K, N_sample, N_obs)
        temp_sample: (np.ndarray) list of un-normalized temperature parameter
            for sparse ensemble.
        link_func: (function) a link function that transforms the unnormalized
            base ensemble weights to a K-dimension simplex.
            This function has args (logits, temp)

    Returns:
        (tf.Tensor of float32) Posterior samples of f_ensemble of dimension
            (N_sample, N_obs, )
    Raises:
        ValueError: If first dimension of weight_sample does not equal to
            that of the temp_sample
    """
    N_sample, = temp_sample.shape

    if isinstance(weight_sample, list):
        weight_sample = np.asanyarray(weight_sample)
        # shape is now (N_sample, N_obs, K)
        weight_sample = np.moveaxis(weight_sample, 0, -1)
        if not weight_sample.shape[0] == N_sample:
            raise ValueError(
                "Sample size of weight_sample (dim={}) doesn't match "
                "that of the temp_sample ({})".format(
                    weight_sample.shape, N_sample))

    # compute ensemble weights
    ensemble_weight_sample = link_func(weight_sample, tf.exp(temp_sample),
                                       name="ensemble_weight")

    return ensemble_weight_sample


def sample_posterior_mean(base_pred, weight_sample, temp_sample,
                          link_func=sparse_softmax):
    """Computes posterior sample for f_ensemble functions.

    Args:
        base_pred: (dict of np.ndarray) A dictionary of out-of-sample prediction
            from base models. For each item in the dictionary,
            key is the model name, and value is the model prediction with
            dimension (N_obs, ).
        weight_sample: (np.ndarray or list) list of un-normalized GP weights
            for sparse ensemble, each with dimension (K, N_sample, N_obs)
        temp_sample: (np.ndarray) list of un-normalized temperature parameter
            for sparse ensemble.
        link_func: (function) a link function that transforms the unnormalized
            base ensemble weights to a K-dimension simplex.
            This function has args (logits, temp)

    Returns:
        (tf.Tensor of float32) Posterior samples of f_ensemble of dimension
            (N_obs, N_sample, )
    """
    # compute ensemble weights
    W_sample = sample_posterior_weight(
        weight_sample, temp_sample, link_func=link_func)

    # compute ensemble function
    F = np.asarray(list(base_pred.values())).T  # (N_obs, K)
    FW_sample = tf.matmul(F, W_sample, transpose_b=True)
    f_ens_sample = tf.transpose(FW_sample, name="f_ensemble")

    return f_ens_sample


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Variational Family """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def _parametric_weight_variational_family(base_pred):
    """Defines the variational family for parametric ensemble weights.

    Args:
        X: (np.ndarray) Input features of dimension (N, D)
        base_pred: (dict of np.ndarray) A dictionary of out-of-sample prediction
            from base models. For detail, see calibre.adaptive_ensemble.model.

    Returns:
        weight_dict: (dict of ed.RandomVariable) Dictionary of GP random variables
            for each non-root model/model family.
        temp: (ed.RandomVariable) Ttemperature random variable.
        weight_mean_dict: (dict of tf.Variable) Dictionary of variational parameters for
            the mean of GP.
        weight_vcov_dict: (dict of tf.Variable) Dictionary of variational parameters for
            the stddev or covariance matrix of GP.
        temp_mean: (tf.Variable) Variational parameters for
            the mean of temperatures.
        temp_sdev: (tf.Variable) Variational parameters for
            the stddev or temperatures.
    """
    # temperature
    temp, temp_mean, temp_sdev = (
        inference_util.scalar_gaussian_variational(name='temp'))

    # base weights
    model_names = list(base_pred.keys())
    base_weight_names = ['base_weight_{}'.format(name) for name in model_names]

    weight_list = []
    weight_mean_list = []
    weight_vcov_list = []

    for model_name in model_names:
        weight, weight_mean, weight_sdev = (
            inference_util.scalar_gaussian_variational(
                name='base_weight_{}'.format(model_name)))

        weight_list.append(weight)
        weight_mean_list.append(weight_mean)
        weight_vcov_list.append(weight_sdev)

    weight_dict = dict(zip(base_weight_names, weight_list))
    weight_mean_dict = dict(zip(base_weight_names, weight_mean_list))
    weight_vcov_dict = dict(zip(base_weight_names, weight_vcov_list))

    return (weight_dict, temp,
            weight_mean_dict, weight_vcov_dict,
            temp_mean, temp_sdev)


def _parametric_weight_variational_family_sample(
        n_sample, weight_mean_dict, weight_vcov_dict, temp_mean, temp_sdev):
    """Defines the variational family for parametric ensemble weights.

    Args:
        n_sample: (int) Number of samples to draw from variational family.
        weight_mean_dict: (dict of tf.Variable) Dictionary of variational
            parameters for the mean of GP.
        weight_vcov_dict: (dict of tf.Variable) Dictionary of variational
            parameters for the stddev or covariance matrix of GP.
        temp_mean: (tf.Variable) Variational parameters for
            the mean of temperatures.
        temp_sdev: (tf.Variable) Variational parameters for
            the stddev or temperatures.

    Returns:
        weight_dict: (dict of ed.RandomVariable) Dictionary of GP random variables
            for each non-root model/model family.
        weight_mean_dict: (dict of tf.Variable) Dictionary of variational parameters for
            the mean of GP.
        weight_vcov_dict: (dict of tf.Variable) Dictionary of variational parameters for
            the stddev or covariance matrix of GP.
    """
    # sample weight
    weight_sample_dict = dict()
    for model_names in weight_mean_dict.keys():
        weight_sample_dict[model_names] = (
            inference_util.sample_scalar_gaussian_variational(
                n_sample,
                mean=weight_mean_dict[model_names],
                sdev=weight_vcov_dict[model_names],
            ))

    # sample temperature
    temp_sample = (
        inference_util.sample_scalar_gaussian_variational(
            n_sample, mean=temp_mean, sdev=temp_sdev,
        ))

    return weight_sample_dict, temp_sample


def variational_family(X, base_pred,
                       log_ls_resid=None,
                       gp_vi_family=gp.variational_mfvi,
                       **kwargs):
    """Defines the variational family for sparse adaptive ensemble model.

    Args:
        X: (np.ndarray) Input features of dimension (N, D)
        base_pred: (dict of np.ndarray) A dictionary of out-of-sample prediction
            from base models. For each item in the dictionary,
            key is the model name, and value is the model prediction with
            dimension (N, ).
        log_ls_resid: (float32) Value of lengthscale parameter for residual GP,
            if None then perform variational approximation.
        gp_vi_family: (function) A variational family for node weight
            GPs in the family tree.
        **kwargs: Additional parameters to pass to tail_free/gp.variational_family.

    Returns:
        Collections of variational random variables/parameters:

        > Random variables

        temp_dict: (dict of ed.RandomVariable) Dictionary of temperature RVs
            for each parent model.
        weight_f_dict: (dict of ed.RandomVariable) Dictionary of GP random variables
            for each non-root model/model family.
        resid_f: (ed.RandomVariable) GP random variable for residual process.
        sigma: (ed.RandomVariable) normal RV for log standard derivation of
            observation noise.

        > GP variational parameters:

        weight_gp_mean_dict: (dict of tf.Variable) Dictionary of variational parameters for
            the mean of node weight GP.
        weight_gp_vcov_dict: (dict of tf.Variable) Dictionary of variational parameters for
            the stddev or covariance matrix of node weight GP.
        resid_gp_mean_dict: (dict of tf.Variable) Dictionary of variational parameters for
            the mean of residual GP.
        resid_gp_vcov_dict: (dict of tf.Variable) Dictionary of variational parameters for
            the stddev or covariance matrix of residual GP.
        temp_mean_dict: (dict of tf.Variable) Dictionary of variational parameters for
            the mean of temperature parameters.
        temp_sdev_dict: (dict of tf.Variable) Dictionary of variational parameters for
            the stddev of temperature parameters.
        sigma_mean: (tf.Variable) Variational parameters for the mean of obs noise.
        sigma_sdev: (tf.Variable) Variational parameters for the stddev of obs noise.
    """
    # length-scale parameters
    if log_ls_resid is None:
        log_ls_resid, log_ls_resid_mean, log_ls_resid_sdev = (
            inference_util.scalar_gaussian_variational(name='ls_resid'))
    else:
        log_ls_resid = tf.convert_to_tensor(log_ls_resid, dtype=tf.float32)
        log_ls_resid_mean = None
        log_ls_resid_sdev = None

    # temperature and base weight
    (weight_dict, temp,
     weight_mean_dict, weight_vcov_dict,
     temp_mean, temp_sdev) = (
        _parametric_weight_variational_family(base_pred)
    )

    # residual gp
    (resid_gp, resid_gp_mean, resid_gp_vcov,
     mixture_par_resid,) = gp_vi_family(X, ls=tf.exp(log_ls_resid),
                                        name='vi_ensemble_resid',
                                        **kwargs)

    # observation noise
    sigma, sigma_mean, sigma_sdev = (
        inference_util.scalar_gaussian_variational(name='sigma'))

    return (weight_dict, resid_gp, temp, sigma, log_ls_resid,  # variational RVs
            # variational parameters
            weight_mean_dict, weight_vcov_dict,  # weight GP
            resid_gp_mean, resid_gp_vcov, mixture_par_resid,  # resid GP
            temp_mean, temp_sdev,  # temperature
            sigma_mean, sigma_sdev,  # obs noise
            log_ls_resid_mean, log_ls_resid_sdev  # residual length-scale
            )


def variational_family_sample(n_sample,
                              weight_mean_dict, weight_vcov_dict,
                              temp_mean, temp_sdev,
                              resid_gp_mean, resid_gp_vcov, mixture_par_resid,
                              sigma_mean, sigma_sdev,
                              log_ls_resid_mean, log_ls_resid_sdev,
                              mfvi_mixture=False,
                              gp_sample_func=gp.variational_mfvi_sample):
    """Samples from the variational family for adaptive ensemble.

    Args:
        n_sample: (int) Number of samples to draw from variational family.
        weight_mean_dict: (dict of np.ndarray) Dictionary of variational parameters for
            the mean of node weight GP.
        weight_vcov_dict: (dict of np.ndarray) Dictionary of variational parameters for
            the stddev or covariance matrix of node weight GP.
        temp_mean: (np.ndarray) Variational parameters for
            the mean of temperature parameters.
        temp_sdev: (np.ndarray) Variational parameters for
            the stddev of temperature parameters.
        resid_gp_mean: (np.ndarray of float32) Dictionary of variational parameters for
            the mean of residual GP.
        resid_gp_vcov: (np.ndarray of float32) Dictionary of variational parameters for
            the stddev or covariance matrix of residual GP.
        mixture_par_resid: (list of np.ndarray) List of mixture parameters for
            SGP-MFVI mixture [mixture_logits, qf_mean_mfvi, qf_sdev_mfvi].
        sigma_mean: (float32) Variational parameters for the mean of obs noise.
        sigma_sdev: (float32) Variational parameters for the stddev of obs noise.
        log_ls_resid_mean: (float32) Variational parameters for the mean of ls_resid.
        log_ls_resid_sdev: (float32) Variational parameters for the stddev of ls_resid.
        mfvi_mixture: (bool) Whether the family is a GP-MF mixture.
        gp_sample_func: (function): Sampling function for Gaussian Process variational family.

    Returns:
        weight_sample_dict: (dict of tf.Tensor) Dictionary of temperature random variables
            for each parent model.
        temp_sample_dict: (dict of tf.Tensor) Dictionary of GP samples of raw weights
            for each non-root model/model family.
        resid_gp_sample: (tf.Tensor) GP samples for residual process.
        sigma_sample: (tf.Tensor) Samples of observation noise.
    """
    # sample model weight gp and temperature.
    weight_sample_dict, temp_sample_dict = (
        _parametric_weight_variational_family_sample(n_sample,
                                                     weight_mean_dict, weight_vcov_dict,
                                                     temp_mean, temp_sdev))

    # sample residual process gp
    resid_gp_sample = gp_sample_func(n_sample, resid_gp_mean, resid_gp_vcov,
                                     mfvi_mixture=mfvi_mixture,
                                     mixture_par_list=mixture_par_resid)

    # sample observational noise
    sigma_sample = inference_util.sample_scalar_gaussian_variational(
        n_sample, sigma_mean, sigma_sdev)
    ls_resid_sample = inference_util.sample_scalar_gaussian_variational(
        n_sample, log_ls_resid_mean, log_ls_resid_sdev)

    return (weight_sample_dict, temp_sample_dict, resid_gp_sample,
            sigma_sample, ls_resid_sample)


variational_mfvi = functools.partial(variational_family,
                                     gp_vi_family=gp.variational_mfvi)
variational_sgpr = functools.partial(variational_family,
                                     gp_vi_family=gp.variational_sgpr)
variational_dgpr = functools.partial(variational_family,
                                     gp_vi_family=gp.variational_dgpr)

variational_mfvi_sample = functools.partial(variational_family_sample,
                                            gp_sample_func=gp.variational_mfvi_sample)
variational_sgpr_sample = functools.partial(variational_family_sample,
                                            gp_sample_func=gp.variational_sgpr_sample)
variational_dgpr_sample = functools.partial(variational_family_sample,
                                            gp_sample_func=gp.variational_dgpr_sample)
