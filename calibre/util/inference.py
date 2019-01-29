"""Utility functions for posterior inference"""
import numpy as np

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


def make_sparse_gp_parameters(m, S,
                              X, Z, ls, kern_func,
                              ridge_factor=1e-3,
                              mean_name='qf_mean', compute_mean=True):
    """Produces variational parameters for sparse GP approximation.

    Args:
        m: (tf.Tensor or None) Variational parameter for mean of latent GP, shape (Nz, )
            Can be None if compute_mean=False
        S: (tf.Tensor) Variational parameter for covariance of latent GP, shape (Nz, Nz)
        X: (np.ndarray of float32) input training features, with dimension (Nx, D).
        Z: (np.ndarray of float32) inducing points, with dimension (Nz, D).
        ls: (float32) length scale parameter.
        kern_func: (function) kernel function.
        ridge_factor: (float32) small ridge factor to stabilize Cholesky decomposition
        mean_name: (str) name for the mean parameter
        compute_mean: (bool) If False, mean variational parameter is not computed.
            In this case, its ok to have m=None

    Returns:
        Mu (tf.Tensor or none) Mean parameters for sparse Gaussian Process, shape (Nx, ).
            if compute_mean=False, then Mu is None.
        Sigma (tf.Tensor) Covariance parameters for sparse Gaussian Process, shape (Nx, Nx).
    """
    Nx, Nz = X.shape[0], Z.shape[0]

    # compute matrix constants
    Kxx = kern_func(X, ls=ls)
    Kxz = kern_func(X, Z, ls=ls)
    Kzz = kern_func(Z, ls=ls, ridge_factor=ridge_factor)

    # compute null covariance matrix using Cholesky decomposition
    Kzz_chol_inv = tf.matrix_inverse(tf.cholesky(Kzz))
    Kzz_inv = tf.matmul(Kzz_chol_inv, Kzz_chol_inv, transpose_a=True)

    Kxz_Kzz_chol_inv = tf.matmul(Kxz, Kzz_chol_inv, transpose_b=True)
    Kxz_Kzz_inv = tf.matmul(Kxz, Kzz_inv)
    Sigma_pre = Kxx - tf.matmul(Kxz_Kzz_chol_inv, Kxz_Kzz_chol_inv, transpose_b=True)

    # compute sparse gp variational parameter (i.e. mean and covariance of P(f_obs | f_latent))
    Sigma = (Sigma_pre +
             tf.matmul(Kxz_Kzz_inv,
                       tf.matmul(S, Kxz_Kzz_inv, transpose_b=True)) +
             ridge_factor * tf.eye(Nx))

    if compute_mean:
        Mu = tf.tensordot(Kxz_Kzz_inv, m, [[1], [0]], name=mean_name)
    else:
        Mu = None

    return Mu, Sigma


def make_cond_gp_parameters(K_00, K_11, K_22,
                            K_01, K_20, K_21,
                            ridge_factor_K=1e-3,
                            ridge_factor_Sigma=1e-3):
    """Computes the conditional posterior for f_new|f_obs, f_deriv.

    For stability, numpy is used instead of tensorflow.

    """
    # convert to np array
    with tf.Session() as sess:
        K_00, K_11, K_22, K_01, K_20, K_21 = sess.run([
            K_00, K_11, K_22, K_01, K_20, K_21
        ])
        K_00 = K_00.astype(np.float64)
        K_11 = K_11.astype(np.float64)
        K_22 = K_22.astype(np.float64)
        K_01 = K_01.astype(np.float64)
        K_20 = K_20.astype(np.float64)
        K_21 = K_21.astype(np.float64)

    # compute matrix components
    K_11_inv_12 = np.matmul(np.linalg.pinv(K_11), K_21.T)
    K_22_inv_21 = np.matmul(np.linalg.pinv(K_22), K_21)

    # assemble projection matrix
    K_02_1 = K_20.T - np.matmul(K_01, K_11_inv_12)
    K_22_1 = (K_22 - np.matmul(K_21, K_11_inv_12) +
              ridge_factor_K * np.eye(K_22.shape[0]))
    K_01_2 = K_01 - np.matmul(K_20.T, K_22_inv_21)
    K_11_2 = K_11 - np.matmul(K_21.T, K_22_inv_21)

    # compute mean projection matrix
    P_01 = np.matmul(K_01_2, np.linalg.pinv(K_11_2))
    P_02 = np.matmul(K_02_1, np.linalg.pinv(K_22_1))

    # compute Cholesky decomposition for covariance matrix.
    Sigma = K_00 - K_01.dot(np.linalg.pinv(K_11).dot(K_01.T))
    # np.matmul(P_01, K_01.T)
    # - np.matmul(P_02, K_20) +
    # ridge_factor_Sigma * np.eye(K_00.shape[0]))

    # Sigma_chol = np.linalg.cholesky(Sigma).astype(np.float32)

    return P_01.astype(np.float32), P_02.astype(np.float32), Sigma


def make_mfvi_mixture_family(n_mixture, N, name):
    """Makes mixture of MFVI variational prior

    Args:
        n_mixture: (int) Number of MFVI mixture.
        N: (int) Number of sample observations.
        name: (str) Name prefix of parameters

    Returns:
        mfvi_mix_dist: (tfd.Distribution) Mixture distribution.
        mixture_logits_mfvi_mix: (tf.Variable or None) Mixture probability
            (logit) for MFVI families. If n_mixture=1 then None.
        qf_mean_mfvi_mix, qf_sdev_mfvi_mix (tf.Variable) Mean and sdev for
            MFVI families. Shape (n_mixture, Nx) if n_mixture > 1, and shape
            (Nx, ) if n_mixture = 1.
    """
    # define mixture probability
    mixture_logits_mfvi_mix = tf.get_variable(shape=[n_mixture],
                                              name='{}_mixture_logits_mfvi_mix'.format(name))

    # define variational parameter
    param_shape = [n_mixture, N] if n_mixture > 1 else [N]
    qf_mean_mfvi_mix = tf.get_variable(shape=param_shape,
                                       name='{}_mean_mfvi_mix'.format(name))
    qf_sdev_mfvi_mix = tf.exp(tf.get_variable(shape=param_shape,
                                              name='{}_sdev_mfvi_mix'.format(name)))

    if n_mixture == 1:
        mfvi_mix_dist = tfd.MultivariateNormalDiag(loc=qf_mean_mfvi_mix,
                                                   scale_diag=qf_sdev_mfvi_mix)

    else:
        mfvi_mix_dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=mixture_logits_mfvi_mix),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=qf_mean_mfvi_mix,
                scale_diag=qf_sdev_mfvi_mix),
        )

    return mfvi_mix_dist, mixture_logits_mfvi_mix, qf_mean_mfvi_mix, qf_sdev_mfvi_mix


def sample_mfvi_mixture_family(N_sample, mixture_logits,
                               mean_mfvi_mix, sdev_mfvi_mix):
    """Samples from mixture of MFVI family.

    Args:
        N_sample: (int) Number of samples.
        mixture_logits: (or None) Number of MFVI mixture.
        mean_mfvi_mix: (np.ndarray) Means for MFVI components,
            shape (n_mixture, N), dtype float32.
        sdev_mfvi_mix: (np.ndarray) Stddev for MFVI components,
            shape (n_mixture, N), dtype float32.

    Returns:
        mfvi_mix_sample: (tf.Tensor) Samples from mixture family,
            shape (N, ), dtype float32.

    """
    # define mixture distribution
    if mixture_logits.shape[0] > 1:
        mfvi_mix_dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=mixture_logits),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=mean_mfvi_mix, scale_diag=sdev_mfvi_mix),
        )
    else:
        mfvi_mix_dist = tfd.MultivariateNormalDiag(loc=mean_mfvi_mix,
                                                   scale_diag=sdev_mfvi_mix)

    return mfvi_mix_dist.sample(N_sample)


def make_mfvi_sgp_mixture_family(n_mixture, N, gp_dist, name,
                                 use_logistic_link=False):
    """Makes mixture of MFVI and Sparse GP variational prior

    Args:
        n_mixture: (int) Number of MFVI mixture.
        N: (int) Number of sample observations.
        gp_dist: (tfd.Distribution) variational family for gaussian process.
        name: (str) Name prefix of parameters

    Returns:
        mfvi_mix_dist: (tfd.Distribution) Mixture distribution.
        mixture_logits_mfvi_mix: (tf.Variable or None) Mixture probability
            (logit) for MFVI families. If n_mixture=1 then None.
        qf_mean_mfvi_mix, qf_sdev_mfvi_mix (tf.Variable) Mean and sdev for
            MFVI families. Shape (n_mixture, Nx) if n_mixture > 1, and shape
            (Nx, ) if n_mixture = 1.
    """
    # define mixture probability
    mixture_logits = tf.get_variable(name="{}_mixture_logits".format(name), shape=[2])

    (mfvi_mix_dist, mixture_logits_mfvi_mix,
     qf_mean_mfvi_mix, qf_sdev_mfvi_mix
     ) = make_mfvi_mixture_family(n_mixture=n_mixture, N=N, name=name)

    mixture_par_list = [mixture_logits, mixture_logits_mfvi_mix,
                        qf_mean_mfvi_mix, qf_sdev_mfvi_mix]

    if use_logistic_link:
        mfvi_sgp_mix_dist = ed.TransformedDistribution(
            tfd.Mixture(
                cat=tfd.Categorical(logits=mixture_logits),
                components=[mfvi_mix_dist, gp_dist]),
            bijector=tfp.bijectors.Sigmoid(),
            name=name)
    else:
        mfvi_sgp_mix_dist = ed.Mixture(
            cat=tfd.Categorical(logits=mixture_logits),
            components=[mfvi_mix_dist, gp_dist],
            name=name)

    return mfvi_sgp_mix_dist, mixture_par_list


def scalar_gaussian_variational(name, mean=None, sdev=None):
    """
    Creates a scalar Gaussian random variable for variational approximation.

    Args:
        name: (str) name of the output random variable.

    Returns:
        (ed.RandomVariable of float32) A normal scalar random variable.
    """
    if mean is None:
        mean = tf.get_variable(shape=[], name='{}_mean'.format(name))
    else:
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)

    if sdev is None:
        sdev = tf.exp(tf.get_variable(shape=[], name='{}_sdev'.format(name)))
    else:
        sdev = tf.convert_to_tensor(sdev, dtype=tf.float32)

    scalar_gaussian_rv = ed.Normal(loc=mean, scale=sdev, name=name)
    return scalar_gaussian_rv, mean, sdev


def sample_scalar_gaussian_variational(n_sample, mean, sdev):
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
