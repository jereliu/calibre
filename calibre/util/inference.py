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
