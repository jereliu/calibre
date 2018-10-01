"""
Define a classic Gaussian Process model with RBF kernel.


#### References

[1]:    Michalis Titsias. Variational Learning of Inducing Variables in
        Sparse Gaussian Processes. _12th Proceedings of AISTATS, PMLR 5:567-574_, 2009.
        http://proceedings.mlr.press/v5/titsias09a.html
[2]:    Ching-An Cheng and Byron Boots. Variational Inference for Gaussian
        Process Models with Linear Complexity. _Advances in NIPS 30_, 2017.
        http://papers.nips.cc/paper/7103-variational-inference-for-gaussian-process-models-with-linear-complexity
[3]:    Marton Havasi, José Miguel Hernández-Lobato and Juan José Murillo-Fuentes.
        Deep Gaussian Processes with Decoupled Inducing Inputs.
        _arXiv preprint arXiv:1801.02939_, 2018.
        https://arxiv.org/pdf/1801.02939.pdf
"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
from tensorflow.python.ops.distributions.util import fill_triangular

from calibre.model.gaussian_process import rbf

tfd = tfp.distributions

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Main model definition """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def model(X, ls=1., ridge_factor=1e-4):
    """Defines the Gaussian Process Model.

    Args:
        X: (np.ndarray of float32) input training features.
        with dimension (N, D).
        ls: (float32) length scale parameter.
        ridge_factor: (float32) ridge factor to stabilize Cholesky decomposition.

    Returns:
         (tf.Tensors of float32) model parameters.
    """
    N = X.shape[0]
    K_mat = rbf(X, ls=ls, ridge_factor=ridge_factor)

    gp_f = ed.MultivariateNormalTriL(loc=tf.zeros(N),
                                     scale_tril=tf.cholesky(K_mat),
                                     name="gp_f")
    sigma = ed.Normal(loc=-5., scale=1., name='sigma')
    y = ed.MultivariateNormalDiag(loc=gp_f,
                                  scale_identity_multiplier=tf.exp(sigma),
                                  name="y")

    return gp_f, sigma, y


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Variational family I: Mean field """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def variational_mfvi(X):
    """Defines the mean-field variational family for GPR.

    Args:
        X: (np.ndarray of float32) input training features.
        with dimension (N, D).

    Returns:
        q_f, q_sig: (ed.RandomVariable) variational family.
        q_f_mean, q_f_sdev: (tf.Variable) variational parameters for q_f
    """
    N, D = X.shape

    # define variational parameters
    qf_mean = tf.get_variable(shape=[N], name='qf_mean')
    qf_sdev = tf.exp(tf.get_variable(shape=[N], name='qf_sdev'))
    q_sig_mean = tf.get_variable(shape=[], name='q_sig_mean')
    q_sig_sdev = tf.exp(tf.get_variable(shape=[], name='q_sig_sdev'))

    # define variational family
    q_f = ed.MultivariateNormalDiag(loc=qf_mean, scale_diag=qf_sdev,
                                    name='q_f')
    q_sig = ed.Normal(loc=q_sig_mean, scale=q_sig_sdev, name='q_sig')

    return q_f, q_sig, qf_mean, qf_sdev


def variational_mfvi_sample(n_sample, qf_mean, qf_sdev):
    """Generates f samples from GPR mean-field variational family.

    Args:
        n_sample: (int) number of samples to draw
        qf_mean: (tf.Tensor of float32) mean parameters for
        variational family
        qf_sdev: (tf.Tensor of float32) standard deviation
        parameters for variational family

    Returns:
         (np.ndarray) sampled values.
    """

    """Generates f samples from GPR mean-field variational family."""
    q_f = tfd.MultivariateNormalDiag(loc=qf_mean, scale_diag=qf_sdev,
                                     name='q_f')
    return q_f.sample(n_sample)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Variational family II: Sparse Gaussian Process """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""Implements the Sparse GP VI method by [1].

Select a set of inducing points Z, then:

Original posterior:
p(Y, F, U) = p(Y|F) p(F|U) p(U), where:
    p(Y|F) ~ MVN(Y| Mu =    F, 
                    Sigma = s^2 * I)
    p(F|U) ~ MVN(F| Mu =    Kxz Kzz^{-1} U 
                    Sigma = Kxx - Kxz Kzz^{-1} Kxz^T )
    p(U)   ~ MVN(U| Mu = 0, Sigma = Kzz)

Variational posterior:
    q(Y)   = p(Y|F)
    q(F|U) = p(F|U)
    q(U|m, S) ~ MVN(U| Mu = m, Sigma = S)

Consequently, U can be marginalized out, such that q(F) becomes 
    q(F|m, S) ~ MVN(F| Mu =     Kxz Kzz^{-1} m
                       Sigma =  Kxx - Kxz Kzz^{-1} (Kzz - S) Kzz^{-1} Kxz^T)
"""


def variational_sgpr(X, Z, ls=1., kern_func=rbf, ridge_factor=1e-3):
    """Defines the mean-field variational family for GPR.

    Args:
        X: (np.ndarray of float32) input training features, with dimension (Nx, D).
        Z: (np.ndarray of float32) inducing points, with dimension (Nz, D).
        ls: (float32) length scale parameter.
        kern_func: (function) kernel function.
        ridge_factor: (float32) small ridge factor to stabilize Cholesky decomposition

    Returns:
        q_f, q_sig: (ed.RandomVariable) variational family.
        q_f_mean, q_f_sdev: (tf.Variable) variational parameters for q_f
    """
    Nx, Nz = X.shape[0], Z.shape[0]

    # 1. Prepare constants
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

    # 2. Define variational parameters
    # define mean and variance for sigma
    q_sig_mean = tf.get_variable(shape=[], name='q_sig_mean')
    q_sig_sdev = tf.exp(tf.get_variable(shape=[], name='q_sig_sdev'))

    # define free parameters (i.e. mean and full covariance of f_latent)
    m = tf.get_variable(shape=[Nz], name='qf_m')
    s = tf.get_variable(shape=[Nz * (Nz + 1) / 2],
                        # initializer=tf.zeros_initializer(),
                        name='qf_s')
    L = fill_triangular(s, name='qf_chol')
    S = tf.matmul(L, L, transpose_b=True)

    # compute sparse gp variational parameter (i.e. mean and covariance of P(f_obs | f_latent))
    qf_mean = tf.tensordot(Kxz_Kzz_inv, m, [[1], [0]], name='qf_mean')
    qf_cov = (Sigma_pre +
              tf.matmul(Kxz_Kzz_inv,
                        tf.matmul(S, Kxz_Kzz_inv, transpose_b=True)) +
              ridge_factor * tf.eye(Nx)
              )

    # define variational family
    q_f = ed.MultivariateNormalFullCovariance(loc=qf_mean,
                                              covariance_matrix=qf_cov,
                                              name='q_f')
    q_sig = ed.Normal(loc=q_sig_mean,
                      scale=q_sig_sdev, name='q_sig')

    return (q_f, q_sig, qf_mean, qf_cov,
            Sigma_pre, S, Kxx, Kxz, Kzz, Kzz_inv, Kxz_Kzz_inv)


def variational_sgpr_sample(n_sample, qf_mean, qf_cov):
    """Generates f samples from GPR mean-field variational family.

    Args:
        n_sample: (int) number of samples to draw
        qf_mean: (tf.Tensor of float32) mean parameters for
        variational family
        qf_cov: (tf.Tensor of float32) covariance for
        parameters for variational family

    Returns:
        (np.ndarray) sampled values.
    """

    """Generates f samples from GPR mean-field variational family."""
    q_f = tfd.MultivariateNormalFullCovariance(loc=qf_mean,
                                               covariance_matrix=qf_cov,
                                               name='q_f')
    return q_f.sample(n_sample)