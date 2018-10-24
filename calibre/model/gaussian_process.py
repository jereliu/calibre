"""Model definition and variational family for Gaussian Process Prior.

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
[4]:    Carl Rasmussen and Christopher Williams. Gaussian Processes for Machine Learning.
        _The MIT Press. ISBN 0-262-18253-X_. 2006
"""
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed

from tensorflow.python.ops.distributions.util import fill_triangular

tfd = tfp.distributions

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Kernel function """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def square_dist(X, X2=None, ls=1.):
    """Computes Square distance between two sets of features.

    Referenced from GPflow.kernels.Stationary.

    Args:
        X: (tf.Tensor) First set of features of dim N x D.
        X2: (tf.Tensor or None) Second set of features of dim N2 x D.
        ls: (float) value for length scale.

    Returns:
        (tf.Tensor) A N x N2 tensor for ||x-x'||^2 / ls**2

    Raises:
        (ValueError) If feature dimension of X and X2 disagrees.
    """
    N, D = X.shape

    X = X / ls
    Xs = tf.reduce_sum(tf.square(X), axis=1)

    if X2 is None:
        dist = -2 * tf.matmul(X, X, transpose_b=True)
        dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
        return tf.clip_by_value(dist, 0., np.inf)

    N2, D2 = X2.shape
    if D != D2:
        raise ValueError('Dimension of X and X2 does not match.')

    X2 = X2 / ls
    X2s = tf.reduce_sum(tf.square(X2), axis=1)
    dist = -2 * tf.matmul(X, X2, transpose_b=True)
    dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))
    return tf.clip_by_value(dist, 0., np.inf)


def rbf(X, X2=None, ls=1., ridge_factor=0.):
    """Defines RBF kernel.

     k(x, x') = - exp(- |x-x'| / ls**2)

    Args:
        X: (tf.Tensor) First set of features of dim N x D.
        X2: (tf.Tensor or None) Second set of features of dim N2 x D.
        ls: (float) value for length scale
        ridge_factor: (float32) ridge factor to stabilize Cholesky decomposition.

    Returns:
        (tf.Tensor) A N x N2 tensor for exp(-||x-x'||**2 / 2 * ls**2)
    """
    N, _ = X.shape.as_list()
    if ridge_factor and X2 is None:
        ridge_mat = ridge_factor * tf.eye(N, dtype=tf.float32)
    else:
        ridge_mat = 0

    return tf.exp(-square_dist(X, X2, ls=ls) / 2) + ridge_mat


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Gaussian Process Prior """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def prior(X, ls, kernel_func=rbf,
          ridge_factor=1e-3, name=None):
    """Defines Gaussian Process prior with kernel_func.

    Args:
        X: (np.ndarray of float32) input training features.
        with dimension (N, D).
        kernel_func: (function) kernel function for the gaussian process.
            Default to rbf.
        ls: (float32) length scale parameter.
        ridge_factor: (float32) ridge factor to stabilize Cholesky decomposition.
        name: (str) name of the random variable

    Returns:
        (ed.RandomVariable) A random variable representing the Gaussian Process,
            dimension (N,)

    """
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    N, _ = X.shape.as_list()

    K_mat = kernel_func(X, ls=ls, ridge_factor=ridge_factor)

    return ed.MultivariateNormalTriL(loc=tf.zeros(N, dtype=tf.float32),
                                     scale_tril=tf.cholesky(K_mat),
                                     name=name)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Predictive Sampling functions """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def sample_posterior_mean(X_new, X, f_sample, ls, kern_func=rbf, ridge_factor=1e-3):
    """Sample posterior mean for f^*.

    Posterior for f_new is conditionally independent from other parameters
        in the model, therefore it's conditional posterior mean
        can be obtained by sampling from the posterior conditional f^* | f:

    In particular, we care about posterior predictive mean, i.e.
        E(f^*|f) =  K(X^*, X)K(X, X)^{-1}f

    Args:
        X_new: (np.ndarray of float) testing locations, N_new x D
        X: (np.ndarray of float) training locations, N x D
        f_sample: (np.ndarray of float) M samples of posterior GP sample, N x M
        ls: (float) training lengthscale
        kern_func: (function) kernel function.
        ridge_factor: (float32) small ridge factor to stabilize Cholesky decomposition.

    Returns:
        (np.ndarray) N_new x M vectors of posterior predictive mean samples
    """
    Kx = kern_func(X, X_new, ls=ls)
    K = kern_func(X, ls=ls, ridge_factor=ridge_factor)
    # add ridge factor to stabilize inversion.
    K_inv_f = tf.matrix_solve(K, f_sample)
    return tf.matmul(Kx, K_inv_f, transpose_a=True)


def sample_posterior_full(X_new, X, f_sample, ls,
                          kernel_func=rbf,
                          kernel_func_xn=None,
                          kernel_func_nn=None,
                          ridge_factor=1e-3):
    """Sample posterior predictive distribution.

    Sample posterior conditional from f^* | f ~ MVN, where:

        E(f*|f) = K(X*, X)K(X, X)^{-1}f
        Var(f*|f) = K(X*, X*) - K(X*, X)K(X, X)^{-1}K(X, X*)

    Args:
        X_new: (np.ndarray of float32) testing locations, N_new x D
        X: (np.ndarray of float32) training locations, N x D
        f_sample: (np.ndarray of float32) M samples of posterior GP sample,
            N_obs x N_sample
        ls: (float) training lengthscale
        kernel_func: (function) kernel function for distance among X.
        kernel_func_xn: (function or None) kernel function for distance between X and X_new,
            if None then set to kernel_func.
        kernel_func_nn: (function or None) kernel function for distance among X_new,
            if None then set to kernel_func.
        ridge_factor: (float32) small ridge factor to stabilize Cholesky decomposition.

    Returns:
         (np.ndarray of float32) N_new x M vectors of posterior predictive mean samples
    """
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    X_new = tf.convert_to_tensor(X_new, dtype=tf.float32)
    f_sample = tf.convert_to_tensor(f_sample, dtype=tf.float32)

    N_new, _ = X_new.shape.as_list()
    N, M = f_sample.shape.as_list()

    if kernel_func_xn is None:
        kernel_func_xn = kernel_func
    if kernel_func_nn is None:
        kernel_func_nn = kernel_func

    # compute basic components
    Kxx = kernel_func_nn(X_new, X_new, ls=ls)
    Kx = kernel_func_xn(X, X_new, ls=ls)
    K = kernel_func(X, ls=ls, ridge_factor=ridge_factor)
    K_inv = tf.matrix_inverse(K)

    # compute conditional mean and variance.
    mu_sample = tf.matmul(Kx, tf.matmul(K_inv, f_sample), transpose_a=True)
    Sigma = Kxx - tf.matmul(Kx, tf.matmul(K_inv, Kx), transpose_a=True)

    # sample
    with tf.Session() as sess:
        cond_means, cond_cov = sess.run([mu_sample, Sigma])

    f_new_centered = np.random.multivariate_normal(
        mean=[0] * N_new, cov=cond_cov, size=M).T
    f_new = f_new_centered + cond_means
    return f_new.astype(np.float32)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Variational Family, Mean-field """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def variational_mfvi(X, name, **kwargs):
    """Defines the mean-field variational family for Gaussian Process.

    Args:
        X: (np.ndarray of float32) input training features, with dimension (N, D).
        name: (str) name for variational parameters.
        kwargs: Dict of other keyword variables.
            For compatibility purpose with other variational family.

    Returns:
        q_f, q_sig: (ed.RandomVariable) variational family.
        q_f_mean, q_f_sdev: (tf.Variable) variational parameters for q_f
    """
    X = tf.convert_to_tensor(X, dtype=tf.float32)

    N, D = X.shape.as_list()

    # define variational parameters
    qf_mean = tf.get_variable(shape=[N], name='{}_mean'.format(name))
    qf_sdev = tf.exp(tf.get_variable(shape=[N], name='{}_sdev'.format(name)))

    # define variational family
    q_f = ed.MultivariateNormalDiag(loc=qf_mean,
                                    scale_diag=qf_sdev,
                                    name=name)

    return q_f, qf_mean, qf_sdev


def variational_mfvi_sample(n_sample, qf_mean, qf_sdev):
    """Generates f samples from GPR mean-field variational family.

    Args:
        n_sample: (int) number of samples to draw
        qf_mean: (tf.Tensor of float32) mean parameters for variational family
        qf_sdev: (tf.Tensor of float32) standard deviation for variational family

    Returns:
         (np.ndarray) sampled values.
    """

    """Generates f samples from GPR mean-field variational family."""
    q_f = tfd.MultivariateNormalDiag(loc=qf_mean,
                                     scale_diag=qf_sdev,
                                     name='q_f')
    return q_f.sample(n_sample)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Variational Family: Sparse Gaussian Process """
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


def variational_sgpr(X, Z, ls=1., kernel_func=rbf, ridge_factor=1e-3, name=""):
    """Defines the mean-field variational family for GPR.

    Args:
        X: (np.ndarray of float32) input training features, with dimension (Nx, D).
        Z: (np.ndarray of float32) inducing points, with dimension (Nz, D).
        ls: (float32) length scale parameter.
        kernel_func: (function) kernel function.
        ridge_factor: (float32) small ridge factor to stabilize Cholesky decomposition
        name: (str) name for the variational parameter/random variables.

    Returns:
        q_f, q_sig: (ed.RandomVariable) variational family.
        q_f_mean, q_f_sdev: (tf.Variable) variational parameters for q_f
    """
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    Z = tf.convert_to_tensor(Z, dtype=tf.float32)

    Nx, Nz = X.shape.as_list()[0], Z.shape.as_list()[0]

    # 1. Prepare constants
    # compute matrix constants
    Kxx = kernel_func(X, ls=ls)
    Kxz = kernel_func(X, Z, ls=ls)
    Kzz = kernel_func(Z, ls=ls, ridge_factor=ridge_factor)

    # compute null covariance matrix using Cholesky decomposition
    Kzz_chol_inv = tf.matrix_inverse(tf.cholesky(Kzz))
    Kzz_inv = tf.matmul(Kzz_chol_inv, Kzz_chol_inv, transpose_a=True)

    Kxz_Kzz_chol_inv = tf.matmul(Kxz, Kzz_chol_inv, transpose_b=True)
    Kxz_Kzz_inv = tf.matmul(Kxz, Kzz_inv)
    Sigma_pre = Kxx - tf.matmul(Kxz_Kzz_chol_inv, Kxz_Kzz_chol_inv, transpose_b=True)

    # 2. Define variational parameters
    # define free parameters (i.e. mean and full covariance of f_latent)
    m = tf.get_variable(shape=[Nz], name='{}_mean_latent'.format(name))
    s = tf.get_variable(shape=[Nz * (Nz + 1) / 2], name='{}_cov_latent_s'.format(name))
    L = fill_triangular(s, name='{}_cov_latent_chol'.format(name))
    S = tf.matmul(L, L, transpose_b=True, name='{}_cov_latent'.format(name))

    # compute sparse gp variational parameter
    # (i.e. mean and covariance of P(f_obs | f_latent))
    qf_mean = tf.tensordot(Kxz_Kzz_inv, m, [[1], [0]], name='{}_mean'.format(name))
    qf_cov = (Sigma_pre +
              tf.matmul(Kxz_Kzz_inv,
                        tf.matmul(S, Kxz_Kzz_inv, transpose_b=True)) +
              ridge_factor * tf.eye(Nx, dtype=tf.float32))

    # define variational family
    q_f = ed.MultivariateNormalFullCovariance(loc=qf_mean,
                                              covariance_matrix=qf_cov,
                                              name=name)

    return q_f, qf_mean, qf_cov


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
                                               covariance_matrix=qf_cov)
    return q_f.sample(n_sample)
