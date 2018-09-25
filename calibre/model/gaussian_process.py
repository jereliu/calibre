"""Model definitions and sampling functions for Gaussian Process Prior"""
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed

tfd = tfp.distributions

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Helper functions """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def square_dist(X, X2=None, ls=1.):
    """Computes Square distance between two sets of features.

    Taken from GPflow.kernels.Stationary.

    Args:
        X: (np.ndarray) First set of features of dim N x D.
        X2: (np.ndarray or None) Second set of features of dim N2 x D.
        ls: (float) value for length scale.

    Returns:
        (tf.Tensor) A N_new x N tensor for ||x-x'||^2
    """
    X = X / ls
    Xs = tf.reduce_sum(tf.square(X), axis=1)

    if X2 is None:
        dist = -2 * tf.matmul(X, X, transpose_b=True)
        dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
        return tf.clip_by_value(dist, 0., np.inf)

    X2 = X2 / ls
    X2s = tf.reduce_sum(tf.square(X2), axis=1)
    dist = -2 * tf.matmul(X, X2, transpose_b=True)
    dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))
    return tf.clip_by_value(dist, 0., np.inf)


def rbf(X, X2=None, ls=1., ridge_factor=0.):
    """Defines RBF kernel.

    Args:
        X: (np.ndarray) First set of features of dim N x D.
        X2: (np.ndarray or None) Second set of features of dim N2 x D.
        ls: (float) value for length scale
        ridge_factor: (float32) ridge factor to stabilize Cholesky decomposition.

    Returns:
        (tf.Tensor) A N x N2 tensor for exp(-||x-x'||^2/2*ls)
    """
    if ridge_factor:
        ridge_mat = ridge_factor * tf.eye(X.shape[0])
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
        (ed.RandomVariable) A random variable representing the Gaussian Process

    """
    N = X.shape[0]
    K_mat = kernel_func(X, ls=ls, ridge_factor=ridge_factor)

    return ed.MultivariateNormalTriL(loc=tf.zeros(N),
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


def sample_posterior_full(X_new, X, f_sample, ls, kern_func=rbf, ridge_factor=1e-3):
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
        kern_func: (function) kernel function.
        ridge_factor: (float32) small ridge factor to stabilize Cholesky decomposition.
    Returns:
         (np.ndarray of float32) N_new x M vectors of posterior predictive mean samples
    """
    N_new, _ = X_new.shape
    N, M = f_sample.shape

    # compute basic components
    Kxx = kern_func(X_new, X_new, ls=ls)
    Kx = kern_func(X, X_new, ls=ls)
    K = kern_func(X, ls=ls, ridge_factor=ridge_factor)
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
