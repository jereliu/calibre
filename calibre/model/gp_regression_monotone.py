"""Defines 1D Monotonic Gaussian Process regression model.

For a Gaussian Process f ~ N(0, K), its derivative f' ~ N(0, ddK)
also follows a Gaussian Process (since derivative is a linear operator).
We can then write out its joint likelihood:

[f, f'] ~ N( [0], [ K,   dK  ]  )
             [0], [ dK', ddK ]

where if K[i, j] = k(x, x'), then
    dK is the "Gradient Kernel Matrix":  dK[i, j] = dx k(x, x')
    ddK is the "Hessian Kernel Matrix": ddK[i, j] = dx dx' k(x, x'))

Consequently, we can perform monotonic regression by define a
    random variable C representing the positivity constraint on the
    derivative, such that it has support on {f' | f' > 0}.

Write out the likelihood:
    P(f, f' | y, C) ~ P(y | f) P(C | f')  P(f, f')

where P(C | f') can be Probit [1] or Logistic [2].

#### References

[1]:    Jaakko Riihimaki and Aki Vehtari. Gaussian processes with monotonicity information.
        _13th International Conference on Artificial Intelligence and Statistics (AISTATS)_
        2010. http://proceedings.mlr.press/v9/riihimaki10a/riihimaki10a.pdf
[2]:    Marco Lorenzi and Maurizio Filippone. Constraining the Dynamics of Deep Probabilistic Models.
        _35th International Conference on Machine Learning_, 2018.
        http://proceedings.mlr.press/v80/lorenzi18a.html
[3]:    Carl Rasmussen and Christopher Williams. Gaussian Processes for Machine Learning.
        _The MIT Press. ISBN 0-262-18253-X_. 2006
"""
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed

import calibre.model.gaussian_process as gp
import calibre.model.gp_regression as gpr

from tensorflow.python.ops.distributions.util import fill_triangular

tfd = tfp.distributions

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Gradient and Hessian Kernel functions """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def pair_diff_1d(X, X2=None, ls=1.):
    """Computes pairwise difference between two sets of 1D features.

    Args:
        X: (tf.Tensor) First set of features of dim N x D.
        X2: (tf.Tensor or None) Second set of features of dim N2 x D.
        ls: (float) value for length scale.

    Returns:
        (tf.Tensor) A N x N2 tensor for (x - x') / ls

    Raises:
        (ValueError) If feature dimension of X and X2 disagrees.
    """
    N, D = X.shape

    X = X / ls

    if X2 is None:
        return X - tf.reshape(X, (D, N))

    N2, D2 = X2.shape
    if D != D2:
        raise ValueError('Dimension of X and X2 does not match.')

    X2 = X2 / ls
    return X - tf.reshape(X2, (D2, N2))


def rbf_grad_1d(X, X2=None, ls=1., ridge_factor=0.):
    """Defines RBF gradient kernel for 1D input.

    Here gradient is taken with respect to left input x:

    dx k(x, x') = - (1 / ls) * ( (x - x') / ls ) *
                    exp(- || x - x' ||**2 / 2 * ls**2)

    Args:
        X: (tf.Tensor) First set of features of dim N x D.
        X2: (tf.Tensor or None) Second set of features of dim N2 x D.
        ls: (float) value for length scale
        ridge_factor: (float32) ridge factor to stabilize Cholesky decomposition.

    Returns:
        (tf.Tensor) A N x N2 tensor.

    Raises:
        (ValueError) If Dimension of X is not 1.
    """
    N, D = X.shape
    if D != 1:
        raise ValueError('Feature dimension of X must be 1.')

    if ridge_factor:
        ridge_mat = ridge_factor * tf.eye(X.shape[0])
    else:
        ridge_mat = 0

    return (-(1 / ls) * pair_diff_1d(X, X2, ls=ls) *
            tf.exp(-gp.square_dist(X, X2, ls=ls) / 2) + ridge_mat)


def rbf_hess_1d(X, X2=None, ls=1., ridge_factor=0.):
    """Defines RBF hessian kernel for 1D input.

    dxdx' k(x, x') = (1 / ls**2) *
                     (1 -  ||x - x'||**2 / ls**2 ) *
                     exp(- ||x - x'||**2 / 2 * ls**2 )

    Args:
        X: (tf.Tensor) First set of features of dim N x D.
        X2: (tf.Tensor or None) Second set of features of dim N2 x D.
        ls: (float) value for length scale
        ridge_factor: (float32) ridge factor to stabilize Cholesky decomposition.

    Returns:
        (tf.Tensor) A N x N2 tensor.

    Raises:
        (ValueError) If Dimension of X is not 1.
    """
    N, D = X.shape
    if D != 1:
        raise ValueError('Feature dimension of X must be 1.')

    if ridge_factor:
        ridge_mat = ridge_factor * tf.eye(X.shape[0])
    else:
        ridge_mat = 0

    return ((1 / ls ** 2) * (1 - gp.square_dist(X, X2, ls=ls)) *
            tf.exp(-gp.square_dist(X, X2, ls=ls) / 2) + ridge_mat)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Main model definition """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def deriv_prior(f, X, ls,
                kernel_func=gp.rbf,
                grad_func=rbf_grad_1d,
                hess_func=rbf_hess_1d,
                ridge_factor=1e-3,
                name="gp_f_deriv"):
    """Defines the conditional prior distribution of f' | f.

    Recall that the joint prior of [f, f'] is multivariate Gaussian

        [f, f'] ~ N( [0], [ K,   dK  ]  )
                     [0], [ dK', ddK ]

    Consequently, the conditional prior f' | f is:
        f' | f ~ N( Mu    = dK inv(K) f,
                    Sigma = ddK - dK inv(K) dK' )

    Args:
        f: (ed.RandomVariable) Gaussian Process prior
        X: (tf.Tensor) Features with dimension (N, D).
        ls: (tf.Tensor of float32) length scale parameter for kernel function.
        kernel_func: (function) Model's kernel function, default to gp.rbf
        grad_func: (function) Gradient kernel function for kernel_func
        hess_func: (function) Hessian kernel function for kernel_func
        ridge_factor: (float32) ridge factor to stabilize Cholesky decomposition.
        name: (str) name of the random variable.

    Returns:
        (ed.RandomVariable) Conditional Prior for Gaussian Process Derivative.
    """
    # compute basic components
    dK = grad_func(X, ls=ls)
    ddK = hess_func(X, ls=ls, ridge_factor=ridge_factor)
    K = kernel_func(X, ls=ls, ridge_factor=ridge_factor)
    K_inv = tf.matrix_inverse(K)

    # compute conditional mean and variance.
    Sigma = ddK - tf.matmul(dK, tf.matmul(K_inv, dK, transpose_b=True))
    Mu = tf.matmul(dK, tf.matmul(K_inv, tf.expand_dims(f, -1)))

    return ed.MultivariateNormalTriL(loc=tf.squeeze(Mu),
                                     scale_tril=tf.cholesky(Sigma),
                                     name=name)


def model(X, ls=1., ridge_factor=1e-3):
    """Defines the Gaussian Process Model with derivative random variable.

    Args:
        X: (np.ndarray of float32) input training features.
        with dimension (N, D).
        ls: (float32) length scale parameter.
        ridge_factor: (float32) ridge factor to stabilize Cholesky decomposition.

    Returns:
         (tf.Tensors of float32) model parameters.
    """
    if not ls:
        ls = ed.Normal(loc=-5., scale=1., name='ls')
    else:
        ls = tf.convert_to_tensor(ls, dtype=tf.float32)

    gp_f = gp.prior(X, ls, kernel_func=gp.rbf,
                    ridge_factor=ridge_factor, name="gp_f")
    gp_f_deriv = deriv_prior(gp_f, X, ls, kernel_func=gp.rbf,
                             grad_func=rbf_grad_1d, hess_func=rbf_hess_1d,
                             ridge_factor=ridge_factor, name="gp_f_deriv")

    sigma = ed.Normal(loc=-5., scale=1., name='sigma')

    y = ed.MultivariateNormalDiag(loc=gp_f,
                                  scale_identity_multiplier=tf.exp(sigma),
                                  name="y")

    return gp_f, gp_f_deriv, sigma, y, ls


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Variational family I: Mean field """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def variational_mfvi(X, ls,
                     kernel_func=gp.rbf,
                     grad_func=rbf_grad_1d,
                     hess_func=rbf_hess_1d,
                     ridge_factor=1e-3):
    """Defines the mean-field variational family for GPR.

    Approximate with independent, fully-factorized Gaussian RVs,
    with exception of dependency between qf_mean and qf_deriv_mean
    through the conditional mean formula:
        qf_mean = dK^T ddK qf_deriv_mean

    Args:
        X: (np.ndarray of float32) input training features.
        with dimension (N, D).
        ls: (tf.Tensor of float32) length scale parameter for kernel function.
        kernel_func: (function) Model's kernel function, default to gp.rbf
        grad_func: (function) Gradient kernel function for kernel_func
        hess_func: (function) Hessian kernel function for kernel_func
        ridge_factor: (float32) ridge factor to stabilize Cholesky decomposition.

    Returns:
        q_f, q_f_deriv, q_sig:
            (ed.RandomVariable) variational family.
        q_f_mean, q_f_sdev, qf_deriv_mean, qf_deriv_sdev:
            (tf.Variable) variational parameters for q_f
    """
    N, D = X.shape

    # compute basic components
    dK = grad_func(X, ls=ls)
    ddK = hess_func(X, ls=ls, ridge_factor=ridge_factor)
    ddK_inv = tf.matrix_inverse(ddK)

    # define variational parameters
    qf_deriv_mean = tf.get_variable(shape=[N], name='qf_deriv_mean')
    qf_deriv_sdev = tf.exp(tf.get_variable(shape=[N], name='qf_deriv_sdev'))

    qf_mean = tf.squeeze(tf.matmul(
        dK, tf.matmul(ddK_inv, tf.expand_dims(qf_deriv_mean, -1)),
        transpose_a=True, name='qf_mean'))
    qf_sdev = tf.exp(tf.get_variable(shape=[N], name='qf_sdev'))

    q_sig_mean = tf.get_variable(shape=[], name='q_sig_mean')
    q_sig_sdev = tf.exp(tf.get_variable(shape=[], name='q_sig_sdev'))

    # define variational family
    q_f = ed.MultivariateNormalDiag(loc=qf_mean, scale_diag=qf_sdev,
                                    name='q_f')
    q_f_deriv = ed.MultivariateNormalDiag(loc=qf_deriv_mean,
                                          scale_diag=qf_deriv_sdev,
                                          name='q_f_deriv')
    q_sig = ed.Normal(loc=q_sig_mean, scale=q_sig_sdev, name='q_sig')

    return q_f, q_f_deriv, q_sig, qf_mean, qf_sdev, qf_deriv_mean, qf_deriv_sdev


variational_mfvi_sample = gpr.variational_mfvi_sample

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


def variational_sgpr(X, Z, ls=1.,
                     kern_func=gp.rbf,
                     grad_func=rbf_grad_1d,
                     hess_func=rbf_hess_1d,
                     ridge_factor=1e-3):
    """Defines the mean-field variational family for GPR.

    f_deriv: mean-field approximation
    f: Mu is approximated using conditional mean wrt f_deriv,
        Cov is approximated using sparse GP.

    Args:
        X: (np.ndarray of float32) input training features, with dimension (Nx, D).
        Z: (np.ndarray of float32) inducing points, with dimension (Nz, D).
        ls: (float32) length scale parameter.
        kern_func: (function) kernel function.
        grad_func: (function) Gradient kernel function for kernel_func
        hess_func: (function) Hessian kernel function for kernel_func
        ridge_factor: (float32) small ridge factor to stabilize Cholesky decomposition

    Returns:
        q_f, q_f_deriv, q_sig:
            (ed.RandomVariable) variational family.
        q_f_mean, q_f_sdev, qf_deriv_mean, qf_deriv_sdev:
            (tf.Variable) variational parameters for q_f
    """
    Nx, Nz = X.shape[0], Z.shape[0]

    # 1. Prepare constants
    # compute matrix constants
    dK = grad_func(X, ls=ls)
    ddK = hess_func(X, ls=ls, ridge_factor=ridge_factor)
    ddK_inv = tf.matrix_inverse(ddK)

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
    s = tf.get_variable(shape=[Nz * (Nz + 1) / 2],
                        # initializer=tf.zeros_initializer(),
                        name='qf_s')
    L = fill_triangular(s, name='qf_chol')
    S = tf.matmul(L, L, transpose_b=True)

    # compute sparse gp variational parameter (i.e. mean and covariance of P(f_obs | f_latent))
    # define variational parameters
    qf_deriv_mean = tf.get_variable(shape=[Nx], name='qf_deriv_mean')
    qf_deriv_sdev = tf.exp(tf.get_variable(shape=[Nx], name='qf_deriv_sdev'))

    qf_mean = tf.squeeze(tf.matmul(
        dK, tf.matmul(ddK_inv, tf.expand_dims(qf_deriv_mean, -1)),
        transpose_a=True, name='qf_mean'))
    qf_cov = (Sigma_pre +
              tf.matmul(Kxz_Kzz_inv,
                        tf.matmul(S, Kxz_Kzz_inv, transpose_b=True)) +
              ridge_factor * tf.eye(Nx)
              )

    # define variational family
    q_f = ed.MultivariateNormalFullCovariance(loc=qf_mean,
                                              covariance_matrix=qf_cov,
                                              name='q_f')
    q_f_deriv = ed.MultivariateNormalDiag(loc=qf_deriv_mean,
                                          scale_diag=qf_deriv_sdev,
                                          name='q_f_deriv')
    q_sig = ed.Normal(loc=q_sig_mean,
                      scale=q_sig_sdev, name='q_sig')

    return (q_f, q_f_deriv, q_sig,
            qf_mean, qf_cov,
            qf_deriv_mean, qf_deriv_sdev)


variational_sgpr_sample = gpr.variational_sgpr_sample
