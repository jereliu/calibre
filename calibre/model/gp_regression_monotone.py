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

import calibre.util.matrix as matrix_util
import calibre.util.inference as inference_util

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


def deriv_prior(f, X, X_deriv=None, ls=1.,
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
        X_deriv: (tf.Tensor or None) Feature location to place derivative
            constraint on shape (N_deriv, D). If None, X_deriv=X
        ls: (tf.Tensor of float32) length scale parameter for kernel function.
        kernel_func: (function) Model's kernel function, default to gp.rbf
        grad_func: (function) Gradient kernel function for kernel_func
        hess_func: (function) Hessian kernel function for kernel_func
        ridge_factor: (float32) ridge factor to stabilize Cholesky decomposition.
        name: (str) name of the random variable.

    Returns:
        (ed.RandomVariable) Conditional Prior for Gaussian Process Derivative.
    """
    if X_deriv is None:
        X_deriv = X

    # compute basic components
    dK = grad_func(X_deriv, X, ls=ls)
    ddK = hess_func(X_deriv, ls=ls, ridge_factor=ridge_factor)
    K = kernel_func(X, ls=ls, ridge_factor=ridge_factor)
    K_inv = tf.matrix_inverse(K)

    # compute conditional mean and variance.
    Sigma = ddK - tf.matmul(dK, tf.matmul(K_inv, dK, transpose_b=True))
    Mu = tf.matmul(dK, tf.matmul(K_inv, tf.expand_dims(f, -1)))

    return ed.MultivariateNormalTriL(loc=tf.squeeze(Mu),
                                     scale_tril=tf.cholesky(Sigma),
                                     name=name)


def model(X, X_deriv=None, ls=1., ridge_factor=1e-3):
    """Defines the Gaussian Process Model with derivative random variable.

    Args:
        X: (np.ndarray of float32) input training features.
        with dimension (N, D).
        X_deriv: (tf.Tensor or None) Feature location to place derivative
            constraint on shape (N_deriv, D). If None, X_deriv=X
        ls: (float32) length scale parameter.
        ridge_factor: (float32) ridge factor to stabilize Cholesky decomposition.

    Returns:
         (tf.Tensors of float32) model parameters.
    """
    # TODO(jereliu): Add predictive prior
    if not ls:
        ls = ed.Normal(loc=-5., scale=1., name='ls')
    else:
        ls = tf.convert_to_tensor(ls, dtype=tf.float32)

    gp_f = gp.prior(X, ls, kernel_func=gp.rbf,
                    ridge_factor=ridge_factor, name="gp_f")
    gp_f_deriv = deriv_prior(gp_f, X, X_deriv, ls,
                             kernel_func=gp.rbf,
                             grad_func=rbf_grad_1d, hess_func=rbf_hess_1d,
                             ridge_factor=ridge_factor, name="gp_f_deriv")

    sigma = ed.Normal(loc=-5., scale=1., name='sigma')

    y = ed.MultivariateNormalDiag(loc=gp_f,
                                  scale_identity_multiplier=tf.exp(sigma),
                                  name="y")

    return gp_f, gp_f_deriv, sigma, y, ls


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Model likelihood function for MCMC and VI """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def make_log_likelihood_function(X_train, X_deriv, y_train, ls=None,
                                 ridge_factor=5e-3, deriv_prior_scale=1e-3,
                                 cdf_constraint=False):
    """Makes log joint likelihood function for monotonic GP regression.

    To be used for MCMC sampling.

    Args:
        X_train: (tf.Tensor) Training samples, shape (N_obs, D)
        X_deriv: (tf.Tensor) Locations of derivative constraints,
            shape (N_deriv, D)
        y_train: (tf.Tensor) Training labels, shape (N_obs, )
        ls: (tf.Tensor or None) Value of length scale parameter,
            if None then will be added to the likelihood function.
        ridge_factor: (float32) ridge factor to stabilize Cholesky decomposition.
        deriv_prior_scale: (float32) Value for Prior scale parameter for
            probit likelihood function imposing positivity contraint on
            f_deriv. For detail, see [1].
        cdf_constraint: (bool) Whether to impose cdf constraint
            (i.e. f < 0 and f > 1).

    Returns:
        (function) A log-joint probability function.
            Its inputs are `model`'s parameters, and output the model's
            un-normalized log likelihood (a scalar tf.Tensor).

    """
    model_var_names = ["gp_f", "gp_f_deriv", "sigma"]

    if not ls:
        model_var_names += ["ls"]

    log_joint = ed.make_log_joint_fn(model)

    def target_log_prob_fn(*model_var_positional_args):
        """Unnormalized target density as a function of states."""

        # prepare positional arguments,
        # specifically, add back ls if its not a model parameter.
        model_var_kwargs = dict(zip(model_var_names,
                                    model_var_positional_args))

        if "ls" not in model_var_kwargs.keys():
            model_var_kwargs["ls"] = ls

        # the probit likelihood on derivative
        log_lkhd_constraint = tfd.Normal(
            loc=0., scale=deriv_prior_scale).log_cdf(
            model_var_kwargs["gp_f_deriv"])

        if cdf_constraint:
            # constraint 1: f > 0
            log_lkhd_constraint_ge_zero = tfd.Normal(
                loc=0., scale=deriv_prior_scale).log_cdf(
                model_var_kwargs["gp_f"])
            # constraint 2: f < 1
            log_lkhd_constraint_le_one = tfd.Normal(
                loc=0., scale=deriv_prior_scale).log_cdf(
                1 - model_var_kwargs["gp_f"])

            log_lkhd_constraint = tf.concat([
                log_lkhd_constraint,
                log_lkhd_constraint_ge_zero,
                log_lkhd_constraint_le_one
            ], axis=0)

        log_joint_rest = log_joint(
            X=X_train, X_deriv=X_deriv, y=y_train,
            ridge_factor=ridge_factor,
            **model_var_kwargs)
        return tf.reduce_mean(log_lkhd_constraint) + log_joint_rest

    return target_log_prob_fn


def make_log_likelihood_tensor(gp_f, gp_f_deriv,
                               y, y_train,
                               deriv_prior_scale=1e-3,
                               cdf_constraint=False):
    """Makes log joint likelihood tensor for monotonic GP regression.

    To be used for Variational Inference.

    Args:
        gp_f: (ed.RandomVariable) RV from Monotonic GP prob program
            corresponding to function.
        gp_f_deriv: (ed.RandomVariable) RV from Monotonic GP prob program
            corresponding to function derivative.
        y: (ed.RandomVariable) RV from Monotonic GP prob program
            corresponding to observed label.
        y_train: (np.ndarray) Training labels corresponding to y
        deriv_prior_scale: (float32) Value for Prior scale parameter for
            probit likelihood function imposing positivity contraint on
            f_deriv. For detail, see [1].
        cdf_constraint: (bool) Whether to impose cdf constraint
            (i.e. f > 0 and f < 1).

    Returns:
        (tf.Tensor) A scalar tf.Tensor corresponding to model likelihood,
            with dtype float32
    """
    # probit likelihood for derivative constraint
    log_lkhd_constraint = tfd.Normal(loc=0.,
                                     scale=deriv_prior_scale).log_cdf(gp_f_deriv)
    if cdf_constraint:
        # constraint 1: f > 0
        log_lkhd_constraint_ge_zero = tfd.Normal(
            loc=0., scale=deriv_prior_scale).log_cdf(gp_f)
        # constraint 2: f < 1
        log_lkhd_constraint_le_one = tfd.Normal(
            loc=0., scale=deriv_prior_scale).log_cdf(1 - gp_f)

        log_lkhd_constraint = tf.concat([
            log_lkhd_constraint,
            log_lkhd_constraint_ge_zero,
            log_lkhd_constraint_le_one
        ], axis=0)

    # likelihood for the rest of the model parameters
    log_lkhd_rest = y.distribution.log_prob(y_train)
    return tf.reduce_mean(log_lkhd_constraint) + log_lkhd_rest


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Posterior sampling """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def sample_posterior_predictive(X_new, X_obs, X_deriv,
                                f_sample, f_deriv_sample,
                                kernel_func_ff=gp.rbf,
                                kernel_func_df=rbf_grad_1d,
                                kernel_func_dd=rbf_hess_1d,
                                ls=None, ridge_factor=1e-3):
    """Sample posterior predictive of f based on f_train and f_derive.

    The full joint likelihood of f_new (f_n), f_obs (f_o), f_deriv (f_d) is:

                    [[f_n]
                     [f_o]  ~ N(0, K)
                     [f_d]]

    where (denote K'/K'' the gradient/hessian kernel matrix)

                    [K_nn   K_no    K'_nd]
            K    =  [K_on   K_oo    K'_od]
                    [K'_dn  K'_do   K''_dd]

    Therefore, if we denote
        K_n_od = [K_no, K'_nd]      and     K_odod =    [K_oo    K'_od]
                                                        [K'_do   K''_dd]

    then
        f_n ~ N(Mu, Sigma), where (note * is matrix product here):

        Mu = K_n_od * inv(K_odod) * [f_o, f_d]^T
        Sigma = K_nn - K_n_od * inv(K_odod) * K_n_od^T

    Args:
        X_new: (np.ndarray of float32) testing locations, N_new x D
        X_obs: (np.ndarray of float32) training locations, N_obs x D
        X_deriv: (np.ndarray of float32) training locations, N_deriv x D
        f_sample: (np.ndarray of float32) Samples for f in training set,
            N_train x N_sample
        f_deriv_sample: (np.ndarray of float32) Samples for f_deriv,
            N_deriv x N_sample
        kernel_func_ff: (function) kernel function for k(x, x')
        kernel_func_df: (function) gradient kernel function dx k(x, x')
        kernel_func_dd: (function) hessian kernel function dxdx' k(x, x')
        ls: (float32) Length scale parameter
        ridge_factor: (float32) ridge factor to stabilize Cholesky decomposition.

    Returns:
         (np.ndarray of float32) N_new x N_sample vectors of posterior
            predictive samples.

    Raises:
        (ValueError) If length scale parameter is not provided.
    """
    N_new = X_new.shape[0]
    _, N_sample = f_sample.shape
    _, N_sample_2 = f_deriv_sample.shape

    if N_sample != N_sample_2:
        raise ValueError('Sample size for f_sample ({}) and '
                         'f_deriv_sample ({}) must be same.'.format(N_sample, N_sample_2))

    if not ls:
        raise ValueError('Length scale parameter ("ls") must be provided.')

    # compute matrix components
    K_nn = kernel_func_ff(X_new, ls=ls)

    K_no = kernel_func_ff(X_new, X_obs, ls=ls)
    dK_dn = kernel_func_df(X_deriv, X_new, ls=ls)

    K_oo = kernel_func_ff(X_obs, ls=ls)
    dK_do = kernel_func_df(X_deriv, X_obs, ls=ls)
    ddK_dd = kernel_func_dd(X_deriv, ls=ls)

    # assemble matrix and sample
    K_n_od = matrix_util.make_block_matrix(K_no, tf.transpose(dK_dn))
    K_odod = matrix_util.make_block_matrix(K_oo, tf.transpose(dK_do), ddK_dd,
                                           ridge_factor=ridge_factor)
    K_odod_inv = tf.matrix_inverse(K_odod)

    f_sample_all = tf.concat([f_sample, f_deriv_sample], axis=0)

    # compute conditional mean and variance.
    mu_sample = tf.matmul(K_n_od, tf.matmul(K_odod_inv, f_sample_all))
    Sigma = K_nn - tf.matmul(K_n_od, tf.matmul(K_odod_inv, K_n_od, transpose_b=True))

    # sample
    with tf.Session() as sess:
        cond_means, cond_cov = sess.run([mu_sample, Sigma])

    f_new_centered = np.random.multivariate_normal(
        mean=[0] * N_new, cov=cond_cov, size=N_sample).T
    f_new = f_new_centered + cond_means
    return f_new.astype(np.float32)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Variational family I: Mean field """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def variational_mfvi(X, X_deriv=None, ls=1.,
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
        X: (np.ndarray of float32) input training features,
            with dimension (N, D).
        X_deriv: (tf.Tensor or None) Feature location to place derivative
            constraint on shape (Nd, D). If None, X_deriv=X
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

    Raises:
        (ValueError) If Feature dimension of X and X_deriv are not same.
    """
    N, D = X.shape
    Nd, Dd = X_deriv.shape

    if D != Dd:
        raise ValueError("Feature dimension of X and X_deriv must be same!")

    # compute basic components
    dK = grad_func(X_deriv, X, ls=ls)
    ddK = hess_func(X_deriv, ls=ls, ridge_factor=ridge_factor)
    ddK_inv = tf.matrix_inverse(ddK)

    # define variational parameters
    qf_deriv_mean = tf.get_variable(shape=[Nd],
                                    name='qf_deriv_mean')
    qf_deriv_sdev = tf.exp(tf.get_variable(shape=[Nd],
                                           name='qf_deriv_sdev'))

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


def variational_sgpr(X, Z, ls,
                     X_deriv=None, Z_deriv=None,
                     kern_func=gp.rbf,
                     grad_func=rbf_grad_1d,
                     hess_func=rbf_hess_1d,
                     ridge_factor=1e-3):
    """Defines the mean-field variational family for GPR.

    f_deriv: Both Mu and Cov approximated using sparse GP
    f: Mu is approximated using conditional mean wrt f_deriv,
       Cov is approximated using sparse GP.

    Args:
        X: (np.ndarray of float32) input training features, with dimension (Nx, D).
        Z: (np.ndarray of float32) inducing points, with dimension (Nz, D).
        ls: (float32) length scale parameter.
        X_deriv: (np.ndarray of float32) Feature location to place derivative
            constraint on shape (N_deriv, D). If None, X_deriv=X.
        Z_deriv: (np.ndarray of float32) inducing points, with dimension (Nz, D).
            If None, then Z_deriv=Z.
        kern_func: (function) kernel function.
        grad_func: (function) Gradient kernel function for kernel_func
        hess_func: (function) Hessian kernel function for kernel_func
        ridge_factor: (float32) small ridge factor to stabilize Cholesky decomposition

    Returns:
        q_f, q_f_deriv, q_sig:
            (ed.RandomVariable) variational family.
        Mu_f, Sigma_f, Mu_df, Sigma_df:
            (tf.Variable) variational parameters for q_f and q_f_deriv
            
    Raises:
        (ValueError) If Feature dimension of X / X_deriv / Z are not same.
    """
    if X_deriv is None:
        X_deriv = X
    if Z_deriv is None:
        Z_deriv = Z

    Nx, D = X.shape
    Nd, Dd = X_deriv.shape
    Nxz, Dxz = Z.shape
    Ndz, Ddz = Z_deriv.shape

    if np.any([D != Dd, D != Dxz, D != Ddz]):
        raise ValueError("Feature dimension of Z, X_deriv, Z_deriv "
                         "must be same as X!")

    # 1. Define variational parameters
    # define mean and variance for sigma
    q_sig_mean = tf.get_variable(shape=[], name='q_sig_mean')
    q_sig_sdev = tf.exp(tf.get_variable(shape=[], name='q_sig_sdev'))

    # define mean and cov for latent gp_deriv
    m_df = tf.get_variable(shape=[Ndz], name='qf_deriv_m')

    s_df = tf.get_variable(shape=[Ndz * (Ndz + 1) / 2], name='qf_deriv_s')
    L_df = fill_triangular(s_df, name='qf_deriv_chol')
    S_df = tf.matmul(L_df, L_df, transpose_b=True)

    # define cov for latent gp_deriv
    s_f = tf.get_variable(shape=[Nxz * (Nxz + 1) / 2], name='qf_s')
    L_f = fill_triangular(s_f, name='qf_chol')
    S_f = tf.matmul(L_f, L_f, transpose_b=True)

    # 2. Define sparse GP parameters
    # parameters for observed f_deriv
    Mu_df, Sigma_df = inference_util.make_sparse_gp_parameters(
        m_df, S_df, X_deriv, Z_deriv, ls=ls,
        kern_func=hess_func,
        mean_name='qf_deriv_mean')

    # parameters for observed f
    dK = grad_func(X_deriv, X, ls=ls)
    ddK = hess_func(X_deriv, ls=ls, ridge_factor=ridge_factor)
    ddK_inv = tf.matrix_inverse(ddK)

    Mu_f = tf.squeeze(tf.matmul(
        dK, tf.matmul(ddK_inv, tf.expand_dims(Mu_df, -1)),
        transpose_a=True, name='qf_mean'))
    _, Sigma_f = inference_util.make_sparse_gp_parameters(
        None, S_f, X, Z, ls=ls,
        kern_func=gp.rbf, compute_mean=False)

    # 3. Define variational family
    q_f = ed.MultivariateNormalFullCovariance(loc=Mu_f,
                                              covariance_matrix=Sigma_f,
                                              name='q_f')
    q_f_deriv = ed.MultivariateNormalFullCovariance(loc=Mu_df,
                                                    covariance_matrix=Sigma_df,
                                                    name='q_f_deriv')
    q_sig = ed.Normal(loc=q_sig_mean,
                      scale=q_sig_sdev, name='q_sig')

    return (q_f, q_f_deriv, q_sig,
            Mu_f, Sigma_f, Mu_df, Sigma_df)


variational_sgpr_sample = gpr.variational_sgpr_sample
