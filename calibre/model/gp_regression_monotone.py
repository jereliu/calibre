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

where P(C | f') can be Probit [2] or Logistic [3].

#### References

[1]:    Carl Rasmussen and Christopher Williams. Gaussian Processes for Machine Learning.
        _The MIT Press. ISBN 0-262-18253-X_. 2006
[2]:    Jaakko Riihimaki and Aki Vehtari. Gaussian processes with monotonicity information.
        _13th International Conference on Artificial Intelligence and Statistics (AISTATS)_
        2010. http://proceedings.mlr.press/v9/riihimaki10a/riihimaki10a.pdf
[3]:    Marco Lorenzi and Maurizio Filippone. Constraining the Dynamics of Deep Probabilistic Models.
        _35th International Conference on Machine Learning_, 2018.
        http://proceedings.mlr.press/v80/lorenzi18a.html
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

# TODO(jereliu): add proper equality constraint (f(0)=0 and f(1)=1)
# through spike likelihood on f(0)=0 and f(1)=1.

DEFAULT_PAR_SHIFT = np.array(-5.).astype(np.float32)
DEFAULT_PAR_SCALE = np.array(1.).astype(np.float32)
DEFAULT_CDF_CENTER = np.array(0.).astype(np.float32)

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
    N, D = X.shape.as_list()
    if D != 1:
        raise ValueError('Feature dimension of X must be 1.')

    if ridge_factor and X2 is None:
        ridge_mat = ridge_factor * tf.eye(N, dtype=tf.float32)
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
    N, D = X.shape.as_list()
    if D != 1:
        raise ValueError('Feature dimension of X must be 1.')

    if ridge_factor and X2 is None:
        ridge_mat = ridge_factor * tf.eye(N, dtype=tf.float32)
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


def pred_cond_prior_v0(gp, gp_deriv,
                       X_new, X_obs, X_deriv, ls,
                       kernel_func_ff=gp.rbf,
                       kernel_func_df=rbf_grad_1d,
                       kernel_func_dd=rbf_hess_1d,
                       ridge_factor=1e-3,
                       name="gp_pred"):
    """Defines the prior of f_pred | f_obs, f_derive.

    The full joint likelihood of f_new, f_obs, f_deriv (denote as: f_n, f_o, f_d) is:

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
        gp: (ed.RandomVariable) f_obs corresponding to X_obs in
            training dataset, shape (N_train, )
        gp_deriv: (ed.RandomVariable) f_deriv, (N_deriv, )
        X_new: (tf.Tensor of float32) testing locations, (N_new, D)
        X_obs: (tf.Tensor of float32) training locations, (N_obs, D)
        X_deriv: (tf.Tensor of float32) training locations, (N_deriv, D)
        ls: (float32) Length scale parameter
        kernel_func_ff: (function) kernel function for k(x, x')
        kernel_func_df: (function) gradient kernel function dx k(x, x')
        kernel_func_dd: (function) hessian kernel function dxdx' k(x, x')
        ridge_factor: (float32) ridge factor to stabilize Cholesky decomposition.
        name: (str) name of the random variable.

    Returns:
         (tf.Tensor of float32) N_new x N_sample vectors of posterior
            predictive samples.

    Raises:
        (ValueError) If length scale parameter is not provided.
    """
    # compute matrix components
    K_nn = kernel_func_ff(X_new, ls=ls, ridge_factor=ridge_factor)

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

    gp_all = tf.expand_dims(tf.concat([gp, gp_deriv], axis=0), -1)

    # compute conditional mean and variance.
    Mu = tf.matmul(K_n_od, tf.matmul(K_odod_inv, gp_all))
    Sigma = K_nn - tf.matmul(K_n_od, tf.matmul(K_odod_inv, K_n_od, transpose_b=True))

    # build random variable
    return ed.MultivariateNormalTriL(loc=tf.squeeze(Mu),
                                     scale_tril=tf.cholesky(Sigma),
                                     name=name)


def pred_cond_prior_v1(gp, gp_deriv,
                       X_new, X_obs, X_deriv, ls,
                       kernel_func_ff=gp.rbf,
                       kernel_func_df=rbf_grad_1d,
                       kernel_func_dd=rbf_hess_1d,
                       ridge_factor=1e-3,
                       name="gp_pred"):
    """Defines the prior of f_pred | f_obs, f_derive.

    The full joint likelihood of f_new, f_obs, f_deriv (denote as: f_n, f_o, f_d) is:

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
        gp: (ed.RandomVariable) f_obs corresponding to X_obs in
            training dataset, shape (N_train, )
        gp_deriv: (ed.RandomVariable) f_deriv, (N_deriv, )
        X_new: (tf.Tensor of float32) testing locations, (N_new, D)
        X_obs: (tf.Tensor of float32) training locations, (N_obs, D)
        X_deriv: (tf.Tensor of float32) training locations, (N_deriv, D)
        ls: (float32) Length scale parameter
        kernel_func_ff: (function) kernel function for k(x, x')
        kernel_func_df: (function) gradient kernel function dx k(x, x')
        kernel_func_dd: (function) hessian kernel function dxdx' k(x, x')
        ridge_factor: (float32) ridge factor to stabilize Cholesky decomposition.
        name: (str) name of the random variable.

    Returns:
         (tf.Tensor of float32) N_new x N_sample vectors of posterior
            predictive samples.

    Raises:
        (ValueError) If length scale parameter is not provided.
    """
    X_new = tf.convert_to_tensor(X_new, dtype=tf.float32)
    X_obs = tf.convert_to_tensor(X_obs, dtype=tf.float32)
    X_deriv = tf.convert_to_tensor(X_deriv, dtype=tf.float32)

    N_obs, _ = X_obs.shape.as_list()
    N_deriv, _ = X_deriv.shape.as_list()

    # compute matrix components
    K_nn = tf.stop_gradient(kernel_func_ff(X_new, ls=ls,
                                           ridge_factor=ridge_factor))

    K_no = tf.stop_gradient(kernel_func_ff(X_new, X_obs, ls=ls,
                                           ridge_factor=ridge_factor))
    K_dn = tf.stop_gradient(kernel_func_df(X_deriv, X_new, ls=ls,
                                           ridge_factor=ridge_factor))

    K_oo = tf.stop_gradient(kernel_func_ff(X_obs, ls=ls,
                                           ridge_factor=ridge_factor))
    K_do = tf.stop_gradient(kernel_func_df(X_deriv, X_obs, ls=ls,
                                           ridge_factor=ridge_factor))
    K_dd = tf.stop_gradient(kernel_func_dd(X_deriv, ls=ls,
                                           ridge_factor=ridge_factor))

    P_no, P_nd, Sigma_chol = inference_util.make_cond_gp_parameters(
        K_00=K_nn, K_11=K_oo, K_22=K_dd,
        K_01=K_no, K_20=K_dn, K_21=K_do,
        ridge_factor=ridge_factor)

    # compute conditional mean and variance.
    Mu = (tf.matmul(P_no, gp[:, tf.newaxis]) +
          tf.matmul(P_nd, gp_deriv[:, tf.newaxis]))

    # build random variable
    return ed.MultivariateNormalTriL(loc=tf.squeeze(Mu),
                                     scale_tril=Sigma_chol,
                                     name=name)


def compute_pred_cond_params(X_new, X_obs, X_deriv, ls,
                             kernel_func_ff=gp.rbf,
                             kernel_func_df=rbf_grad_1d,
                             kernel_func_dd=rbf_hess_1d,
                             ridge_factor_K=1e-3,
                             ridge_factor_Sigma=1e-3):
    """Computes model parameters for prior f_pred | f_obs, f_deriv.

    Args:
        X_new: (tf.Tensor of float32) testing locations, (N_new, D)
        X_obs: (tf.Tensor of float32) training locations, (N_obs, D)
        X_deriv: (tf.Tensor of float32) training locations, (N_deriv, D)
        ls: (float32) Length scale parameter
        kernel_func_ff: (function) kernel function for k(x, x')
        kernel_func_df: (function) gradient kernel function dx k(x, x')
        kernel_func_dd: (function) hessian kernel function dxdx' k(x, x')
        ridge_factor_K: (float32) ridge factor to stabilize Cholesky decomposition.
        ridge_factor_Sigma: (float32) ridge factor to stabilize Cholesky decomposition.

    Returns:
        P_01 (np.ndarray) Projection from f_obs to f_new
        P_02 (np.ndarray) Projection from f_deriv to f_new
        Sigma_chol (np.ndarray) Cholesky decomposition for Sigma
    """
    X_new = tf.convert_to_tensor(X_new, dtype=tf.float32)
    X_obs = tf.convert_to_tensor(X_obs, dtype=tf.float32)
    X_deriv = tf.convert_to_tensor(X_deriv, dtype=tf.float32)

    N_obs, _ = X_obs.shape.as_list()
    N_deriv, _ = X_deriv.shape.as_list()

    # compute matrix components
    K_nn = kernel_func_ff(X_new, ls=ls)

    K_no = kernel_func_ff(X_new, X_obs, ls=ls)
    K_dn = kernel_func_df(X_deriv, X_new, ls=ls)

    K_oo = kernel_func_ff(X_obs, ls=ls)
    K_do = kernel_func_df(X_deriv, X_obs, ls=ls)
    K_dd = kernel_func_dd(X_deriv, ls=ls)

    return inference_util.make_cond_gp_parameters(
        K_00=K_nn, K_11=K_oo, K_22=K_dd,
        K_01=K_no, K_20=K_dn, K_21=K_do,
        ridge_factor_K=ridge_factor_K,
        ridge_factor_Sigma=ridge_factor_Sigma
    )


def pred_cond_prior(gp, gp_deriv,
                    pred_cond_pars,
                    name="gp_pred"):
    """Defines the prior of f_pred | f_obs, f_derive.

    The full joint likelihood of f_new, f_obs, f_deriv (denote as: f_n, f_o, f_d) is:

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
        gp: (ed.RandomVariable) f_obs corresponding to X_obs in
            training dataset, shape (N_train, )
        gp_deriv: (ed.RandomVariable) f_deriv, (N_deriv, )
        pred_cond_pars: (list of np.ndarray) list of parameters
            (P_01, P_02, Sigma_chol) for pred_cond_prior. See
            inference_util.make_cond_gp_parameters
        name: (str) name of the random variable.

    Returns:
         (tf.Tensor of float32) N_new x N_sample vectors of posterior
            predictive samples.

    Raises:
        (ValueError) If length scale parameter is not provided.
    """
    P_no, P_nd, Sigma_chol = pred_cond_pars
    # compute conditional mean and variance.
    Mu = (tf.matmul(P_no, gp[:, tf.newaxis]) +
          tf.matmul(P_nd, gp_deriv[:, tf.newaxis]))

    # build random variable
    return ed.MultivariateNormalTriL(loc=tf.squeeze(Mu),
                                     scale_tril=Sigma_chol,
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
    X = tf.convert_to_tensor(X, dtype=tf.float32)

    if X_deriv is not None:
        X_deriv = tf.convert_to_tensor(X_deriv, dtype=tf.float32)
    else:
        X_deriv = X

    if ls is not None:
        ls = ed.Normal(loc=DEFAULT_PAR_SHIFT, scale=DEFAULT_PAR_SCALE, name='ls')
    else:
        ls = tf.convert_to_tensor(ls, dtype=tf.float32)

    gp_f = gp.prior(X, ls, kernel_func=gp.rbf,
                    ridge_factor=ridge_factor, name="gp_f")
    gp_f_deriv = deriv_prior(gp_f, X, X_deriv, ls,
                             kernel_func=gp.rbf,
                             grad_func=rbf_grad_1d, hess_func=rbf_hess_1d,
                             ridge_factor=ridge_factor, name="gp_f_deriv")

    sigma = ed.Normal(loc=DEFAULT_PAR_SHIFT, scale=DEFAULT_PAR_SCALE, name='sigma')

    y = ed.MultivariateNormalDiag(loc=gp_f,
                                  scale_identity_multiplier=tf.exp(sigma),
                                  name="y")

    return gp_f, gp_f_deriv, sigma, y, ls


def model_pred(X, X_deriv=None, X_pred=None, ls=1.,
               pred_cond_pars=None, ridge_factor=1e-3):
    """Defined Model for GP with Derivative and also predictions.

    Args:
        X: (np.Tensor of float32) input training features.
        with dimension (N, D).
        X_deriv: (tf.Tensor or None) Feature location to place derivative
            constraint on shape (N_deriv, D). If None, X_deriv=X
        X_pred: (tf.Tensor or None) Feature for prediction (N_pred, D).
        ls: (float32) length scale parameter.
        pred_cond_pars: (list of np.ndarray) list of parameters
            (P_01, P_02, Sigma_chol) for pred_cond_prior. See
            inference_util.make_cond_gp_parameters
        ridge_factor: (float32) ridge factor to stabilize Cholesky decomposition.

    Returns:
         (tf.Tensors of float32) model parameters.
    """
    X = tf.convert_to_tensor(X, dtype=tf.float32)

    if X_deriv is not None:
        X_deriv = tf.convert_to_tensor(X_deriv, dtype=tf.float32)
    else:
        X_deriv = X

    if X_pred is not None:
        X_pred = tf.convert_to_tensor(X_pred, dtype=tf.float32)
    else:
        X_pred = X

    if ls is not None:
        ls = ed.Normal(loc=DEFAULT_PAR_SHIFT, scale=DEFAULT_PAR_SCALE, name='ls')
    else:
        ls = tf.convert_to_tensor(ls, dtype=tf.float32)

    gp_f = gp.prior(X, ls, kernel_func=gp.rbf,
                    ridge_factor=ridge_factor, name="gp_f")
    gp_f_deriv = deriv_prior(gp_f, X, X_deriv, ls,
                             kernel_func=gp.rbf,
                             grad_func=rbf_grad_1d,
                             hess_func=rbf_hess_1d,
                             ridge_factor=ridge_factor, name="gp_f_deriv")
    gp_f_pred = pred_cond_prior(gp_f, gp_f_deriv,
                                pred_cond_pars=pred_cond_pars,
                                name="gp_f_pred")

    sigma = ed.Normal(loc=DEFAULT_PAR_SHIFT, scale=DEFAULT_PAR_SCALE, name='sigma')

    y = ed.MultivariateNormalDiag(loc=gp_f,
                                  scale_identity_multiplier=tf.exp(sigma),
                                  name="y")

    return gp_f, gp_f_pred, gp_f_deriv, sigma, y, ls


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Model likelihood function for MCMC and VI """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def make_log_likelihood_function(X_train, X_deriv,
                                 y_train, ls=None,
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
            f_deriv. For detail, see [2].
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
            loc=DEFAULT_CDF_CENTER, scale=deriv_prior_scale).log_cdf(
            model_var_kwargs["gp_f_deriv"])

        if cdf_constraint:
            # constraint 1: f > 0
            log_lkhd_constraint_ge_zero = tfd.Normal(
                loc=DEFAULT_CDF_CENTER, scale=deriv_prior_scale).log_cdf(
                model_var_kwargs["gp_f"])

            # constraint 2: f < 1
            log_lkhd_constraint_le_one = tfd.Normal(
                loc=DEFAULT_CDF_CENTER, scale=deriv_prior_scale).log_cdf(
                1 - model_var_kwargs["gp_f"])

            log_lkhd_constraint = tf.concat([
                log_lkhd_constraint,
                log_lkhd_constraint_ge_zero,
                log_lkhd_constraint_le_one,
            ], axis=0)

        log_joint_rest = log_joint(
            X=X_train, X_deriv=X_deriv, y=y_train,
            ridge_factor=ridge_factor,
            **model_var_kwargs)
        return tf.reduce_mean(log_lkhd_constraint) + log_joint_rest

    return target_log_prob_fn


def make_log_likelihood_function_with_pred(
        X_train, X_deriv, X_pred,
        y_train, ls=None, pred_cond_pars=None,
        ridge_factor=5e-3,
        deriv_prior_scale=1e-3,
        cdf_constraint=False):
    """Makes log joint likelihood function for monotonic GP regression.

    To be used for MCMC sampling.

    Args:
        X_train: (tf.Tensor) Training samples, shape (N_obs, D)
        X_deriv: (tf.Tensor) Locations of derivative constraints,
            shape (N_deriv, D)
        X_pred: (tf.Tensor or None) Feature for prediction (N_pred, D).
        y_train: (tf.Tensor) Training labels, shape (N_obs, )
        ls: (tf.Tensor or None) Value of length scale parameter,
            if None then will be added to the likelihood function.
        pred_cond_pars: (list of np.ndarray) list of parameters
            (P_01, P_02, Sigma_chol) for pred_cond_prior. See
            inference_util.make_cond_gp_parameters
        ridge_factor: (float32) ridge factor to stabilize Cholesky decomposition.
        deriv_prior_scale: (float32) Value for Prior scale parameter for
            probit likelihood function imposing positivity contraint on
            f_deriv. For detail, see [2].
        cdf_constraint: (bool) Whether to impose cdf constraint
            (i.e. f < 0 and f > 1).

    Returns:
        (function) A log-joint probability function.
            Its inputs are `model`'s parameters, and output the model's
            un-normalized log likelihood (a scalar tf.Tensor).

    """
    model_var_names = ["gp_f", "gp_f_pred", "gp_f_deriv", "sigma"]

    if ls is None:
        model_var_names += ["ls"]

    log_joint = ed.make_log_joint_fn(model_pred)

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
            loc=np.asarray(0., np.float32),
            scale=deriv_prior_scale).log_cdf(
            model_var_kwargs["gp_f_deriv"])

        if cdf_constraint:
            # constraint 1: f > 0
            log_lkhd_constraint_ge_zero = tfd.Normal(
                loc=np.asarray(0., np.float32),
                scale=deriv_prior_scale).log_cdf(
                model_var_kwargs["gp_f"])
            log_lkhd_constraint_ge_zero_pred = tfd.Normal(
                loc=np.asarray(0., np.float32),
                scale=deriv_prior_scale).log_cdf(
                model_var_kwargs["gp_f_pred"])

            # constraint 2: f < 1
            log_lkhd_constraint_le_one = tfd.Normal(
                loc=np.asarray(0., np.float32),
                scale=deriv_prior_scale).log_cdf(
                1 - model_var_kwargs["gp_f"])
            log_lkhd_constraint_le_one_pred = tfd.Normal(
                loc=np.asarray(0., np.float32),
                scale=deriv_prior_scale).log_cdf(
                1 - model_var_kwargs["gp_f_pred"])

            log_lkhd_constraint = tf.concat([
                log_lkhd_constraint,
                log_lkhd_constraint_ge_zero,
                log_lkhd_constraint_ge_zero_pred,
                log_lkhd_constraint_le_one,
                log_lkhd_constraint_le_one_pred
            ], axis=0)

        log_joint_rest = log_joint(
            X=X_train, X_deriv=X_deriv, X_pred=X_pred,
            y=y_train,
            ridge_factor=ridge_factor,
            pred_cond_pars=pred_cond_pars,
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
            f_deriv. For detail, see [2].
        cdf_constraint: (bool) Whether to impose cdf constraint
            (i.e. f > 0 and f < 1).

    Returns:
        (tf.Tensor) A scalar tf.Tensor corresponding to model likelihood,
            with dtype float32
    """
    # probit likelihood for derivative constraint
    log_lkhd_constraint = tfd.Normal(loc=DEFAULT_CDF_CENTER,
                                     scale=deriv_prior_scale).log_cdf(gp_f_deriv)
    if cdf_constraint:
        # constraint 1: f > 0
        log_lkhd_constraint_ge_zero = tfd.Normal(
            loc=DEFAULT_CDF_CENTER,
            scale=deriv_prior_scale).log_cdf(gp_f)
        # constraint 2: f < 1
        log_lkhd_constraint_le_one = tfd.Normal(
            loc=DEFAULT_CDF_CENTER,
            scale=deriv_prior_scale).log_cdf(1 - gp_f)

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

    The full joint likelihood of f_new (f_0), f_obs (f_1), f_deriv (f_2) is:

                    [[f_0]
                     [f_1]  ~ N(0, K)
                     [f_2]]

    where (denote K'/K'' the gradient/hessian kernel matrix)

                    [K_00   K_01    K'_02]
            K    =  [K_10   K_11    K'_12]
                    [K'_20  K'_21   K''_22]

    Therefore, if we denote
        K_0_12 = [K_01, K'_02]      and     K_1212 =    [K_11    K'_12]
                                                        [K'_21   K''_22]

    then
        f_n ~ N(Mu, Sigma), where (note * is matrix product here):

        Mu = K_n_od * inv(K_odod) * [f_o, f_d]^T
        Sigma = K_nn - K_n_od * inv(K_odod) * K_n_od^T

    Args:
        X_new: (tf.Tensor of float32) testing locations, (N_new, D)
        X_obs: (tf.Tensor of float32) training locations, (N_obs, D)
        X_deriv: (tf.Tensor of float32) training locations, (N_deriv, D)
        f_sample: (tf.Tensor of float32) Samples for f in training set,
            (N_train, N_sample)
        f_deriv_sample: (tf.Tensor of float32) Samples for f_deriv,
            (N_deriv, N_sample)
        kernel_func_ff: (function) kernel function for k(x, x')
        kernel_func_df: (function) gradient kernel function dx k(x, x')
        kernel_func_dd: (function) hessian kernel function dxdx' k(x, x')
        ls: (float32) Length scale parameter
        ridge_factor: (float32) ridge factor to stabilize Cholesky decomposition.

    Returns:
         (tf.Tensor of float32) N_new x N_sample vectors of posterior
            predictive samples.

    Raises:
        (ValueError) If length scale parameter is not provided.
    """
    X_new = tf.convert_to_tensor(X_new, dtype=tf.float32)
    X_obs = tf.convert_to_tensor(X_obs, dtype=tf.float32)
    X_deriv = tf.convert_to_tensor(X_deriv, dtype=tf.float32)

    f_sample = tf.convert_to_tensor(f_sample, dtype=tf.float32)
    f_deriv_sample = tf.convert_to_tensor(f_deriv_sample, dtype=tf.float32)

    N_new, _ = X_new.shape.as_list()
    N_obs, _ = X_obs.shape.as_list()
    N_deriv, _ = X_deriv.shape.as_list()

    _, N_sample = f_sample.shape.as_list()
    _, N_sample_2 = f_deriv_sample.shape.as_list()

    if N_sample != N_sample_2:
        raise ValueError('Sample size for f_sample ({}) and '
                         'f_deriv_sample ({}) must be same.'.format(N_sample, N_sample_2))

    if not ls:
        raise ValueError('Length scale parameter ("ls") must be provided.')

    # compute matrix components
    K_nn = kernel_func_ff(X_new, ls=ls)

    K_no = kernel_func_ff(X_new, X_obs, ls=ls)
    K_dn = kernel_func_df(X_deriv, X_new, ls=ls)

    K_oo = kernel_func_ff(X_obs, ls=ls)
    K_do = kernel_func_df(X_deriv, X_obs, ls=ls)
    K_dd = kernel_func_dd(X_deriv, ls=ls)

    K_oo_inv_od = tf.linalg.solve(K_oo, tf.transpose(K_do))
    K_dd_inv_do = tf.linalg.solve(K_dd, K_do)

    # assemble projection matrix
    K_nd_o = tf.transpose(K_dn) - tf.matmul(K_no, K_oo_inv_od)
    K_dd_o = K_dd - tf.matmul(K_do, K_oo_inv_od)
    K_no_d = K_no - tf.matmul(K_dn, K_dd_inv_do, transpose_a=True)
    K_oo_d = K_oo - tf.matmul(K_do, K_dd_inv_do, transpose_a=True)

    K_oo_d_inv = tf.matrix_inverse(K_oo_d)
    K_dd_o_inv = tf.matrix_inverse(
        K_dd_o + ridge_factor * tf.eye(N_deriv, dtype=tf.float32))

    P_no = tf.matmul(K_no_d, K_oo_d_inv)
    P_nd = tf.matmul(K_nd_o, K_dd_o_inv)

    # compute conditional mean and variance.
    mu_sample = tf.matmul(P_no, f_sample) + tf.matmul(P_nd, f_deriv_sample)
    Sigma = (K_nn -
             tf.matmul(P_no, K_no, transpose_b=True) + tf.matmul(P_nd, K_dn))

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
