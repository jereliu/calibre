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

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
from tensorflow.python.ops.distributions.util import fill_triangular

from scipy import stats
import gpflowSlim as gpf

tfd = tfp.distributions

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Helper functions """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def square_dist(X, X2=None, ls=1.):
    """Computes Square distance between two sets of features.

    Taken from GPflow.kernels.Stationary.

    :param X: (np.ndarray) First set of features of dim N x D.
    :param X2: (np.ndarray or None) Second set of features of dim N2 x D.
    :param ls: (float) value for length scale
    :return: (tf.Tensor) A N_new x N tensor for ||x-x'||^2
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


def rbf(X, X2=None, ls=1.):
    """Defines RBF kernel.

    :param X: (np.ndarray) First set of features of dim N x D.
    :param X2: (np.ndarray or None) Second set of features of dim N2 x D.
    :param ls: (float) value for length scale
    :return: (tf.Tensor) A N x N2 tensor for exp(-||x-x'||^2/2*ls)
    """
    return tf.exp(-square_dist(X, X2, ls=ls) / 2)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Main model definition """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def model(X, ls=1., ridge_factor=1e-4):
    """Defines the Gaussian Process Model.

    :param X: (np.ndarray of float32) input training features.
        with dimension (N, D).
    :param ls: (float32) length scale parameter.
    :param ridge_factor: (float32) small ridge factor to stabilize Cholesky decomposition
    :return: (tf.Tensors of float32) model parameters.
    """
    N = X.shape[0]
    K_mat = rbf(X, ls=ls) + ridge_factor * tf.eye(N)

    gp_f = ed.MultivariateNormalTriL(loc=tf.zeros(N),
                                     scale_tril=tf.cholesky(K_mat),
                                     name="gp_f")
    sigma = ed.Normal(loc=-5., scale=1., name='sigma')
    y = ed.MultivariateNormalDiag(loc=gp_f,
                                  scale_identity_multiplier=tf.exp(sigma),
                                  name="y")

    return gp_f, sigma, y


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Predictive Sampling functions """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def predmean(X_new, X, f_sample, ls, kernfunc=rbf, ridge_factor=1e-3):
    """Sample posterior mean for f^*.

    Posterior for f_new is conditionally independent from other parameters
        in the model, therefore it's conditional posterior mean
        can be obtained by sampling from the posterior conditional f^* | f:

    In particular, we care about posterior predictive mean, i.e.
        E(f^*|f) =  K(X^*, X)K(X, X)^{-1}f

    :param X_new: (np.ndarray of float) testing locations, N_new x D
    :param X: (np.ndarray of float) training locations, N x D
    :param f_sample: (np.ndarray of float) M samples of posterior GP sample, N x M
    :param ls: (float) training lengthscale
    :param kern_func: (function) kernel function.
    :param ridge_factor: (float32) small ridge factor to stabilize Cholesky decomposition.
    :return: (np.ndarray) N_new x M vectors of posterior predictive mean samples
    """
    Kx = kernfunc(X, X_new, ls=ls)
    K = kernfunc(X, ls=ls)
    # add ridge factor to stabilize inversion.
    K_inv_f = tf.matrix_solve(K + ridge_factor * tf.eye(tf.shape(X)[0]),
                              f_sample)
    return tf.matmul(Kx, K_inv_f, transpose_a=True)


def predsample(X_new, X, f_sample, ls, kernfunc=rbf, ridge_factor=1e-3):
    """Sample posterior predictive distribution.

    Sample posterior conditional from f^* | f ~ MVN, where:

        E(f*|f) = K(X*, X)K(X, X)^{-1}f
        Var(f*|f) = K(X*, X*) - K(X*, X)K(X, X)^{-1}K(X, X*)

    :param X_new: (np.ndarray of float) testing locations, N_new x D
    :param X: (np.ndarray of float) training locations, N x D
    :param f: (np.ndarray of float) M samples of posterior GP sample, N x M
    :param ls: (float) training lengthscale
    :param kern_func: (function) kernel function.
    :param ridge_factor: (float32) small ridge factor to stabilize Cholesky decomposition.
    :return: (np.ndarray) N_new x M vectors of posterior predictive mean samples
    """
    N_new, _ = X_new.shape
    N, M = f_sample.shape

    # compute basic components
    Kxx = kernfunc(X_new, X_new, ls=ls)
    Kx = kernfunc(X, X_new, ls=ls)
    K = kernfunc(X, ls=ls)
    K_inv = tf.matrix_inverse(K + ridge_factor * tf.eye(N))

    # compute conditional mean and variance.
    mu_sample = tf.matmul(Kx, tf.matmul(K_inv, f_sample), transpose_a=True)
    Sigma = Kxx - tf.matmul(Kx, tf.matmul(K_inv, Kx), transpose_a=True)

    # sample
    with tf.Session() as sess:
        cond_means, cond_cov = sess.run([mu_sample, Sigma])

    f_new_centered = np.random.multivariate_normal(
        mean=[0] * N_new, cov=cond_cov, size=M).T
    return f_new_centered + cond_means


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Variational family I: Mean field """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def variational_meanfield(X):
    """Defines the mean-field variational family for GPR.

    :param X: (np.ndarray of float32) input training features.
        with dimension (N, D).
    :return:
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


def variational_meanfield_sample(n_sample, qf_mean, qf_sdev):
    """Generates f samples from GPR mean-field variational family.

    :param n_sample: (int) number of samples to draw
    :param qf_mean: (tf.Tensor of float32) mean parameters for
        variational family
    :param qf_sdev: (tf.Tensor of float32) standard deviation
        parameters for variational family
    :return: (np.ndarray) sampled values.
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


def variational_sgp(X, Z, ls=1., kern_func=rbf, ridge_factor=1e-3):
    """Defines the mean-field variational family for GPR.

    :param X: (np.ndarray of float32) input training features, with dimension (Nx, D).
    :param Z: (np.ndarray of float32) inducing points, with dimension (Nz, D).
    :param ls: (float32) length scale parameter.
    :param kern_func: (function) kernel function.
    :param ridge_factor: (float32) small ridge factor to stabilize Cholesky decomposition
    :return:
        q_f, q_sig: (ed.RandomVariable) variational family.
        q_f_mean, q_f_sdev: (tf.Variable) variational parameters for q_f
    """
    Nx, Nz = X.shape[0], Z.shape[0]

    # compute matrix constants
    Kxx = kern_func(X, ls=ls)
    Kxz = kern_func(X, Z, ls=ls)
    Kzz = kern_func(Z, ls=ls) + ridge_factor * tf.eye(Nz)

    # compute null covariance matrix using Cholesky decomposition
    Kzz_chol_inv = tf.matrix_inverse(tf.cholesky(Kzz))
    Kzz_inv = tf.matmul(Kzz_chol_inv, Kzz_chol_inv, transpose_a=True)

    Kxz_Kzz_chol_inv = tf.matmul(Kxz, Kzz_chol_inv, transpose_b=True)
    Kxz_Kzz_inv = tf.matmul(Kxz, Kzz_inv)
    Sigma_pre = Kxx - tf.matmul(Kxz_Kzz_chol_inv, Kxz_Kzz_chol_inv, transpose_b=True)

    # define free parameters
    m = tf.get_variable(shape=[Nz], name='qf_m')
    s = tf.get_variable(shape=[Nz * (Nz + 1) / 2],
                        # initializer=tf.zeros_initializer(),
                        name='qf_s')
    L = fill_triangular(s, name='qf_chol')
    S = tf.matmul(L, L, transpose_b=True)

    q_sig_mean = tf.get_variable(shape=[], name='q_sig_mean')
    q_sig_sdev = tf.exp(tf.get_variable(shape=[], name='q_sig_sdev'))

    # compute sparse gp variational parameter
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


def variational_sgp_sample(n_sample, qf_mean, qf_cov):
    """Generates f samples from GPR mean-field variational family.

    :param n_sample: (int) number of samples to draw
    :param qf_mean: (tf.Tensor of float32) mean parameters for
        variational family
    :param qf_cov: (tf.Tensor of float32) covariance for
        parameters for variational family
    :return: (np.ndarray) sampled values.
    """

    """Generates f samples from GPR mean-field variational family."""
    q_f = tfd.MultivariateNormalFullCovariance(loc=qf_mean,
                                               covariance_matrix=qf_cov,
                                               name='q_f')
    return q_f.sample(n_sample)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Predictive functions, GPflow Implementation """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def fit_gpflow(X_train, y_train, X_test, y_test,
               kern_func=None, tune_kern_param=False, n_iter=20000,
               **kwargs
               ):
    """Fits GP regression using GPflow

    :param X_train: (np.ndarray of float32) Training data (N_train, D).
    :param y_train: (np.ndarray of float32) Training labels (N_train, D).
    :param X_test: (np.ndarray of float32) Testintg features (N_test, D).
    :param y_test: (np.ndarray of float32) Testing labels (N_test, D).
    :param kern_func: (gpflow.kernels) GPflow kernel function.
    :param tune_kern_param: (bool) Whether to tune kernel parameters.
    :param n_iter: (int) number of optimization iterations.
    :param kwargs: Additional arguments passed to kern_func.
    :return:
        mu, var: (np.ndarray) Posterior predictive mean/variance.
        par_val: (list of np.ndarray) List of model parameter values
        m: (gpflow.models.gpr) gpflow model object.
        k: (gpflow.kernels) gpflow kernel object.
    """
    if y_train.ndim == 1:
        y_train = np.expand_dims(y_train, 1)


    # define computation graph
    gpr_graph = tf.Graph()
    with gpr_graph.as_default():

        # define model
        if not kern_func:
            k = gpf.kernels.RBF(input_dim=X_train.shape[1], ARD=True)
        else:
            k = kern_func(input_dim=X_train.shape[1],
                          train_kernel_param=tune_kern_param,
                          **kwargs)

        m = gpf.models.GPR(X_train, y_train, kern=k)

        # define optimization
        objective = m.objective
        param_dict = {par.name: par.value for par in m.parameters}
        optimizer = tf.train.AdamOptimizer(1e-3)
        train_op = optimizer.minimize(objective)

        # define prediction
        pred_mu, pred_cov = m.predict_f(X_test)

        init_op = tf.global_variables_initializer()

        gpr_graph.finalize()


    # execute training
    with tf.Session(graph=gpr_graph) as sess:
        sess.run(init_op)
        for iter in range(n_iter):
            _, obj = sess.run([train_op, objective])

            if iter % 1000 == 0:
                print('Iter {}: Loss = {}'.format(iter, obj))

                # evaluate
                mu, var, par_dict = sess.run([pred_mu, pred_cov, param_dict])
                mu, var = mu.squeeze(), var.squeeze()
                rmse = np.mean((mu - y_test) ** 2) ** .5

                log_likelihood = np.mean(
                    np.log(stats.norm.pdf(y_test.squeeze(),
                                          loc=mu, scale=var ** 0.5)))
                print('test rmse = {}'.format(rmse))
        sess.close()

        print('tset ll = {}'.format(log_likelihood))

    return mu, var, par_dict, m, k
