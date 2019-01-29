"""
Gaussian Process regression with expit link function and monotonic constraint.


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

from calibre.util.matrix import replicate_along_zero_axis
from calibre.model.gaussian_process import rbf

import calibre.util.distribution as dist_util
import calibre.util.inference as inference_util

tfd = tfp.distributions

# TODO(jereliu): fix sgpr and dgpr vi family to also have expit link

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Main model definition """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def model(X, log_ls=0., ridge_factor=1e-4, sample_ls=False):
    """Defines the Gaussian Process Model.

    Args:
        X: (np.ndarray of float32) input training features.
        with dimension (N, D).
        log_ls: (float32) length scale parameter in log scale.
        ridge_factor: (float32) ridge factor to stabilize Cholesky decomposition.
        sample_ls: (bool) Whether sample ls parameter.

    Returns:
         (tf.Tensors of float32) model parameters.
    """
    X = tf.convert_to_tensor(X)

    N = X.shape.as_list()[0]

    # specify kernel matrix
    if sample_ls:
        log_ls = ed.Normal(loc=-5., scale=1., name='ls')

    K_mat = rbf(X, ls=tf.exp(log_ls), ridge_factor=ridge_factor)

    # specify model parameters
    gp_f = ed.MultivariateNormalTriL(loc=tf.zeros(N),
                                     scale_tril=tf.cholesky(K_mat),
                                     name="gp_f")

    y = ed.MultivariateNormalDiag(loc=gp_f,
                                  scale_identity_multiplier=.001,
                                  name="y")

    return y, gp_f, log_ls


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Variational family I: Mean field """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def variational_mfvi(X, mfvi_mixture=False, n_mixture=1):
    """Defines the mean-field variational family for GPR.

    Args:
        X: (np.ndarray of float32) input training features, shape (N, D).
        mfvi_mixture: (float32) Whether to output variational family with a
            mixture of MFVI.
        n_mixture: (int) Number of MFVI mixture component to add.

    Returns:
        q_f, q_sig: (ed.RandomVariable) variational family.
        q_f_mean, q_f_sdev: (tf.Variable) variational parameters for q_f
    """
    N, D = X.shape

    # define variational parameters
    qf_mean = tf.get_variable(shape=[N], name='qf_mean')
    qf_sdev = tf.exp(tf.get_variable(shape=[N], name='qf_sdev'))

    # define variational family
    mixture_par_list = []
    if mfvi_mixture:
        gp_dist = tfd.MultivariateNormalDiag(loc=qf_mean, scale_diag=qf_sdev,
                                             name='q_f')
        q_f, mixture_par_list = inference_util.make_mfvi_sgp_mixture_family(
            n_mixture=n_mixture, N=N, gp_dist=gp_dist, name='q_f')
    else:
        q_f = ed.MultivariateNormalDiag(loc=qf_mean,
                                        scale_diag=qf_sdev, name='q_f')

    return q_f, qf_mean, qf_sdev, mixture_par_list


def variational_mfvi_sample(n_sample, qf_mean, qf_vcov,
                            mfvi_mixture=False, mixture_par_list=None):
    """Generates f samples from GPR mean-field variational family.

    Args:
        n_sample: (int) number of samples to draw
        qf_mean: (tf.Tensor of float32) mean parameters for
        variational family
        qf_vcov: (tf.Tensor of float32) standard deviation
        parameters for variational family
        mfvi_mixture: (bool) Whether to sample from a MFVI mixture
        mixture_par_list: (list of np.ndarray) List of mixture distribution
            parameters, containing:

                mixture_logits: mixture logit for sgp-mfvi_mix family
                mixture_logits_mfvi_mix: mixture logit within mfvi_mix family
                qf_mean_mfvi, qf_sdev_mfvi:
                    variational parameters for mfvi_mix family


    Returns:
         (np.ndarray) sampled values.
    """

    q_f = tfd.MultivariateNormalDiag(loc=qf_mean, scale_diag=qf_vcov,
                                     name='q_f')
    q_f_sample = q_f.sample(n_sample)
    if mfvi_mixture:
        (mixture_logits, mixture_logits_mfvi_mix,
         mean_mfvi_mix, sdev_mfvi_mix) = mixture_par_list

        q_f_sample_mfvi = inference_util.sample_mfvi_mixture_family(
            N_sample=n_sample,
            mixture_logits=mixture_logits_mfvi_mix,
            mean_mfvi_mix=mean_mfvi_mix,
            sdev_mfvi_mix=sdev_mfvi_mix, )

        mix_prob = tf.nn.softmax(mixture_logits)

        q_f_sample = tf.tensordot(
            tf.stack([q_f_sample_mfvi, q_f_sample], axis=-1), mix_prob,
            axes=[[-1], [0]])

    return q_f_sample


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


def variational_sgpr(X, Z, ls=1., kern_func=rbf, ridge_factor=1e-3,
                     mfvi_mixture=False, n_mixture=1):
    """Defines the mean-field variational family for GPR.

    Args:
        X: (np.ndarray of float32) input training features, with dimension (Nx, D).
        Z: (np.ndarray of float32) inducing points, with dimension (Nz, D).
        ls: (float32) length scale parameter.
        kern_func: (function) kernel function.
        ridge_factor: (float32) small ridge factor to stabilize Cholesky decomposition
        mfvi_mixture: (float32) Whether to output variational family with a
            mixture of MFVI.
        n_mixture: (int) Number of MFVI mixture component to add.

    Returns:
        q_f, q_sig: (ed.RandomVariable) variational family.
        q_f_mean, q_f_sdev: (tf.Variable) variational parameters for q_f
        mixture_par_list: (list of tf.Variable) variational parameters for
            MFVI mixture ('mixture_logits', 'mixture_logits_mfvi_mix',
            'mean_mfvi', 'sdev_mfvi') if mfvi_mixture=True, else [].
    """
    X = tf.convert_to_tensor(X)
    Z = tf.convert_to_tensor(Z)

    Nx, Nz = X.shape.as_list()[0], Z.shape.as_list()[0]

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
              ridge_factor * tf.eye(Nx, dtype=tf.float32)
              )

    # define variational family
    mixture_par_list = []
    if mfvi_mixture:
        gp_dist = tfd.MultivariateNormalFullCovariance(loc=qf_mean,
                                                       covariance_matrix=qf_cov)
        q_f, mixture_par_list = inference_util.make_mfvi_sgp_mixture_family(
            n_mixture=n_mixture, N=Nx,
            gp_dist=gp_dist, name='q_f')
    else:
        q_f = ed.MultivariateNormalFullCovariance(loc=qf_mean,
                                                  covariance_matrix=qf_cov,
                                                  name='q_f')

    return q_f, qf_mean, qf_cov, mixture_par_list


def variational_sgpr_sample(n_sample, qf_mean, qf_vcov,
                            mfvi_mixture=False, mixture_par_list=None):
    """Generates f samples from GPR mean-field variational family.

    Args:
        n_sample: (int) number of samples to draw
        qf_mean: (tf.Tensor of float32) mean parameters for
            variational family
        qf_vcov: (tf.Tensor of float32) covariance for parameters for
            variational family
        mfvi_mixture: (bool) Whether to sample from a MFVI-SGP mixture
        mixture_par_list: (list of np.ndarray) List of mixture distribution
            parameters, containing:

                mixture_logits: mixture logit for sgp-mfvi_mix family
                mixture_logits_mfvi_mix: mixture logit within mfvi_mix family
                qf_mean_mfvi, qf_sdev_mfvi:
                    variational parameters for mfvi_mix family

    Returns:
        (np.ndarray) sampled values.
    """

    """Generates f samples from GPR mean-field variational family."""
    q_f = tfd.MultivariateNormalFullCovariance(loc=qf_mean,
                                               covariance_matrix=qf_vcov,
                                               name='q_f')
    q_f_sample = q_f.sample(n_sample)

    if mfvi_mixture:
        (mixture_logits, mixture_logits_mfvi_mix,
         mean_mfvi_mix, sdev_mfvi_mix) = mixture_par_list

        q_f_sample_mfvi = inference_util.sample_mfvi_mixture_family(
            N_sample=n_sample,
            mixture_logits=mixture_logits_mfvi_mix,
            mean_mfvi_mix=mean_mfvi_mix,
            sdev_mfvi_mix=sdev_mfvi_mix, )

        mix_prob = tf.nn.softmax(mixture_logits)

        q_f_sample = tf.tensordot(
            tf.stack([q_f_sample_mfvi, q_f_sample], axis=-1), mix_prob,
            axes=[[-1], [0]])

    return q_f_sample


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Variational family III: Decoupled Gaussian Process """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""Implements the Decoupled GP (DGP) VI method by [2].

Select a set of inducing points Zm and Zs, then:

Original posterior:
p(Y, F, U) = p(Y|F) p(F|U) p(U), where:
    p(Y|F) ~ MVN(Y| Mu =    F, 
                    Sigma = s^2 * I)
    p(F|U) ~ MVN(F| Mu =    Kxm Kmm^{-1} U 
                    Sigma = Kxx - Kxs Kss^{-1} Kxs^T )
    p(U)   ~ MVN(U| Mu = 0, Sigma = Kss)

Variational posterior:
    q(Y)   = p(Y|F)
    q(F|U) = p(F|U)
    q(U|m, S) ~ DGP

Consequently, q(F) becomes 
    q(F|m, S) ~ MVN(F| Mu =     Kxm m
                       Sigma =  Kxx - Kxs (Kss + S^{-1})^{-1} Kxs^T)
                       
In practice, to make the problem unconstrained, we model S = LL^T.
Then 

    (Kss + S^{-1})^{-1} = L H^{-1} L^T,

where H = I + L^T Kss L.
"""


def variational_dgpr(X, Zm, Zs, ls=1., kern_func=rbf, ridge_factor=1e-3,
                     mfvi_mixture=False, n_mixture=1):
    """Defines the mean-field variational family for GPR.

    Args:
        X: (np.ndarray of float32) input training features, with dimension (Nx, D).
        Zm: (np.ndarray of float32) inducing points for mean, shape (Nm, D).
        Zs: (np.ndarray of float32) inducing points for covar, shape (Ns, D).
        ls: (float32) length scale parameter.
        kern_func: (function) kernel function.
        ridge_factor: (float32) small ridge factor to stabilize Cholesky decomposition
        mfvi_mixture: (float32) Whether to output variational family with a
            mixture of MFVI.
        n_mixture: (int) Number of MFVI mixture component to add.

    Returns:
        q_f, q_sig: (ed.RandomVariable) variational family.
        q_f_mean, q_f_sdev: (tf.Variable) variational parameters for q_f
    """
    X = tf.convert_to_tensor(X)
    Zm = tf.convert_to_tensor(Zm)
    Zs = tf.convert_to_tensor(Zs)

    Nx, Nm, Ns = X.shape.as_list()[0], Zm.shape.as_list()[0], Zs.shape.as_list()[0]

    # 1. Prepare constants
    # compute matrix constants
    Kxx = kern_func(X, ls=ls)
    Kmm = kern_func(Zm, ls=ls)
    Kxm = kern_func(X, Zm, ls=ls)
    Kxs = kern_func(X, Zs, ls=ls)
    Kss = kern_func(Zs, ls=ls, ridge_factor=ridge_factor)

    # 2. Define variational parameters
    # define free parameters (i.e. mean and full covariance of f_latent)
    m = tf.get_variable(shape=[Nm, 1], name='qf_m')
    s = tf.get_variable(shape=[Ns * (Ns + 1) / 2], name='qf_s')
    L = fill_triangular(s, name='qf_chol')

    # components for KL objective
    H = tf.eye(Ns) + tf.matmul(L, tf.matmul(Kss, L), transpose_a=True)
    cond_cov_inv = tf.matmul(L, tf.matrix_solve(H, tf.transpose(L)))

    func_norm_mm = tf.matmul(m, tf.matmul(Kmm, m), transpose_a=True)
    log_det_ss = tf.log(tf.matrix_determinant(H))
    cond_norm_ss = tf.reduce_sum(tf.multiply(Kss, cond_cov_inv))

    # compute sparse gp variational parameter (i.e. mean and covariance of P(f_obs | f_latent))
    qf_mean = tf.squeeze(tf.tensordot(Kxm, m, [[1], [0]]), name='qf_mean')
    qf_cov = (Kxx -
              tf.matmul(Kxs, tf.matmul(cond_cov_inv, Kxs, transpose_b=True)) +
              ridge_factor * tf.eye(Nx, dtype=tf.float32)
              )

    # define variational family
    mixture_par_list = []
    if mfvi_mixture:
        gp_dist = dist_util.VariationalGaussianProcessDecoupledDistribution(
            loc=qf_mean,
            covariance_matrix=qf_cov,
            func_norm_mm=func_norm_mm,
            log_det_ss=log_det_ss,
            cond_norm_ss=cond_norm_ss)

        q_f, mixture_par_list = inference_util.make_mfvi_sgp_mixture_family(
            n_mixture=n_mixture, N=Nx,
            gp_dist=gp_dist, name='q_f')
    else:
        q_f = dist_util.VariationalGaussianProcessDecoupled(loc=qf_mean,
                                                            covariance_matrix=qf_cov,
                                                            func_norm_mm=func_norm_mm,
                                                            log_det_ss=log_det_ss,
                                                            cond_norm_ss=cond_norm_ss,
                                                            name='q_f')
    return q_f, qf_mean, qf_cov, mixture_par_list


variational_dgpr_sample = variational_sgpr_sample
