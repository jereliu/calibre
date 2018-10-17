"""Monotonic GPR using MCMC and VI.


#### References

[1]:    Jaakko Riihimaki and Aki Vehtari. Gaussian processes with monotonicity information.
        _13th International Conference on Artificial Intelligence and Statistics (AISTATS)_
        2010. http://proceedings.mlr.press/v9/riihimaki10a/riihimaki10a.pdf
[2]:    Marco Lorenzi and Maurizio Filippone. Constraining the Dynamics of Deep Probabilistic Models.
        _35th International Conference on Machine Learning_, 2018.
        http://proceedings.mlr.press/v80/lorenzi18a.html
"""
# TODO(jereliu): Allow pseudo positions for gradient constraint
# TODO(jereliu): Try on real calibration data.

import os
import time
from importlib import reload

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
import gpflowSlim as gpf

# sys.path.extend([os.getcwd()])

from calibre.model import gaussian_process as gp
from calibre.model import gp_regression as gpr
from calibre.model import gp_regression_monotone as gpr_mono

import calibre.util.data as data_util
import calibre.util.visual as visual_util

from calibre.util.data import sin_curve_1d, cos_curve_1d, generate_1d_data
from calibre.util.inference import make_value_setter

import matplotlib.pyplot as plt

tfd = tfp.distributions

_SAVE_ADDR_PREFIX = "./result/gpr_mono"

DEFAULT_LS_VAL = 0.2
DEFAULT_DERIV_CDF_SCALE = 0.001

"""""""""""""""""""""""""""""""""
# 1. Generate data
"""""""""""""""""""""""""""""""""
N = 50

X_train, y_train = generate_1d_data(N=N, f=sin_curve_1d,
                                    noise_sd=0.03, seed=100,
                                    uniform_x=True)
X_train = np.expand_dims(X_train, 1).astype(np.float32)
y_train = y_train.astype(np.float32)
std_y_train = np.std(y_train)

X_test = np.expand_dims(np.linspace(-1, 2, 100), 1).astype(np.float32)
y_test = sin_curve_1d(X_test)

N, D = X_train.shape

#
plt.plot(np.linspace(-0.5, 1.5, 100),
         sin_curve_1d(np.linspace(-0.5, 1.5, 100)), c='black')
plt.plot(X_train.squeeze(), y_train.squeeze(),
         'o', c='red', markeredgecolor='black')
plt.close()

"""""""""""""""""""""""""""""""""
# 2. MCMC
"""""""""""""""""""""""""""""""""
"""2.1. sampler basic config"""
num_results = 5000
num_burnin_steps = 5000

# define mcmc computation graph
mcmc_graph = tf.Graph()
with mcmc_graph.as_default():
    # build likelihood by explicitly
    log_joint = ed.make_log_joint_fn(gpr_mono.model)


    # TODO(jereliu): move to gpr_mono
    def target_log_prob_fn(gp_f, gp_f_deriv, sigma):
        """Unnormalized target density as a function of states.
            with additional likelihood for
        """
        log_deriv_lkhd = tfd.Normal(
            loc=0., scale=DEFAULT_DERIV_CDF_SCALE).log_cdf(gp_f_deriv)
        log_joint_rest = log_joint(
            X_train, y=y_train,
            ls=DEFAULT_LS_VAL, ridge_factor=5e-3,
            gp_f=gp_f, gp_f_deriv=gp_f_deriv, sigma=sigma)
        return tf.reduce_sum(log_deriv_lkhd) + log_joint_rest


    # set up state container
    initial_state = [
        tf.random_normal([N], stddev=0.01, name='init_gp_func'),
        tf.random_normal([N], stddev=0.01, name='init_gp_derv'),
        tf.constant(0.1, name='init_sigma'),
    ]

    # set up HMC transition kernel
    step_size = tf.get_variable(
        name='step_size',
        initializer=1.,
        use_resource=True,  # For TFE compatibility.
        trainable=False)

    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        num_leapfrog_steps=3,
        step_size=step_size,
        step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy())

    # set up main sampler
    state, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=initial_state,
        kernel=hmc,
        parallel_iterations=10
    )

    gpf_sample, gpf_deriv_sample, sigma_sample, = state

    # set up init op
    init_op = tf.global_variables_initializer()

    mcmc_graph.finalize()

""" 2.2. execute sampling"""
with tf.Session(graph=mcmc_graph) as sess:
    init_op.run()
    [
        f_samples_val,
        f_deriv_sample_val,
        sigma_sample_val,
        is_accepted_,
    ] = sess.run(
        [
            gpf_sample,
            gpf_deriv_sample,
            sigma_sample,
            kernel_results.is_accepted,
        ])
    print('Acceptance Rate: {}'.format(np.mean(is_accepted_)))
    sess.close()

""" 2.3. prediction and visualization"""
# prediction
f_test_val = gp.sample_posterior_full(X_new=X_test, X=X_train,
                                      f_sample=f_samples_val.T,
                                      ls=DEFAULT_LS_VAL,
                                      kernel_func=gp.rbf)

df_test_val = gp.sample_posterior_full(X_new=X_test, X=X_train,
                                       f_sample=f_deriv_sample_val.T,
                                       ls=DEFAULT_LS_VAL,
                                       kernel_func=gpr_mono.rbf_hess_1d)

# visualize
mu = np.mean(f_test_val, axis=1)
mu_deriv = np.mean(df_test_val, axis=1)
cov = np.var(f_test_val, axis=1)
cov_deriv = np.var(df_test_val, axis=1)

visual_util.gpr_1d_visual(mu, cov,
                          X_train=X_train, y_train=y_train,
                          X_test=X_test, y_test=y_test,
                          title="RBF, Hamilton MC",
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX, "hmc_gpr.png"))

visual_util.gpr_1d_visual(mu_deriv, cov_deriv,
                          X_train=X_train, y_train=y_train,
                          X_test=X_test, y_test=y_test,
                          title="RBF Derivative, Hamilton MC",
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX, "hmc_gpr_deriv.png"),
                          add_reference=True)

"""""""""""""""""""""""""""""""""
# 3. Mean-field VI
"""""""""""""""""""""""""""""""""

""" 3.1. Set up the computational graph """
mfvi_graph = tf.Graph()

with mfvi_graph.as_default():
    # sample from variational family
    (q_f, q_f_deriv, q_sig,
     qf_mean, qf_sdev,
     qf_deriv_mean, qf_deriv_sdev) = gpr_mono.variational_mfvi(X=X_train,
                                                               ls=DEFAULT_LS_VAL)

    # compute the expected predictive log-likelihood
    with ed.tape() as model_tape:
        with ed.interception(make_value_setter(gp_f=q_f,
                                               gp_f_deriv=q_f_deriv,
                                               sigma=q_sig)):
            _, gp_f_deriv, _, y, _ = gpr_mono.model(X=X_train, ls=DEFAULT_LS_VAL)

    # add penalized likelihood
    # TODO(jereliu): move to gpr_mono
    log_lkhd_derv = tf.reduce_mean(tfd.Normal(
        loc=0., scale=DEFAULT_DERIV_CDF_SCALE).log_cdf(gp_f_deriv))
    log_lkhd_rest = y.distribution.log_prob(y_train)
    log_likelihood = log_lkhd_derv + log_lkhd_rest

    # compute the KL divergence
    kl = 0.
    for rv_name, variational_rv in [("gp_f", q_f), ("sigma", q_sig)]:
        kl += tf.reduce_sum(
            variational_rv.distribution.kl_divergence(
                model_tape[rv_name].distribution)
        )

    # define loss op: ELBO = E_q(p(x|z)) + KL(q || p)
    elbo = tf.reduce_mean(log_likelihood - kl)

    # define optimizer
    optimizer = tf.train.AdamOptimizer(5e-2)
    train_op = optimizer.minimize(-elbo)

    # define init op
    init_op = tf.global_variables_initializer()

    mfvi_graph.finalize()

""" 3.2. execute optimization """
max_steps = 50000  # number of training iterations

with tf.Session(graph=mfvi_graph) as sess:
    start_time = time.time()

    sess.run(init_op)
    for step in range(max_steps):
        start_time = time.time()
        _, elbo_value = sess.run([train_op, elbo])
        if step % 1000 == 0:
            duration = time.time() - start_time
            print("Step: {:>3d} Loss: {:.3f} ({:.3f} sec)".format(
                step, elbo_value, duration))
    (qf_mean_val, qf_sdev_val,
     qf_deriv_mean_val, qf_deriv_sdev_val) = sess.run([
        qf_mean, qf_sdev, qf_deriv_mean, qf_deriv_sdev])

    sess.close()

""" 3.3. prediction & visualization """
with tf.Session() as sess:
    f_samples = gpr_mono.variational_mfvi_sample(n_sample=5000,
                                                 qf_mean=qf_mean_val,
                                                 qf_sdev=qf_sdev_val)
    f_deriv_samples = gpr_mono.variational_mfvi_sample(n_sample=5000,
                                                       qf_mean=qf_deriv_mean_val,
                                                       qf_sdev=qf_deriv_sdev_val)
    f_samples_val, f_deriv_samples_val = sess.run([f_samples,
                                                   f_deriv_samples])
    sess.close()

# still use exact posterior predictive
f_test_val = gp.sample_posterior_full(X_new=X_test, X=X_train,
                                      f_sample=f_samples_val.T,
                                      ls=DEFAULT_LS_VAL,
                                      kernel_func=gp.rbf)

df_test_val = gp.sample_posterior_full(X_new=X_test, X=X_train,
                                       f_sample=f_deriv_sample_val.T,
                                       ls=DEFAULT_LS_VAL,
                                       kernel_func=gpr_mono.rbf_hess_1d)

# visualize
mu = np.mean(f_test_val, axis=1)
mu_deriv = np.mean(df_test_val, axis=1)
cov = np.var(f_test_val, axis=1)
cov_deriv = np.var(df_test_val, axis=1)

visual_util.gpr_1d_visual(mu, cov,
                          X_train=X_train, y_train=y_train,
                          X_test=X_test, y_test=y_test,
                          title="RBF, Mean-field VI",
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX, "mfvi_gpr.png"))

visual_util.gpr_1d_visual(mu_deriv, cov_deriv,
                          X_train=X_train, y_train=y_train,
                          X_test=X_test, y_test=y_test,
                          title="RBF Derivative, Mean-field VI",
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX, "mfvi_gpr_deriv.png"),
                          add_reference=True)

"""""""""""""""""""""""""""""""""
# 4. Sparse GP (Structured VI)
"""""""""""""""""""""""""""""""""

""" 4.1. Set up the computational graph """
sgp_graph = tf.Graph()
X_induce = np.expand_dims(np.linspace(np.min(X_train),
                                      np.max(X_train), 10), 1).astype(np.float32)

with sgp_graph.as_default():
    # sample from variational family
    (q_f, q_f_deriv, q_sig,
     qf_mean, qf_vcov,
     qf_deriv_mean, qf_deriv_sdev) = gpr_mono.variational_sgpr(X=X_train,
                                                               Z=X_induce,
                                                               ls=DEFAULT_LS_VAL)

    # compute the expected predictive log-likelihood
    with ed.tape() as model_tape:
        with ed.interception(make_value_setter(gp_f=q_f,
                                               gp_f_deriv=q_f_deriv,
                                               sigma=q_sig)):
            _, gp_f_deriv, _, y, _ = gpr_mono.model(X=X_train, ls=DEFAULT_LS_VAL)

    # add penalized likelihood
    # TODO(jereliu): move to gpr_mono
    log_lkhd_derv = tf.reduce_mean(tfd.Normal(
        loc=0., scale=DEFAULT_DERIV_CDF_SCALE).log_cdf(gp_f_deriv))
    log_lkhd_rest = y.distribution.log_prob(y_train)
    log_likelihood = log_lkhd_derv + log_lkhd_rest

    # compute the KL divergence
    kl = 0.
    for rv_name, variational_rv in [("gp_f", q_f), ("sigma", q_sig)]:
        kl += tf.reduce_sum(
            variational_rv.distribution.kl_divergence(
                model_tape[rv_name].distribution)
        )

    # define loss op: ELBO = E_q(p(x|z)) + KL(q || p)
    elbo = tf.reduce_mean(log_likelihood - kl)

    # define optimizer
    optimizer = tf.train.AdamOptimizer(1e-2)
    train_op = optimizer.minimize(-elbo)

    # define init op
    init_op = tf.global_variables_initializer()

    sgp_graph.finalize()

""" 4.2. execute optimization """
max_steps = 50000  # number of training iterations

with tf.Session(graph=sgp_graph) as sess:
    start_time = time.time()

    sess.run(init_op)
    for step in range(max_steps):
        start_time = time.time()
        _, elbo_value = sess.run([train_op, elbo])
        if step % 1000 == 0:
            duration = time.time() - start_time
            print("Step: {:>3d} Loss: {:.3f} ({:.3f} sec)".format(
                step, elbo_value, duration))
    (qf_mean_val, qf_vcov_val,
     qf_deriv_mean_val, qf_deriv_sdev_val) = sess.run([
        qf_mean, qf_vcov, qf_deriv_mean, qf_deriv_sdev])

    sess.close()

""" 4.3. prediction & visualization """
with tf.Session() as sess:
    f_samples = gpr_mono.variational_sgpr_sample(n_sample=10000,
                                                 qf_mean=qf_mean_val,
                                                 qf_cov=qf_vcov_val)
    f_deriv_samples = gpr_mono.variational_mfvi_sample(n_sample=5000,
                                                       qf_mean=qf_deriv_mean_val,
                                                       qf_sdev=qf_deriv_sdev_val)
    f_samples_val, f_deriv_samples_val = sess.run([f_samples,
                                                   f_deriv_samples])
    sess.close()

# still use exact posterior predictive
f_test_val = gp.sample_posterior_full(X_new=X_test, X=X_train,
                                      f_sample=f_samples_val.T,
                                      ls=DEFAULT_LS_VAL,
                                      kernel_func=gp.rbf)

df_test_val = gp.sample_posterior_full(X_new=X_test, X=X_train,
                                       f_sample=f_deriv_sample_val.T,
                                       ls=DEFAULT_LS_VAL,
                                       kernel_func=gpr_mono.rbf_hess_1d)

# visualize
mu = np.mean(f_test_val, axis=1)
mu_deriv = np.mean(df_test_val, axis=1)
cov = np.var(f_test_val, axis=1)
cov_deriv = np.var(df_test_val, axis=1)

visual_util.gpr_1d_visual(mu, cov,
                          X_train=X_train, y_train=y_train,
                          X_test=X_test, y_test=y_test,
                          title="RBF, Sparse Gaussian Process",
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX, "sgpr_gpr.png"))

visual_util.gpr_1d_visual(mu_deriv, cov_deriv,
                          X_train=X_train, y_train=y_train,
                          X_test=X_test, y_test=y_test,
                          title="RBF Derivative, Sparse Gaussian Process",
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX, "sgpr_gpr_deriv.png"),
                          add_reference=True)
