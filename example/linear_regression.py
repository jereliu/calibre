"""GPR using MCMC, mean-field VI, and augmented VI"""
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed

from calibre.model import gaussian_process as gp
from calibre.util.data import generate_1d_data, sin_curve_1d
from calibre.util.visual import gpr_1d_visual

import matplotlib.pyplot as plt

tfd = tfp.distributions

"""""""""""""""""""""""""""""""""
# 1. Generate data
"""""""""""""""""""""""""""""""""
N = 20

X_train, y_train = generate_1d_data(N=N, f=sin_curve_1d,
                                    noise_sd=0.03, seed=100)
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

"""""""""""""""""""""""""""""""""
# 2. Main
"""""""""""""""""""""""""""""""""
""" 2.1. define model"""


def linear_regression(features):
    coeffs = ed.Normal(loc=0., scale=1.,
                       sample_shape=features.shape[1], name="coeffs")
    outcomes = ed.Normal(loc=tf.tensordot(features, coeffs, [[1], [0]]),
                         scale=np.sqrt(0.05).astype(np.float32),
                         name="outcomes")
    return outcomes


""" 2.2. define comp graph"""
num_results = 10000
num_burnin_steps = 5000
ls_val = .14

# define mcmc computation graph
# tf.reset_default_graph()
mcmc_graph = tf.Graph()
with mcmc_graph.as_default():
    # build RBF features
    N = X_train.shape[0]
    K_mat = gp.rbf(X_train, ls=ls_val) + 1e-6 * tf.eye(N)
    S, U, V = tf.svd(K_mat)
    num_feature = tf.reduce_sum(tf.cast(S > 1e-10, tf.int32))
    features = tf.tensordot(U, tf.diag(S), [[1], [0]])[:, :num_feature]

    # features = tf.random_normal([10, 2])
    # alpha = tf.random_normal(shape=[num_feature])
    # outcomes_value = tf.tensordot(features, alpha, [[1], [0]])

    outcomes_value = tf.constant(y_train)
    log_joint = ed.make_log_joint_fn(linear_regression)


    def target_log_prob_fn(coeffs_value):
        return log_joint(features, coeffs=coeffs_value, outcomes=outcomes_value)


    # set up state container
    initial_state = [
        tf.random_normal([num_feature], stddev=0.01, name='init_gp_func'),
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

    alpha_sample, = state

    # set up init op
    init_op = tf.global_variables_initializer()

    mcmc_graph.finalize()

""" 2.3. execute sampling"""

with tf.Session(graph=mcmc_graph) as sess:
    init_op.run()
    [
        alpha_sample_val,
        feature_val, outcome_val,
        is_accepted_,
    ] = sess.run(
        [
            alpha_sample,
            features, outcomes_value,
            kernel_results.is_accepted,
        ])
    print('Acceptance Rate: {}'.format(np.mean(is_accepted_)))
    sess.close()

""" 2.4. prediction and visualization"""
# compute sample
gpf_sample_lr = feature_val.dot(alpha_sample_val.T)

f_test_val = gp.sample_posterior_full(X_new=X_test, X=X_train,
                                      f_sample=gpf_sample_lr,
                                      ls=ls_val, kernfunc=gp.rbf,
                                      ridge_factor=1e-4)

# visualize
mu = np.mean(f_test_val, axis=1)
cov = np.var(f_test_val, axis=1)

gpr_1d_visual(mu, cov,
              X_train, y_train, X_test, y_test,
              title="RBF, MCMC, Linear Regression Approximation",
              save_addr="./plot/gpr_mcmc_lr.png")
