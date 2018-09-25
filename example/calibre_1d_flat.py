"""Calibre (Adaptive Ensemble) with flat model structure. """
import os
import sys

import pickle as pk
import pandas as pd

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed

import gpflowSlim as gpf

sys.path.extend([os.getcwd()])

from calibre.model import gaussian_process as gp
from calibre.model import adaptive_ensemble

import calibre.util.visual as visual_util
from calibre.util.inference import make_value_setter
from calibre.util.data import generate_1d_data, sin_curve_1d
from calibre.util.model import sparse_softmax
from calibre.util.gp_flow import fit_base_gp_models, DEFAULT_KERN_FUNC_DICT

import matplotlib.pyplot as plt
import seaborn as sns

tfd = tfp.distributions

_SAVE_ADDR_PREFIX = "./result/calibre_1d_flat"
_FIT_BASE_MODELS = False

"""""""""""""""""""""""""""""""""
# 1. Generate data
"""""""""""""""""""""""""""""""""

N_train = 20
N_test = 20
N_valid = 500

X_train, y_train = generate_1d_data(N=N_train, f=sin_curve_1d,
                                    noise_sd=0.03, seed=1000,
                                    uniform_x=True)
X_test, y_test = generate_1d_data(N=N_test, f=sin_curve_1d,
                                  noise_sd=0.03, seed=2000,
                                  uniform_x=True)

X_train = np.expand_dims(X_train, 1).astype(np.float32)
y_train = y_train.astype(np.float32)
X_test = np.expand_dims(X_test, 1).astype(np.float32)
y_test = y_test.astype(np.float32)

std_y_train = np.std(y_train)

X_valid = np.expand_dims(np.linspace(-1, 2, N_valid), 1).astype(np.float32)
y_valid = sin_curve_1d(X_valid)

N, D = X_train.shape

#
plt.plot(np.linspace(-0.5, 1.5, 100),
         sin_curve_1d(np.linspace(-0.5, 1.5, 100)), c='black')
plt.plot(X_train.squeeze(), y_train.squeeze(),
         'o', c='red', markeredgecolor='black')
plt.close()

""" 1.1. Build base GP models using GPflow """
if _FIT_BASE_MODELS:
    fit_base_gp_models(X_train, y_train,
                       X_test, y_test,
                       X_valid, y_valid,
                       kern_func_dict=DEFAULT_KERN_FUNC_DICT,
                       n_valid_sample=5000,
                       save_addr_prefix="{}/base".format(_SAVE_ADDR_PREFIX))

"""""""""""""""""""""""""""""""""
# 2. MCMC
"""""""""""""""""""""""""""""""""
with open(os.path.join(_SAVE_ADDR_PREFIX, 'base/base_test_pred.pkl'), 'rb') as file:
    base_test_pred = pk.load(file)

with open(os.path.join(_SAVE_ADDR_PREFIX, 'base/base_valid_pred.pkl'), 'rb') as file:
    base_valid_pred = pk.load(file)

base_test_pred = {key: value for key, value in base_test_pred.items() if
                  ('rbf' in key)}
base_valid_pred = {key: value for key, value in base_valid_pred.items()
                   if key in list(base_test_pred.keys())}

"""2.1. sampler basic config"""
N = X_test.shape[0]
K = len(base_test_pred)
num_results = 10000
num_burnin_steps = 5000
ls_weight = 0.15
ls_resid = 0.1

# define mcmc computation graph
mcmc_graph = tf.Graph()
with mcmc_graph.as_default():
    # build likelihood by explicitly
    log_joint = ed.make_log_joint_fn(adaptive_ensemble.model_flat)

    # Note: ignore the first weight
    base_weight_names = ['base_weight_{}'.format(model_name) for
                         model_name in list(base_test_pred.keys())]


    def target_log_prob_fn(sigma, temp, ensemble_resid,
                           *base_weight_positional_args):
        """Unnormalized target density as a function of states."""
        # build kwargs for base model weight using positional args
        base_weight_kwargs = dict(zip(base_weight_names, base_weight_positional_args))

        return log_joint(X=X_test, base_pred=base_test_pred,
                         family_tree=None,
                         ls_weight=ls_weight, ls_resid=ls_resid,
                         y=y_test.squeeze(),
                         sigma=sigma,
                         temp=temp,
                         ensemble_resid=ensemble_resid,
                         **base_weight_kwargs)


    # set up state container
    initial_state = [
                        # tf.random_normal([N, K], stddev=0.01, name='init_ensemble_weight'),
                        # tf.random_normal([N], stddev=0.01, name='init_f_ensemble'),
                        tf.constant(0.1, name='init_sigma'),
                        tf.constant(0.1, name='init_temp'),
                        tf.random_normal([N], stddev=0.01,
                                         name='init_ensemble_resid'),
                    ] + [
                        tf.random_normal([N], stddev=0.01,
                                         name='init_{}'.format(model_name)) for
                        model_name in base_weight_names
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
        parallel_iterations=1
    )

    sigma_sample, temp_sample, ensemble_resid_sample = state[:3]
    weight_sample = state[3:]

    # set up init op
    init_op = tf.global_variables_initializer()

    mcmc_graph.finalize()

""" 2.2. execute sampling"""
with tf.Session(graph=mcmc_graph) as sess:
    init_op.run()
    [
        sigma_sample_val,
        temp_sample_val,
        resid_sample_val,
        weight_sample_val,
        is_accepted_,
    ] = sess.run(
        [
            sigma_sample,
            temp_sample,
            ensemble_resid_sample,
            weight_sample,
            kernel_results.is_accepted,
        ])
    print('Acceptance Rate: {}'.format(np.mean(is_accepted_)))
    sess.close()

with open(os.path.join(_SAVE_ADDR_PREFIX, 'sigma_sample.pkl'), 'wb') as file:
    pk.dump(sigma_sample_val, file, protocol=pk.HIGHEST_PROTOCOL)
with open(os.path.join(_SAVE_ADDR_PREFIX, 'temp_sample.pkl'), 'wb') as file:
    pk.dump(temp_sample_val, file, protocol=pk.HIGHEST_PROTOCOL)
with open(os.path.join(_SAVE_ADDR_PREFIX, 'ensemble_resid_sample.pkl'), 'wb') as file:
    pk.dump(resid_sample_val, file, protocol=pk.HIGHEST_PROTOCOL)
with open(os.path.join(_SAVE_ADDR_PREFIX, 'weight_sample.pkl'), 'wb') as file:
    pk.dump(weight_sample_val, file, protocol=pk.HIGHEST_PROTOCOL)

""" 2.3. prediction and visualization"""

with open(os.path.join(_SAVE_ADDR_PREFIX, 'sigma_sample.pkl'), 'rb') as file:
    sigma_sample_val = pk.load(file)
with open(os.path.join(_SAVE_ADDR_PREFIX, 'temp_sample.pkl'), 'rb') as file:
    temp_sample_val = pk.load(file)
with open(os.path.join(_SAVE_ADDR_PREFIX, 'weight_sample.pkl'), 'rb') as file:
    weight_sample_val = pk.load(file)
with open(os.path.join(_SAVE_ADDR_PREFIX, 'ensemble_resid_sample.pkl'), 'rb') as file:
    resid_sample_val = pk.load(file)

""" 2.3.1. prediction """

# compute sample for ensemble weight
model_weight_valid_sample = []
for model_weight_sample in weight_sample_val:
    model_weight_valid_sample.append(
        gp.sample_posterior_full(X_new=X_valid, X=X_test,
                                 f_sample=model_weight_sample.T,
                                 ls=ls_weight, kern_func=gp.rbf).T.astype(np.float32)
    )

# compute sample for ensemble residual
ensemble_resid_valid_sample = gp.sample_posterior_full(
    X_new=X_valid, X=X_test,
    f_sample=resid_sample_val.T,
    ls=ls_resid, kern_func=gp.rbf).T

# compute sample for posterior mean
with tf.Session() as sess:
    W_ensemble = adaptive_ensemble.sample_posterior_weight_flat(
        model_weight_valid_sample, temp_sample_val, link_func=sparse_softmax)

    ensemble_mean = adaptive_ensemble.sample_posterior_mean_flat(
        base_valid_pred,
        weight_sample=model_weight_valid_sample,
        temp_sample=temp_sample_val,
        link_func=sparse_softmax)
    ensemble_mean_val, W_ensemble_val = sess.run([ensemble_mean, W_ensemble])

# compute sample for full posterior
ensemble_sample_val = ensemble_mean_val + ensemble_resid_valid_sample

with open(os.path.join(_SAVE_ADDR_PREFIX,
                       'ensemble_posterior_mean_sample.pkl'), 'wb') as file:
    pk.dump(ensemble_mean_val, file, protocol=pk.HIGHEST_PROTOCOL)

with open(os.path.join(_SAVE_ADDR_PREFIX,
                       'ensemble_posterior_dist_sample.pkl'), 'wb') as file:
    pk.dump(ensemble_sample_val, file, protocol=pk.HIGHEST_PROTOCOL)

""" 2.3.2. visualize: base prediction """

visual_util.plot_base_prediction(base_pred=base_valid_pred,
                                 X_valid=X_valid, y_valid=y_valid,
                                 X_train=X_train, y_train=y_train,
                                 save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                        "ensemble_base_model_fit.png"))

""" 2.3.3. visualize: base ensemble weight with uncertainty """

model_names = list(base_valid_pred.keys())
weight_sample = W_ensemble_val
X = X_valid

visual_util.plot_ensemble_weight_1d(X=X_valid, weight_sample=W_ensemble_val,
                                    model_names=list(base_valid_pred.keys()),
                                    save_addr_prefix=os.path.join(_SAVE_ADDR_PREFIX,
                                                                  "ensemble_hmc"))

""" 2.3.4. visualize: ensemble posterior predictive mean """

posterior_mean_mu = np.nanmean(ensemble_mean_val, axis=0)
posterior_mean_cov = np.nanvar(ensemble_mean_val, axis=0)

visual_util.gpr_1d_visual(posterior_mean_mu, posterior_mean_cov,
                          X_test, y_test, X_valid, y_valid,
                          title="Ensemble Posterior Mean, Hamilton MC",
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                 "ensemble_hmc_posterior_mean.png")
                          )

""" 2.3.5. visualize: ensemble residual """

posterior_resid_mu = np.nanmean(ensemble_resid_valid_sample, axis=0)
posterior_resid_cov = np.nanvar(ensemble_resid_valid_sample, axis=0)

visual_util.gpr_1d_visual(posterior_resid_mu, posterior_resid_cov,
                          X_test, y_test, X_valid, y_valid,
                          title="Ensemble Posterior Residual, Hamilton MC",
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                 "ensemble_hmc_posterior_residual.png"))

""" 2.3.6. visualize: ensemble posterior full """

posterior_dist_mu = np.nanmean(ensemble_sample_val, axis=0)
posterior_dist_cov = np.nanvar(ensemble_sample_val, axis=0)

visual_util.gpr_1d_visual(posterior_dist_mu, posterior_dist_cov,
                          X_test, y_test, X_valid, y_valid,
                          title="Ensemble Posterior Predictive, Hamilton MC",
                          save_addr="{}/ensemble_hmc_posterior_full.png")

""" 2.3.7. visualize: ensemble posterior reliability """

visual_util.plot_reliability_diagram_1d(
    y_valid, ensemble_sample_val.T,
    title="Ensemble, Hamilton MC",
    save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                           "ensemble_hmc_reliability.png"))

"""""""""""""""""""""""""""""""""
# 3. PSR Augmented VI
"""""""""""""""""""""""""""""""""
# TODO(jereliu): to implement
