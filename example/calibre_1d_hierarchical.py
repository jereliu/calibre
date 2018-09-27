"""Calibre (Adaptive Ensemble) with hierarchical structure using MCMC and Penalized VI. """
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
from calibre.model import tailfree_process as tail_free
from calibre.model import adaptive_ensemble

import calibre.util.visual as visual_util
from calibre.util.inference import make_value_setter
from calibre.util.data import generate_1d_data, sin_curve_1d
from calibre.util.model import sparse_softmax
from calibre.util.gp_flow import fit_base_gp_models, DEFAULT_KERN_FUNC_DICT

import matplotlib.pyplot as plt
import seaborn as sns

tfd = tfp.distributions

_SAVE_ADDR_PREFIX = "./result/calibre_1d_tree"
_FIT_BASE_MODELS = False

_EXAMPLE_DICTIONARY_SHORT = {
    "root": ["rbf", "poly"],
    "rbf": ["rbf_1", "rbf_0.5", "rbf_0.2",
            "rbf_0.05", "rbf_0.01", "rbf_auto"],
    "poly": ["poly_1", "poly_2", "poly_3"]
}

_EXAMPLE_DICTIONARY_FULL = {
    "root": ["rbf", "period", "rquad", "poly"],
    "rbf": ["rbf_1", "rbf_0.5", "rbf_0.2",
            "rbf_0.05", "rbf_0.01", "rbf_auto"],
    "period": ["period0.5_0.15", "period1_0.15",
               "period1.5_0.15", "period_auto"],
    "rquad": ["rquad1_0.1", "rquad1_0.2", "rquad1_0.5",
              "rquad2_0.1", "rquad2_0.2", "rquad2_0.5", "rquad_auto"],
    "poly": ["poly_1", "poly_2", "poly_3"]
}

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

"""2.1. sampler basic config"""
N = X_test.shape[0]
K = len(base_test_pred)
ls_weight = 0.15
ls_resid = 0.1
family_tree_dict = _EXAMPLE_DICTIONARY_SHORT

num_results = 10000
num_burnin_steps = 5000

# define mcmc computation graph
mcmc_graph = tf.Graph()
with mcmc_graph.as_default():
    # build likelihood by explicitly
    log_joint = ed.make_log_joint_fn(adaptive_ensemble.model_tailfree)

    # aggregate node-specific variable names
    cond_weight_temp_names = ['temp_{}'.format(model_name) for
                       model_name in
                       tail_free.get_nonleaf_node_names(family_tree_dict)]
    node_weight_names = ['base_weight_{}'.format(model_name) for
                         model_name in
                         tail_free.get_nonroot_node_names(family_tree_dict)]
    node_specific_varnames = cond_weight_temp_names + node_weight_names

    def target_log_prob_fn(sigma, ensemble_resid,
                           *node_specific_positional_args):
        """Unnormalized target density as a function of states."""
        # build kwargs for base model weight using positional args
        node_specific_kwargs = dict(zip(node_specific_varnames,
                                        node_specific_positional_args))

        return log_joint(X=X_test, base_pred=base_test_pred,
                         family_tree=family_tree_dict,
                         ls_weight=ls_weight, ls_resid=ls_resid,
                         y=y_test.squeeze(),
                         sigma=sigma,
                         ensemble_resid=ensemble_resid,
                         **node_specific_kwargs)


    # set up state container
    initial_state = [
                        # tf.random_normal([N, K], stddev=0.01, name='init_ensemble_weight'),
                        # tf.random_normal([N], stddev=0.01, name='init_f_ensemble'),
                        tf.constant(0.1, name='init_sigma'),
                        tf.random_normal([N], stddev=0.01,
                                         name='init_ensemble_resid'),
                    ] + [
                        tf.random_normal([], stddev=0.01,
                                         name='init_{}'.format(var_name)) for
                        var_name in cond_weight_temp_names
                    ] + [
                        tf.random_normal([N], stddev=0.01,
                                         name='init_{}'.format(var_name)) for
                        var_name in node_weight_names
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

    sigma_sample, ensemble_resid_sample = state[:2]
    temp_sample = state[2:2 + len(cond_weight_temp_names)]
    weight_sample = state[2 + len(cond_weight_temp_names):]

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






"""""""""""""""""""""""""""""""""
# 3. PSR Augmented VI
"""""""""""""""""""""""""""""""""
# TODO(jereliu): to implement
