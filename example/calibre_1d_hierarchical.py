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
import calibre.util.matrix as matrix_util

from calibre.util.inference import make_value_setter
from calibre.util.data import generate_1d_data, sin_curve_1d
from calibre.util.model import sparse_softmax
from calibre.util.gp_flow import fit_base_gp_models, DEFAULT_KERN_FUNC_DICT

import matplotlib.pyplot as plt
import seaborn as sns

tfd = tfp.distributions

_SAVE_ADDR_PREFIX = "./result/calibre_1d_tree"
_FIT_BASE_MODELS = False

_EXAMPLE_DICTIONARY_SIMPLE = {
    "root": ["rbf", #"poly",
             # "period",
             "rquad"
             ],
    "rbf": ["rbf_1", "rbf_0.5", "rbf_0.2",
            "rbf_0.05", "rbf_0.01", "rbf_auto"],
    # "period": ["period0.5_0.15", "period1_0.15",
    #            "period1.5_0.15", "period_auto"],
    "rquad": ["rquad1_0.1", "rquad1_0.2", "rquad1_0.5",
              "rquad2_0.1", "rquad2_0.2", "rquad2_0.5", #"rquad_auto"
             ],
    #"poly": ["poly_1", "poly_2", "poly_3"]
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

DEFAULT_LS_WEIGHT = 0.2
DEFAULT_LS_RESID = 0.4

"""""""""""""""""""""""""""""""""
# 1. Generate data
"""""""""""""""""""""""""""""""""

N_train = 20
N_test = 50
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
                       n_valid_sample=1000,
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
family_tree_dict = _EXAMPLE_DICTIONARY_SIMPLE

num_results = 10000
num_burnin_steps = 10000

# define mcmc computation graph
mcmc_graph = tf.Graph()
with mcmc_graph.as_default():
    # build likelihood by explicitly
    log_joint = ed.make_log_joint_fn(adaptive_ensemble.model_tailfree)

    # aggregate node-specific variable names
    cond_weight_temp_names = ['temp_{}'.format(model_name) for
                              model_name in
                              tail_free.get_parent_node_names(family_tree_dict)]
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

        return log_joint(X=X_test,
                         base_pred=base_test_pred,
                         family_tree=family_tree_dict,
                         ls_weight=DEFAULT_LS_WEIGHT,
                         ls_resid=DEFAULT_LS_RESID,
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
with open(os.path.join(_SAVE_ADDR_PREFIX, 'weight_sample.pkl'), 'wb') as file:
    pk.dump(weight_sample_val, file, protocol=pk.HIGHEST_PROTOCOL)
with open(os.path.join(_SAVE_ADDR_PREFIX, 'ensemble_resid_sample.pkl'), 'wb') as file:
    pk.dump(resid_sample_val, file, protocol=pk.HIGHEST_PROTOCOL)

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
# compute GP prediction for weight GP and residual GP
model_weight_valid_sample = []
for model_weight_sample in weight_sample_val:
    model_weight_valid_sample.append(
        gp.sample_posterior_full(X_new=X_valid, X=X_test,
                                 f_sample=model_weight_sample.T,
                                 ls=DEFAULT_LS_WEIGHT,
                                 kern_func=gp.rbf).T.astype(np.float32)
    )

ensemble_resid_valid_sample = (
    gp.sample_posterior_full(X_new=X_valid, X=X_test,
                             f_sample=resid_sample_val.T,
                             ls=DEFAULT_LS_RESID, kern_func=gp.rbf).T
)

# compute sample for posterior mean
# TODO(jereliu): refactor into part of adaptive ensemble module
raw_weights_dict = dict(zip(tail_free.get_nonroot_node_names(family_tree_dict),
                            model_weight_valid_sample))
parent_temp_dict = dict(zip(tail_free.get_parent_node_names(family_tree_dict),
                            temp_sample_val))

kernel_func = gp.rbf
link_func = sparse_softmax
ridge_factor = 1e-3

eval_graph = tf.Graph()

with eval_graph.as_default():
    cond_weights_dict = (
        tail_free.compute_cond_weights(X_valid,
                                       family_tree=family_tree_dict,
                                       raw_weights_dict=raw_weights_dict,
                                       parent_temp_dict=parent_temp_dict,
                                       kernel_func=kernel_func,
                                       link_func=link_func,
                                       ridge_factor=ridge_factor,
                                       ls=DEFAULT_LS_WEIGHT))
    ensemble_weights, ensemble_model_names = (
        tail_free.compute_leaf_weights(node_weights=cond_weights_dict,
                                       family_tree=family_tree_dict,
                                       name='ensemble_weight')
    )

    base_model_pred = np.asarray(
        [base_valid_pred[model_name] for model_name in ensemble_model_names]).T

    FW = tf.multiply(base_model_pred, ensemble_weights)
    ensemble_mean = tf.reduce_sum(FW, axis=-1, name="ensemble_mean")

    eval_graph.finalize()

with tf.Session(graph=eval_graph) as sess:
    (ensemble_mean_val,
     ensemble_weights_val,
     cond_weights_dict_val) = sess.run([ensemble_mean,
                                        ensemble_weights,
                                        cond_weights_dict])

    # compute sample for full posterior
    ensemble_sample_val = ensemble_mean_val + ensemble_resid_valid_sample

# compute covariance matrix among model weights
model_weights_raw = np.asarray([raw_weights_dict[model_name]
                                for model_name in ensemble_model_names])
model_weights_raw = np.swapaxes(model_weights_raw, 0, -1)
ensemble_weight_corr = matrix_util.corr_mat(model_weights_raw, axis=0)

with open(os.path.join(_SAVE_ADDR_PREFIX,
                       'ensemble_posterior_mean_sample.pkl'), 'wb') as file:
    pk.dump(ensemble_mean_val, file, protocol=pk.HIGHEST_PROTOCOL)

with open(os.path.join(_SAVE_ADDR_PREFIX,
                       'ensemble_posterior_dist_sample.pkl'), 'wb') as file:
    pk.dump(ensemble_sample_val, file, protocol=pk.HIGHEST_PROTOCOL)

with open(os.path.join(_SAVE_ADDR_PREFIX,
                       'ensemble_posterior_node_weight_dict.pkl'), 'wb') as file:
    pk.dump(cond_weights_dict_val, file, protocol=pk.HIGHEST_PROTOCOL)

with open(os.path.join(_SAVE_ADDR_PREFIX,
                       'ensemble_posterior_model_weights.pkl'), 'wb') as file:
    pk.dump(ensemble_weights_val, file, protocol=pk.HIGHEST_PROTOCOL)

with open(os.path.join(_SAVE_ADDR_PREFIX,
                       'ensemble_posterior_model_weights_corr.pkl'), 'wb') as file:
    pk.dump(ensemble_weight_corr, file, protocol=pk.HIGHEST_PROTOCOL)

""" 2.3.2. visualize: base prediction """
base_pred_dict = {key: value for key, value in base_valid_pred.items()
                  if key in ensemble_model_names}

visual_util.plot_base_prediction(base_pred=base_pred_dict,
                                 X_valid=X_valid, y_valid=y_valid,
                                 X_train=X_train, y_train=y_train,
                                 save_addr=os.path.join(
                                     _SAVE_ADDR_PREFIX, "ensemble_base_model_fit.png"))

""" 2.3.3. visualize: ensemble posterior predictive mean """

posterior_mean_mu = np.nanmean(ensemble_mean_val, axis=0)
posterior_mean_cov = np.nanvar(ensemble_mean_val, axis=0)

posterior_mean_median = np.nanmedian(ensemble_mean_val, axis=0)
posterior_mean_quantiles = [
    np.percentile(ensemble_mean_val,
                  [100 - (100 - q) / 2, (100 - q) / 2], axis=0)
    for q in [68, 95, 99]
]

visual_util.gpr_1d_visual(posterior_mean_mu,
                          pred_cov=posterior_mean_cov,
                          X_train=X_test, y_train=y_test,
                          X_test=X_valid, y_test=y_valid,
                          title="Ensemble Posterior Mean, Hamilton MC",
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                 "ensemble_hmc_posterior_mean.png")
                          )

visual_util.gpr_1d_visual(posterior_mean_median,
                          pred_cov=None,
                          pred_quantiles=posterior_mean_quantiles,
                          X_train=X_test, y_train=y_test,
                          X_test=X_valid, y_test=y_valid,
                          title="Ensemble Posterior Mean Quantiles, Hamilton MC",
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                 "ensemble_hmc_posterior_mean_quantile.png")
                          )

""" 2.3.4. visualize: ensemble residual """

posterior_resid_mu = np.nanmean(ensemble_resid_valid_sample, axis=0)
posterior_resid_cov = np.nanvar(ensemble_resid_valid_sample, axis=0)

posterior_resid_median = np.nanmedian(ensemble_resid_valid_sample, axis=0)
posterior_resid_quantiles = [
    np.percentile(ensemble_resid_valid_sample,
                  [100 - (100 - q) / 2, (100 - q) / 2], axis=0)
    for q in [68, 95, 99]
]

visual_util.gpr_1d_visual(posterior_resid_mu,
                          pred_cov=posterior_resid_cov,
                          X_train=X_test, y_train=y_test,
                          X_test=X_valid, y_test=y_valid,
                          title="Ensemble Posterior Residual, Hamilton MC",
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                 "ensemble_hmc_posterior_resid.png")
                          )

visual_util.gpr_1d_visual(posterior_resid_median,
                          pred_cov=None,
                          pred_quantiles=posterior_resid_quantiles,
                          X_train=X_test, y_train=y_test,
                          X_test=X_valid, y_test=y_valid,
                          title="Ensemble Posterior Residual Quantiles, Hamilton MC",
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                 "ensemble_hmc_posterior_resid_quantile.png")
                          )

""" 2.3.5. visualize: ensemble posterior full """

posterior_dist_mu = np.nanmean(ensemble_sample_val, axis=0)
posterior_dist_cov = np.nanvar(ensemble_sample_val, axis=0)

posterior_dist_median = np.nanmedian(ensemble_sample_val, axis=0)
posterior_dist_quantiles = [
    np.percentile(ensemble_sample_val,
                  [100 - (100 - q) / 2, (100 - q) / 2], axis=0)
    for q in [68, 95, 99]
]

visual_util.gpr_1d_visual(posterior_dist_mu,
                          pred_cov=posterior_dist_cov,
                          X_train=X_test, y_train=y_test,
                          X_test=X_valid, y_test=y_valid,
                          title="Ensemble Posterior Predictive, Hamilton MC",
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                 "ensemble_hmc_posterior_full.png")
                          )

visual_util.gpr_1d_visual(posterior_dist_median,
                          pred_cov=None,
                          pred_quantiles=posterior_dist_quantiles,
                          X_train=X_test, y_train=y_test,
                          X_test=X_valid, y_test=y_valid,
                          title="Ensemble Posterior Predictive Quantiles, Hamilton MC",
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                 "ensemble_hmc_posterior_full_quantile.png")
                          )

""" 2.3.6. visualize: ensemble posterior reliability """

visual_util.prob_calibration_1d(
    y_valid, ensemble_sample_val.T,
    title="Ensemble, Hamilton MC",
    save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                           "ensemble_hmc_calibration_prob.png"))

visual_util.marginal_calibration_1d(
    y_valid, ensemble_sample_val.T,
    title="Ensemble, Hamilton MC",
    save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                           "ensemble_hmc_calibration_marginal.png"))

""" 2.3.7. visualize: base ensemble weight with uncertainty """
visual_util.plot_ensemble_weight_mean_1d(X=X_valid, weight_sample=ensemble_weights_val,
                                         model_names=ensemble_model_names,
                                         save_addr_prefix=os.path.join(
                                             _SAVE_ADDR_PREFIX, "ensemble_hmc_model"))

visual_util.plot_ensemble_weight_median_1d(X=X_valid, weight_sample=ensemble_weights_val,
                                           model_names=ensemble_model_names,
                                           save_addr_prefix=os.path.join(
                                               _SAVE_ADDR_PREFIX, "ensemble_hmc_model"))

# model family weights
ensemble_weights_family = np.stack(
    [cond_weights_dict_val[key] for key in family_tree_dict['root']], axis=-1)
visual_util.plot_ensemble_weight_mean_1d(X=X_valid,
                                         weight_sample=ensemble_weights_family,
                                         model_names=family_tree_dict['root'],
                                         save_addr_prefix=os.path.join(
                                             _SAVE_ADDR_PREFIX, "ensemble_hmc_family"))
visual_util.plot_ensemble_weight_median_1d(X=X_valid,
                                           weight_sample=ensemble_weights_family,
                                           model_names=family_tree_dict['root'],
                                           save_addr_prefix=os.path.join(
                                               _SAVE_ADDR_PREFIX, "ensemble_hmc_family"))

""" 2.3.8. visualize: ensemble weight covariance (model compositionality) """
for i in range(ensemble_weight_corr.shape[0]):
    corr_mat = ensemble_weight_corr[i]
    x_value = X_valid[i][0]
    visual_util.model_composition_1d(
        x_value, corr_mat, ensemble_weights_val,
        base_pred_dict,
        X_valid, y_valid,
        X_test, y_test,
        model_names=ensemble_model_names,
        save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                               "model composition/{}.png".format(i)))

"""""""""""""""""""""""""""""""""
# 3. PSR Augmented VI
"""""""""""""""""""""""""""""""""
# TODO(jereliu): to implement
