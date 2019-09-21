"""Calibre (Adaptive Ensemble) with flat model structure. """
import os
import sys
import time

from importlib import reload

import pickle as pk
import pandas as pd

import numpy as np

from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed

import gpflowSlim as gpf

sys.path.extend([os.getcwd()])

from calibre.model import gaussian_process as gp
from calibre.model import gp_regression_monotone as gpr_mono
from calibre.model import parametric_ensemble

import calibre.util.visual as visual_util
import calibre.util.calibration as calib_util
import calibre.util.experiment_data as experiment_util

from calibre.util.model import sparse_softmax
from calibre.util.inference import make_value_setter
from calibre.util.gp_flow import fit_base_gp_models, DEFAULT_KERN_FUNC_DICT_GPY

import matplotlib.pyplot as plt
import seaborn as sns

tfd = tfp.distributions

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DEFAULT_LOG_LS_WEIGHT = np.log(0.1).astype(np.float32)
DEFAULT_LOG_LS_RESID = np.log(0.1).astype(np.float32)

_SAVE_ADDR_PREFIX = "./result/parametric_ensemble/1d_simple_sine"
_FIT_BASE_MODELS = False
_FIT_MCMC_MODELS = True
_FIT_VI_MODELS = True
_FIT_AUG_VI_MODELS = False
_FIT_CALIB_MODELS = True


_EXAMPLE_DICTIONARY_SIMPLE = {
    "root": ["rbf_0.2",
             "rbf_0.1",
             "rbf_0.02",
             "rbf_0.01"
             ]
}

"""""""""""""""""""""""""""""""""
# 1. Generate data
"""""""""""""""""""""""""""""""""

N_train = 20
N_test = 20
N_valid = 500

(X_train, y_train,
 X_test, y_test,
 X_valid, y_valid, calib_sample_id) = experiment_util.generate_data_1d(
    N_train=20, N_test=20, N_valid=500, noise_sd=0.03,
    data_range=(0., 1.), valid_range=(-0.5, 1.5),
    seed_train=1000, seed_test=1500, seed_calib=100)

#
plt.plot(X_valid, y_valid, c='black')
plt.plot(X_train.squeeze(), y_train.squeeze(),
         'o', c='red', markeredgecolor='black')
plt.plot(X_test.squeeze(), y_test.squeeze(),
         'o', c='blue', markeredgecolor='black')
plt.savefig("{}/data.png".format(_SAVE_ADDR_PREFIX))
plt.close()

""" 1.1. Build base GP models using GPflow """
if _FIT_BASE_MODELS:
    fit_base_gp_models(X_train, y_train,
                       X_test, y_test,
                       X_valid, y_valid,
                       kern_func_dict=DEFAULT_KERN_FUNC_DICT_GPY,
                       n_valid_sample=500,
                       save_addr_prefix="{}/base".format(_SAVE_ADDR_PREFIX))

"""""""""""""""""""""""""""""""""
# 2. MCMC
"""""""""""""""""""""""""""""""""
family_name = "hmc"
family_name_full = "Hamilton MC"

_SAVE_ADDR = _SAVE_ADDR_PREFIX

with open(os.path.join(_SAVE_ADDR_PREFIX, 'base/base_test_pred.pkl'), 'rb') as file:
    base_test_pred = pk.load(file)

with open(os.path.join(_SAVE_ADDR_PREFIX, 'base/base_valid_pred.pkl'), 'rb') as file:
    base_valid_pred = pk.load(file)

model_names = ["rbf_0.2", "rbf_0.05", "rbf_0.5",
               "rquad1", "rquad2"]
base_test_pred = {key: value for key, value in base_test_pred.items() if
                  (key in model_names)}
base_valid_pred = {key: value for key, value in base_valid_pred.items()
                   if key in model_names}

"""2.1. sampler basic config"""
N = X_test.shape[0]
K = len(base_test_pred)
num_results = 5000
num_burnin_steps = 5000

# define mcmc computation graph
mcmc_graph = tf.Graph()
with mcmc_graph.as_default():
    log_joint = ed.make_log_joint_fn(parametric_ensemble.model)

    ensemble_model_names = list(base_test_pred.keys())
    base_weight_names = ['base_weight_{}'.format(model_name) for
                         model_name in ensemble_model_names]
    model_specific_varnames = base_weight_names


    def target_log_prob_fn(sigma, temp, ensemble_resid,
                           *model_specific_positional_args):
        """Unnormalized target density as a function of states."""
        # build kwargs for base model weight using positional args
        model_specific_kwargs = dict(zip(model_specific_varnames,
                                         model_specific_positional_args))

        return log_joint(X=X_test,
                         base_pred=base_test_pred,
                         log_ls_resid=DEFAULT_LOG_LS_RESID,
                         y=y_test.squeeze(),
                         sigma=sigma,
                         temp=temp,
                         ensemble_resid=ensemble_resid,
                         **model_specific_kwargs)


    # set up state container
    initial_state = [
        tf.constant(0.1, name='init_sigma'),
        tf.constant(0.1, name='init_temp'),
        tf.random_normal([N], stddev=0.01,
                         name='init_ensemble_resid'),
    ]
    initial_state_base_weight = [
        tf.constant(1., name='init_base_weight_{}'.format(model_name)) for
        model_name in base_weight_names
    ]

    initial_state = initial_state + initial_state_base_weight

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
    base_weight_sample = state[3:]

    # set up init op
    init_op = tf.global_variables_initializer()

    mcmc_graph.finalize()

""" 2.2. execute sampling"""
time_start = time.time()
with tf.Session(graph=mcmc_graph) as sess:
    init_op.run()
    [
        sigma_sample_val,
        temp_sample_val,
        resid_sample_val,
        base_weight_sample_val,
        is_accepted_,
    ] = sess.run(
        [
            sigma_sample,
            temp_sample,
            ensemble_resid_sample,
            base_weight_sample,
            kernel_results.is_accepted,
        ])
    total_min = (time.time() - time_start) / 60.

    print('Acceptance Rate: {}'.format(np.mean(is_accepted_)))
    print('Total time: {:.2f} min'.format(total_min))
    sess.close()

with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/sigma_sample.pkl'.format(family_name)), 'wb') as file:
    pk.dump(sigma_sample_val, file, protocol=pk.HIGHEST_PROTOCOL)
with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/temp_sample.pkl'.format(family_name)), 'wb') as file:
    pk.dump(temp_sample_val, file, protocol=pk.HIGHEST_PROTOCOL)
with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/ensemble_resid_sample.pkl'.format(family_name)), 'wb') as file:
    pk.dump(resid_sample_val, file, protocol=pk.HIGHEST_PROTOCOL)
with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/ensemble_ls_weight_sample.pkl'.format(family_name)), 'wb') as file:
    pk.dump(base_weight_sample_val, file, protocol=pk.HIGHEST_PROTOCOL)

""" 2.3. prediction and visualization"""

with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/sigma_sample.pkl'.format(family_name)), 'rb') as file:
    sigma_sample_val = pk.load(file)
with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/temp_sample.pkl'.format(family_name)), 'rb') as file:
    temp_sample_val = pk.load(file)
with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/base_weight_sample.pkl'.format(family_name)), 'rb') as file:
    base_weight_sample_val = pk.load(file)
with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/ensemble_resid_sample.pkl'.format(family_name)), 'rb') as file:
    resid_sample_val = pk.load(file)

""" 2.3.1. prediction """

# compute GP prediction for residual GP
ensemble_resid_valid_sample = gp.sample_posterior_full(
    X_new=X_valid, X=X_test,
    f_sample=resid_sample_val.T,
    ls=np.exp(DEFAULT_LOG_LS_RESID), kernel_func=gp.rbf).T

# compute sample for posterior mean
with tf.Session() as sess:
    W_ensemble = parametric_ensemble.sample_posterior_weight(
        base_weight_sample_val,
        temp_sample_val,
        link_func=sparse_softmax)

    ensemble_mean = parametric_ensemble.sample_posterior_mean(
        base_valid_pred,
        weight_sample=base_weight_sample_val,
        temp_sample=temp_sample_val,
        link_func=sparse_softmax)
    ensemble_mean_val, W_ensemble_val = sess.run([ensemble_mean, W_ensemble])

    ensemble_sample_val = ensemble_mean_val + ensemble_resid_valid_sample

with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/ensemble_posterior_mean_sample.pkl'.format(family_name)), 'wb') as file:
    pk.dump(ensemble_mean_val, file, protocol=pk.HIGHEST_PROTOCOL)

with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/ensemble_posterior_dist_sample.pkl'.format(family_name)), 'wb') as file:
    pk.dump(ensemble_sample_val, file, protocol=pk.HIGHEST_PROTOCOL)

""" 2.3.2. visualize: base prediction """

visual_util.plot_base_prediction(base_pred=base_valid_pred,
                                 model_names=list(base_valid_pred.keys()),
                                 X_valid=X_valid, y_valid=y_valid,
                                 X_train=X_train, y_train=y_train,
                                 save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                        "ensemble_base_model_fit.png"))

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
                          title="Posterior Mean, {}".format(family_name_full),
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                 "{}/ensemble_posterior_mean.png".format(family_name))
                          )

visual_util.gpr_1d_visual(posterior_mean_median,
                          pred_cov=None,
                          pred_quantiles=posterior_mean_quantiles,
                          X_train=X_test, y_train=y_test,
                          X_test=X_valid, y_test=y_valid,
                          title="Posterior Mean Quantiles, {}".format(family_name_full),
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                 "{}/ensemble_posterior_mean_quantile.png".format(family_name))
                          )

visual_util.gpr_1d_visual(pred_mean=None, pred_cov=None, pred_quantiles=[],
                          pred_samples=list(ensemble_mean_val)[:2500],
                          X_train=X_test, y_train=y_test,
                          X_test=X_valid, y_test=y_valid,
                          title="Posterior Samples, {}".format(family_name_full),
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                 "{}/ensemble_posterior_sample.png".format(family_name))
                          )

""" 2.3.4. visualize: ensemble residual """

ensemble_resid_valid_sample = ensemble_sample_val - ensemble_mean_val

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
                          rmse_id=calib_sample_id,
                          title="Posterior Residual, {}".format(family_name_full),
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                 "{}/ensemble_posterior_resid.png".format(family_name))
                          )

visual_util.gpr_1d_visual(posterior_resid_median,
                          pred_cov=None,
                          pred_quantiles=posterior_resid_quantiles,
                          X_train=X_test, y_train=y_test,
                          X_test=X_valid, y_test=y_valid,
                          rmse_id=calib_sample_id,
                          title="Posterior Residual Quantiles, {}".format(family_name_full),
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                 "{}/ensemble_posterior_resid_quantile.png".format(family_name))
                          )

visual_util.gpr_1d_visual(pred_mean=None, pred_cov=None, pred_quantiles=[],
                          pred_samples=list(ensemble_resid_valid_sample)[:2500],
                          X_train=X_test, y_train=y_test,
                          X_test=X_valid, y_test=y_valid,
                          rmse_id=calib_sample_id,
                          title="Posterior Residual Samples, {}".format(family_name_full),
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                 "{}/ensemble_posterior_resid_sample.png".format(family_name))
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

# visualize prediction
visual_util.gpr_1d_visual(posterior_dist_mu,
                          pred_cov=posterior_dist_cov,
                          X_train=X_test, y_train=y_test,
                          X_test=X_valid, y_test=y_valid,
                          rmse_id=calib_sample_id,
                          title="Posterior Predictive, {}".format(family_name_full),
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                 "{}/ensemble_posterior_full.png".format(family_name))
                          )

visual_util.gpr_1d_visual(posterior_dist_median,
                          pred_cov=None,
                          pred_quantiles=posterior_dist_quantiles,
                          X_train=X_test, y_train=y_test,
                          X_test=X_valid, y_test=y_valid,
                          rmse_id=calib_sample_id,
                          title="Posterior Predictive Quantiles, {}".format(family_name_full),
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                 "{}/ensemble_posterior_full_quantile.png".format(family_name))
                          )
visual_util.gpr_1d_visual(pred_mean=None, pred_cov=None, pred_quantiles=[],
                          pred_samples=list(ensemble_sample_val)[:2500],
                          X_train=X_test, y_train=y_test,
                          X_test=X_valid, y_test=y_valid,
                          rmse_id=calib_sample_id,
                          title="Posterior Predictive Samples, {}".format(family_name_full),
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                 "{}/ensemble_posterior_full_sample.png".format(family_name))
                          )

visual_util.gpr_1d_visual(pred_mean=posterior_dist_median,
                          pred_cov=None,
                          pred_quantiles=[
                              [posterior_dist_median - 3 * np.sqrt(posterior_mean_cov),
                               posterior_dist_median + 3 * np.sqrt(posterior_mean_cov)],
                              [posterior_dist_median - 3 * np.sqrt(posterior_dist_cov),
                               posterior_dist_median + 3 * np.sqrt(posterior_dist_cov)],
                          ],
                          X_train=X_test, y_train=y_test,
                          X_test=X_valid, y_test=y_valid,
                          rmse_id=calib_sample_id,
                          quantile_colors=["red", "black"],
                          quantile_alpha=0.2,
                          title="Posterior, Uncertainty Decomposition, {}".format(family_name_full),
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                 "{}/ensemble_posterior_unc_decomp.png".format(family_name))
                          )

""" 2.3.6. visualize: ensemble posterior reliability """

y_calib = y_valid[calib_sample_id]
y_sample_calib = ensemble_sample_val[:, calib_sample_id].T

visual_util.prob_calibration_1d(
    y_calib, y_sample_calib,
    title="Ensemble, {}".format(family_name_full),
    save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                           "{}/ensemble_calibration_prob.png".format(family_name)))

visual_util.coverage_index_1d(
    y_calib, y_sample_calib,
    title="Ensemble, {}".format(family_name_full),
    save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                           "{}/ensemble_credible_coverage.png".format(family_name)))

"""""""""""""""""""""""""""""""""
# 3. Variational Inference
"""""""""""""""""""""""""""""""""
_ADD_MFVI_MIXTURE = False
_N_INFERENCE_SAMPLE = 20
_N_MFVI_MIXTURE = 5

with open(os.path.join(_SAVE_ADDR_PREFIX, 'base/base_test_pred.pkl'), 'rb') as file:
    base_test_pred = pk.load(file)

with open(os.path.join(_SAVE_ADDR_PREFIX, 'base/base_valid_pred.pkl'), 'rb') as file:
    base_valid_pred = pk.load(file)

model_names = ["rbf_0.2", "rbf_0.05", "rbf_0.5",
               "rquad1", "rquad2"]
base_test_pred = {key: value for key, value in base_test_pred.items() if
                  (key in model_names)}
base_valid_pred = {key: value for key, value in base_valid_pred.items()
                   if key in model_names}

""" 3.1. basic data/algorithm config"""

family_tree_dict = _EXAMPLE_DICTIONARY_SIMPLE

n_final_sample = 1000  # number of samples to collect from variational family
max_steps = 20000  # number of training iterations

X_induce_mean = KMeans(n_clusters=15, random_state=100).fit(
    X_test).cluster_centers_.astype(np.float32)
X_induce = KMeans(n_clusters=15, random_state=100).fit(
    X_test).cluster_centers_.astype(np.float32)

# X_induce_mean = np.expand_dims(np.linspace(np.min(X_test),
#                                            np.max(X_test), 50), 1).astype(np.float32)
# X_induce = np.expand_dims(np.linspace(np.min(X_test),
#                                       np.max(X_test), 20), 1).astype(np.float32)

family_name_list = ["mfvi",
                    "sgpr",
                    "dgpr",
                    ]

for family_name in family_name_list:
    family_name_root = family_name.split("_")[0]
    if _ADD_MFVI_MIXTURE:
        family_name = "{}_mfvi_{}_mix".format(family_name, _N_MFVI_MIXTURE)

    os.makedirs('{}/{}'.format(_SAVE_ADDR_PREFIX, family_name), exist_ok=True)

    if family_name_root == "mfvi":
        family_name_full = "Mean-field VI"
        ensemble_variational_family = parametric_ensemble.variational_mfvi
        ensemble_variational_family_sample = parametric_ensemble.variational_mfvi_sample
    elif family_name_root == "sgpr":
        family_name_full = "Sparse Gaussian Process"
        ensemble_variational_family = parametric_ensemble.variational_sgpr
        ensemble_variational_family_sample = parametric_ensemble.variational_sgpr_sample
    elif family_name_root == "dgpr":
        family_name_full = "Decoupled Gaussian Process"
        ensemble_variational_family = parametric_ensemble.variational_dgpr
        ensemble_variational_family_sample = parametric_ensemble.variational_dgpr_sample

    if _FIT_VI_MODELS:
        """ 3.2. Set up the computational graph """
        vi_graph = tf.Graph()

        with vi_graph.as_default():
            # sample from variational family
            (weight_dict, resid_gp, temp, sigma, _,  # variational RVs
             # variational parameters
             weight_mean_dict, weight_vcov_dict,  # weight GP
             resid_gp_mean, resid_gp_vcov, mixture_par_resid,  # resid GP
             temp_mean, temp_sdev,  # temperature
             sigma_mean, sigma_sdev,  # obs noise
             log_ls_resid_mean, log_ls_resid_sdev,  # variational parameters, resid GP
             ) = ensemble_variational_family(X=X_test,
                                             Z=X_induce,
                                             Zm=X_induce_mean,
                                             base_pred=base_test_pred,
                                             log_ls_resid=DEFAULT_LOG_LS_RESID,
                                             kernel_func=gp.rbf,
                                             ridge_factor=1e-3,
                                             mfvi_mixture=_ADD_MFVI_MIXTURE,
                                             n_mixture=_N_MFVI_MIXTURE)

            # assemble kwargs for make_value_setter
            variational_rv_dict = {"ensemble_resid": resid_gp,
                                   "sigma": sigma,
                                   "temp": temp}
            variational_rv_dict.update(weight_dict)

            # compute the expected predictive log-likelihood
            with ed.tape() as model_tape:
                with ed.interception(make_value_setter(**variational_rv_dict)):
                    y = parametric_ensemble.model(X=X_test,
                                                  base_pred=base_test_pred,
                                                  log_ls_resid=DEFAULT_LOG_LS_RESID)

            log_likelihood = y.distribution.log_prob(y_test)

            # compute the KL divergence
            kl = 0.
            for rv_name, variational_rv in variational_rv_dict.items():
                if _ADD_MFVI_MIXTURE and rv_name == "ensemble_resid":
                    # compute MC approximation
                    param_approx_sample = variational_rv.distribution.sample(_N_INFERENCE_SAMPLE)
                    param_kl = tf.reduce_mean(
                        variational_rv.distribution.log_prob(param_approx_sample) -
                        model_tape[rv_name].distribution.log_prob(param_approx_sample))
                else:
                    # compute analytical form
                    param_kl = variational_rv.distribution.kl_divergence(
                        model_tape[rv_name].distribution)

                kl += tf.reduce_sum(param_kl)

            # define loss op: ELBO = E_q(p(x|z)) + KL(q || p)
            elbo = tf.reduce_mean(log_likelihood - kl)

            # define optimizer
            optimizer = tf.train.AdamOptimizer(5e-3)
            train_op = optimizer.minimize(-elbo)

            # define init op
            init_op = tf.global_variables_initializer()

            vi_graph.finalize()

        """ 3.3. execute optimization, then sample from variational family """

        with tf.Session(graph=vi_graph) as sess:
            start_time = time.time()

            sess.run(init_op)
            for step in range(max_steps):
                _, elbo_value, = sess.run([train_op, elbo])
                if step % 500 == 0:
                    duration = time.time() - start_time
                    print("Step: {:>3d} ELBO: {:.3f}, ({:.3f} min)".format(
                        step, elbo_value, duration / 60.))

            (weight_mean_dict_val, weight_vcov_dict_val,
             resid_gp_mean_val, resid_gp_vcov_val,
             temp_mean_val, temp_sdev_val,
             sigma_mean_val, sigma_sdev_val,
             mixture_par_resid_val) = sess.run([
                weight_mean_dict, weight_vcov_dict,
                resid_gp_mean, resid_gp_vcov,
                temp_mean, temp_sdev,  # temperature variational parameters
                sigma_mean, sigma_sdev, mixture_par_resid])

            sess.close()

        with tf.Session() as sess:
            (weight_sample_dict, temp_sample,
             resid_gp_sample, sigma_sample, _,) = (
                ensemble_variational_family_sample(
                    n_final_sample,
                    weight_mean_dict_val, weight_vcov_dict_val,
                    temp_mean_val, temp_sdev_val,
                    resid_gp_mean_val, resid_gp_vcov_val, mixture_par_resid_val,
                    sigma_mean_val, sigma_sdev_val,
                    log_ls_resid_mean=DEFAULT_LOG_LS_RESID,
                    log_ls_resid_sdev=.01
                ))

            (weight_sample_dict_val, temp_sample_val,
             resid_gp_sample_val, sigma_sample_val) = sess.run([
                weight_sample_dict, temp_sample,
                resid_gp_sample, sigma_sample])

        with open(os.path.join(_SAVE_ADDR_PREFIX,
                               '{}/sigma_sample.pkl'.format(family_name)), 'wb') as file:
            pk.dump(sigma_sample_val, file, protocol=pk.HIGHEST_PROTOCOL)
        with open(os.path.join(_SAVE_ADDR_PREFIX,
                               '{}/temp_sample.pkl'.format(family_name)), 'wb') as file:
            pk.dump(temp_sample_val, file, protocol=pk.HIGHEST_PROTOCOL)
        with open(os.path.join(_SAVE_ADDR_PREFIX,
                               '{}/weight_sample.pkl'.format(family_name)), 'wb') as file:
            pk.dump(weight_sample_dict_val, file, protocol=pk.HIGHEST_PROTOCOL)
        with open(os.path.join(_SAVE_ADDR_PREFIX,
                               '{}/ensemble_resid_sample.pkl'.format(family_name)), 'wb') as file:
            pk.dump(resid_gp_sample_val, file, protocol=pk.HIGHEST_PROTOCOL)

        """ 3.5. prediction and posterior sampling """

        with open(os.path.join(_SAVE_ADDR_PREFIX,
                               '{}/sigma_sample.pkl'.format(family_name)), 'rb') as file:
            sigma_sample_val = pk.load(file)
        with open(os.path.join(_SAVE_ADDR_PREFIX,
                               '{}/temp_sample.pkl'.format(family_name)), 'rb') as file:
            temp_sample_val = pk.load(file)
        with open(os.path.join(_SAVE_ADDR_PREFIX,
                               '{}/weight_sample.pkl'.format(family_name)), 'rb') as file:
            weight_sample_dict_val = pk.load(file)
        with open(os.path.join(_SAVE_ADDR_PREFIX,
                               '{}/ensemble_resid_sample.pkl'.format(family_name)), 'rb') as file:
            resid_gp_sample_val = pk.load(file)

        # compute GP prediction for residual GP
        ensemble_resid_valid_sample = gp.sample_posterior_full(
            X_new=X_valid, X=X_test,
            f_sample=resid_gp_sample_val.T,
            ls=np.exp(DEFAULT_LOG_LS_RESID), kernel_func=gp.rbf).T

        # compute posterior samples for ensemble weight and outcome
        base_weight_sample_val = list(weight_sample_dict_val.values())
        temp_sample_val = np.squeeze(temp_sample_val)
        with tf.Session() as sess:
            W_ensemble = parametric_ensemble.sample_posterior_weight(
                base_weight_sample_val,
                temp_sample_val,
                link_func=sparse_softmax)

            ensemble_mean = parametric_ensemble.sample_posterior_mean(
                base_valid_pred,
                weight_sample=base_weight_sample_val,
                temp_sample=temp_sample_val,
                link_func=sparse_softmax)
            ensemble_mean_val, W_ensemble_val = sess.run([ensemble_mean, W_ensemble])

            ensemble_sample_val = ensemble_mean_val + ensemble_resid_valid_sample

        with open(os.path.join(_SAVE_ADDR_PREFIX,
                               '{}/ensemble_posterior_mean_sample.pkl'.format(family_name)), 'wb') as file:
            pk.dump(ensemble_mean_val, file, protocol=pk.HIGHEST_PROTOCOL)

        with open(os.path.join(_SAVE_ADDR_PREFIX,
                               '{}/ensemble_posterior_dist_sample.pkl'.format(family_name)), 'wb') as file:
            pk.dump(ensemble_sample_val, file, protocol=pk.HIGHEST_PROTOCOL)

    """ 3.5.2. visualize: base prediction """
    with open(os.path.join(_SAVE_ADDR_PREFIX,
                           '{}/ensemble_posterior_mean_sample.pkl'.format(family_name)), 'rb') as file:
        ensemble_mean_val = pk.load(file)

    with open(os.path.join(_SAVE_ADDR_PREFIX,
                           '{}/ensemble_posterior_dist_sample.pkl'.format(family_name)), 'rb') as file:
        ensemble_sample_val = pk.load(file)

    ensemble_model_names = list(base_valid_pred.keys())
    if 'ensemble_model_names' in globals():
        base_pred_dict = {key: value for key, value in base_valid_pred.items()
                          if key in ensemble_model_names}

        visual_util.plot_base_prediction(base_pred=base_pred_dict,
                                         model_names=ensemble_model_names,
                                         X_valid=X_valid, y_valid=y_valid,
                                         X_train=X_train, y_train=y_train,
                                         X_test=X_test, y_test=y_test,
                                         save_addr=os.path.join(
                                             _SAVE_ADDR_PREFIX,
                                             "{}/ensemble_base_model_fit.png".format(family_name)))

        visual_util.plot_base_prediction(base_pred=base_pred_dict,
                                         model_names=ensemble_model_names,
                                         X_valid=X_valid, y_valid=y_valid,
                                         title_size=16, legend_size=12,
                                         save_addr=os.path.join(
                                             _SAVE_ADDR_PREFIX,
                                             "{}/ensemble_base_model_fit_no_data.png".format(family_name)))

    """ 3.5.3. visualize: ensemble posterior predictive mean """

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
                              rmse_id=calib_sample_id,
                              title="Posterior Mean, {}".format(family_name_full),
                              save_addr=os.path.join(
                                  _SAVE_ADDR_PREFIX,
                                  "{}/ensemble_posterior_mean.png".format(family_name))
                              )

    visual_util.gpr_1d_visual(posterior_mean_median,
                              pred_cov=None,
                              pred_quantiles=posterior_mean_quantiles,
                              X_train=X_test, y_train=y_test,
                              X_test=X_valid, y_test=y_valid,
                              rmse_id=calib_sample_id,
                              title="Posterior Mean Quantiles, {}".format(family_name_full),
                              save_addr=os.path.join(
                                  _SAVE_ADDR_PREFIX,
                                  "{}/ensemble_posterior_mean_quantile.png".format(family_name))
                              )

    visual_util.gpr_1d_visual(pred_mean=None, pred_cov=None, pred_quantiles=[],
                              pred_samples=list(ensemble_mean_val)[:2500],
                              X_train=X_test, y_train=y_test,
                              X_test=X_valid, y_test=y_valid,
                              rmse_id=calib_sample_id,
                              title="Posterior Samples, {}".format(family_name_full),
                              save_addr=os.path.join(
                                  _SAVE_ADDR_PREFIX,
                                  "{}/ensemble_posterior_sample.png".format(family_name))
                              )

    """ 3.5.4. visualize: ensemble residual """

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
                              rmse_id=calib_sample_id,
                              title="Posterior Residual, {}".format(family_name_full),
                              save_addr=os.path.join(
                                  _SAVE_ADDR_PREFIX,
                                  "{}/ensemble_posterior_resid.png".format(family_name))
                              )

    visual_util.gpr_1d_visual(posterior_resid_median,
                              pred_cov=None,
                              pred_quantiles=posterior_resid_quantiles,
                              X_train=X_test, y_train=y_test,
                              X_test=X_valid, y_test=y_valid,
                              rmse_id=calib_sample_id,
                              title="Posterior Residual Quantiles, {}".format(family_name_full),
                              save_addr=os.path.join(
                                  _SAVE_ADDR_PREFIX,
                                  "{}/ensemble_posterior_resid_quantile.png".format(family_name))
                              )

    visual_util.gpr_1d_visual(pred_mean=None, pred_cov=None, pred_quantiles=[],
                              pred_samples=list(ensemble_resid_valid_sample)[:2500],
                              X_train=X_test, y_train=y_test,
                              X_test=X_valid, y_test=y_valid,
                              rmse_id=calib_sample_id,
                              title="Posterior Residual Samples, {}".format(family_name_full),
                              save_addr=os.path.join(
                                  _SAVE_ADDR_PREFIX,
                                  "{}/ensemble_posterior_resid_sample.png".format(family_name))
                              )

    """ 3.5.5. visualize: ensemble posterior full """
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
                              rmse_id=calib_sample_id,
                              title="{}".format("Ours"),
                              fontsize=18,
                              save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                     "{}/ensemble_posterior_full.png".format(family_name))
                              )

    visual_util.gpr_1d_visual(posterior_dist_median,
                              pred_cov=None,
                              pred_quantiles=posterior_dist_quantiles,
                              X_train=X_test, y_train=y_test,
                              X_test=X_valid, y_test=y_valid,
                              rmse_id=calib_sample_id,
                              title="Posterior Predictive Quantiles, {}".format(family_name_full),
                              save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                     "{}/ensemble_posterior_full_quantile.png".format(family_name))
                              )
    visual_util.gpr_1d_visual(pred_mean=None, pred_cov=None, pred_quantiles=[],
                              pred_samples=list(ensemble_sample_val)[:2500],
                              X_train=X_test, y_train=y_test,
                              X_test=X_valid, y_test=y_valid,
                              rmse_id=calib_sample_id,
                              title="Posterior Predictive Samples, {}".format(family_name_full),
                              save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                     "{}/ensemble_posterior_full_sample.png".format(family_name))
                              )

    visual_util.gpr_1d_visual(pred_mean=posterior_dist_median,
                              pred_cov=None,
                              pred_quantiles=[
                                  [posterior_dist_median - 3 * np.sqrt(posterior_mean_cov),
                                   posterior_dist_median + 3 * np.sqrt(posterior_mean_cov)],
                                  [posterior_dist_median - 3 * np.sqrt(posterior_dist_cov),
                                   posterior_dist_median + 3 * np.sqrt(posterior_dist_cov)],
                              ],
                              X_train=X_test, y_train=y_test,
                              X_test=X_valid, y_test=y_valid,
                              rmse_id=calib_sample_id,
                              quantile_colors=["red", "black"],
                              quantile_alpha=0.2,
                              title="Posterior, Uncertainty Decomposition, {}".format(family_name_full),
                              save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                     "{}/ensemble_posterior_unc_decomp.png".format(family_name))
                              )

    """ 3.5.6. visualize: ensemble posterior reliability """
    y_calib = y_valid[calib_sample_id]
    y_sample_calib = ensemble_sample_val[:, calib_sample_id].T

    visual_util.prob_calibration_1d(
        y_calib, y_sample_calib,
        title="Ensemble, {}".format(family_name_full),
        save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                               "{}/ensemble_calibration_prob.png".format(family_name)))

    visual_util.coverage_index_1d(
        y_calib, y_sample_calib,
        title="Ensemble, {}".format(family_name_full),
        save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                               "{}/ensemble_credible_coverage.png".format(family_name)))


"""""""""""""""""""""""""""""""""
# 7. Nonparametric Calibration II: Monotonic GP
"""""""""""""""""""""""""""""""""
# TODO(jereliu): separate ls for main gp and derivative gp
# TODO(jereliu): Visualize additional calibration uncertainty.
# TODO(jereliu): Urgent: alternative method I: sample only posterior predictive mean
# TODO(jereliu): Urgent: alternative method II: pre-compute matrix in np, then convert to tensor

ADD_CONSTRAINT_POINT = True

""" 7.1. prepare hyperparameters """

family_names = []

if _FIT_MCMC_MODELS:
    family_names += ["hmc"]

if _FIT_VI_MODELS:
    family_names += [
        "mfvi", "sgpr", "dgpr",
        # "mfvi_mfvi_5_mix", "sgpr_mfvi_5_mix", "dgpr_mfvi_5_mix",
    ]

if _FIT_AUG_VI_MODELS:
    family_names += ["mfvi_aug", "sgpr_aug", "mfvi_crps", "sgpr_crps"]

# load hyper-parameters
for family_name in family_names:

    family_name_root = family_name.split("_")[0]
    family_name_full = {"hmc": "Hamilton MC",
                        "mfvi": "Mean-field VI",
                        "sgpr": "Sparse Gaussian Process",
                        "dgpr": "Decoupled Gaussian Process"
                        }[family_name_root]

    os.makedirs("{}/{}".format(_SAVE_ADDR_PREFIX, family_name), exist_ok=True)

    # load estimates
    with open(os.path.join(_SAVE_ADDR_PREFIX,
                           '{}/ensemble_posterior_dist_sample.pkl'.format(family_name)), 'rb') as file:
        ensemble_sample_val = pk.load(file)

    """ 7.2. build calibration dataset """
    sample_size = ensemble_sample_val.shape[0]
    calib_train_id = np.random.choice(
        calib_sample_id, int(calib_sample_id.size / 2), replace=False)
    calib_test_id = np.asarray(list(set(calib_sample_id) -
                                    set(calib_train_id)))

    y_calib_train = y_valid[calib_train_id]
    y_calib_test = y_valid[calib_test_id]
    y_calib_sample_train = ensemble_sample_val[:, calib_train_id].T
    y_calib_sample_test = ensemble_sample_val[:, calib_test_id].T

    calib_data_train = calib_util.build_calibration_dataset(Y_obs=y_calib_train,
                                                            Y_sample=y_calib_sample_train)
    calib_data_test = calib_util.build_calibration_dataset(Y_obs=y_calib_test,
                                                           Y_sample=y_calib_sample_test)

    with tf.Session() as sess:
        (orig_prob_train, calib_prob_train,
         orig_prob_test, calib_prob_test) = sess.run(
            [calib_data_train["feature"], calib_data_train["label"],
             calib_data_test["feature"], calib_data_test["label"]])

    # plt.plot([[0., 0.], [1., 1.]])
    # sns.regplot(orig_prob_train, calib_prob_train, fit_reg=False)

    if ADD_CONSTRAINT_POINT:
        # add constraint points at [0,0] and [1,1]
        orig_prob_train = np.concatenate([orig_prob_train,
                                          [0., 1.]]).astype(np.float32)
        calib_prob_train = np.concatenate([calib_prob_train,
                                           [0., 1.]]).astype(np.float32)

    # prepare data
    N = orig_prob_train.size
    N_pred = 1000
    N_deriv = 100

    orig_prob_pred = np.linspace(0, 1, num=N_pred).astype(np.float32)
    orig_prob_derv = np.linspace(0, 1, num=N_deriv).astype(np.float32)

    orig_prob_train = np.atleast_2d(orig_prob_train).T
    orig_prob_derv = np.atleast_2d(orig_prob_derv).T
    calib_prob_train = calib_prob_train
    orig_prob_pred = np.atleast_2d(orig_prob_pred).T

    """ 7.3. fit monotonic GP"""
    # default parameters and data
    DEFAULT_LS_CALIB_VAL = np.asarray(0.2, np.float32)
    DEFAULT_DERIV_CDF_SCALE = np.asarray(1e-3, np.float32)

    if _FIT_CALIB_MODELS:
        """ 7.3.1. define mcmc computation graph"""
        num_results = 5000
        num_burnin_steps = 5000

        # # pre-compute model parameters for f^* | f_obs, f_deriv
        # pred_cond_pars = gpr_mono.compute_pred_cond_params(
        #     X_new=orig_prob_pred,
        #     X_obs=orig_prob_train,
        #     X_deriv=orig_prob_derv,
        #     ls=DEFAULT_LS_CALIB_VAL,
        #     kernel_func_ff=gp.rbf,
        #     kernel_func_df=gpr_mono.rbf_grad_1d,
        #     kernel_func_dd=gpr_mono.rbf_hess_1d,
        #     ridge_factor_K=1e-1,
        #     ridge_factor_Sigma=0.)
        #
        # Sigma = pred_cond_pars[2]
        # # Sigma[np.abs(Sigma) < 1e-1] = 0
        # Sigma_chol = np.linalg.cholesky(Sigma + 1e-3 * np.eye(Sigma.shape[0]))
        # pred_cond_pars = list(pred_cond_pars)
        # pred_cond_pars[2] = Sigma_chol.astype(np.float32)

        mcmc_graph = tf.Graph()
        with mcmc_graph.as_default():
            # build likelihood explicitly
            target_log_prob_fn = gpr_mono.make_log_likelihood_function(
                X_train=orig_prob_train,
                X_deriv=orig_prob_derv,
                # X_pred=orig_prob_pred,
                y_train=calib_prob_train,
                ls=DEFAULT_LS_CALIB_VAL,
                # pred_cond_pars=pred_cond_pars,
                deriv_prior_scale=DEFAULT_DERIV_CDF_SCALE,
                ridge_factor=5e-2,
                cdf_constraint=True
            )

            # set up state container
            initial_state = [
                tf.random_normal([N],
                                 mean=1., stddev=0.01,
                                 dtype=tf.float32,
                                 name='init_gp_func'),
                # tf.random_normal([N_pred],
                #                  mean=1., stddev=0.01,
                #                  dtype=tf.float32,
                #                  name='init_gp_pred'),
                tf.random_normal([N_deriv],
                                 mean=1., stddev=0.01,
                                 dtype=tf.float32,
                                 name='init_gp_derv'),
                tf.constant(0.1, dtype=tf.float32,
                            name='init_sigma'),
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

            # gpf_sample, gpf_pred_sample, gpf_deriv_sample, sigma_sample, = state
            gpf_sample, gpf_deriv_sample, sigma_sample, = state

            # set up init op
            init_op = tf.global_variables_initializer()

            mcmc_graph.finalize()

        """ 7.3.2. execute sampling"""
        time_start = time.time()
        with tf.Session(graph=mcmc_graph) as sess:
            init_op.run()
            [
                f_samples_val,
                # f_pred_samples_val,
                f_deriv_samples_val,
                sigma_sample_val,
                is_accepted_,
            ] = sess.run(
                [
                    gpf_sample,
                    # gpf_pred_sample,
                    gpf_deriv_sample,
                    sigma_sample,
                    kernel_results.is_accepted,
                ])
            total_min = (time.time() - time_start) / 60.

            print('Acceptance Rate: {}'.format(np.mean(is_accepted_)))
            print('Total time: {:.2f} min'.format(total_min))
            sess.close()

        # plt.plot(orig_prob_train, np.mean(f_samples_val, 0), 'o')
        # plt.plot(orig_prob_pred, np.mean(f_pred_samples_val, 0), 'o')

        """ 7.3.3. prediction and visualization"""
        # prediction
        df_pred_val = gp.sample_posterior_full(X_new=orig_prob_pred,
                                               X=orig_prob_derv,
                                               f_sample=f_deriv_samples_val.T,
                                               ls=DEFAULT_LS_CALIB_VAL,
                                               kernel_func=gpr_mono.rbf_hess_1d)

        calib_prob_pred_val = gp.sample_posterior_full(X_new=orig_prob_pred,
                                                       X=orig_prob_train,
                                                       f_sample=f_samples_val.T,
                                                       ls=DEFAULT_LS_CALIB_VAL,
                                                       kernel_func=gp.rbf)

        # # sample f conditional on f_deriv
        # calib_prob_pred_val = (
        #     gpr_mono.sample_posterior_predictive(X_new=orig_prob_pred,
        #                                          X_obs=orig_prob_train,
        #                                          X_deriv=orig_prob_derv,
        #                                          f_sample=f_samples_val.T,
        #                                          f_deriv_sample=f_deriv_samples_val.T,
        #                                          kernel_func_ff=gp.rbf,
        #                                          kernel_func_df=gpr_mono.rbf_grad_1d,
        #                                          kernel_func_dd=gpr_mono.rbf_hess_1d,
        #                                          ls=DEFAULT_LS_CALIB_VAL, )
        # )

        # plt.plot([[0., 0.], [1., 1.]])
        # sns.regplot(orig_prob_train.squeeze(), calib_prob_train,
        #             fit_reg=False, scatter_kws={'color': 'green'})
        #
        # sns.regplot(orig_prob_pred.squeeze(), np.mean(calib_prob_pred, -1),
        #             fit_reg=False, marker='+',
        #             scatter_kws={'color': 'red'})
        # plt.show()

        mu = np.mean(calib_prob_pred_val, axis=1)
        mu_deriv = np.mean(df_pred_val, axis=1)
        cov = np.var(calib_prob_pred_val, axis=1)
        cov_deriv = np.var(df_pred_val, axis=1)

        visual_util.gpr_1d_visual(mu, cov,
                                  X_train=orig_prob_train,
                                  y_train=calib_prob_train,
                                  X_test=orig_prob_pred,
                                  y_test=None,
                                  title="RBF Calibration Fit, Hamilton MC",
                                  save_addr=os.path.join(
                                      _SAVE_ADDR_PREFIX,
                                      "{}/calibration/gpr_calib_fit.png".format(family_name)),
                                  y_range=[-0.1, 1.1])

        visual_util.gpr_1d_visual(mu_deriv, cov_deriv,
                                  X_test=orig_prob_pred,
                                  title="RBF Derivative, Hamilton MC",
                                  save_addr=os.path.join(
                                      _SAVE_ADDR_PREFIX,
                                      "{}/calibration/gpr_calib_deriv.png".format(family_name)),
                                  add_reference=True, y_range=None)

        visual_util.gpr_1d_visual(pred_mean=None, pred_cov=None, pred_quantiles=[],
                                  pred_samples=list(calib_prob_pred_val.T)[:500],
                                  X_train=orig_prob_train,
                                  y_train=calib_prob_train,
                                  X_test=orig_prob_pred,
                                  y_test=None,
                                  title="RBF Calibration Samples, {}".format(family_name_full),
                                  save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                         "{}/calibration/gpr_calib_sample.png".format(
                                                             family_name)),
                                  y_range=[-0.1, 1.1])

        """ 7.4. produce calibrated posterior sample"""
        # re-sample observations based on newly obtained cdf
        calib_prob_pred = np.mean(calib_prob_pred_val, axis=1)
        calib_prob_pred[calib_prob_pred > 1.] = 1.
        calib_prob_pred[calib_prob_pred < 0.] = 0.

        ensemble_sample_calib_val = [
            calib_util.sample_ecdf(n_sample=1000,
                                   base_sample=base_sample,
                                   quantile=calib_prob_pred) for
            base_sample in ensemble_sample_val.T
        ]
        ensemble_sample_calib_val = np.asarray(ensemble_sample_calib_val).T

        with open(os.path.join(_SAVE_ADDR_PREFIX,
                               '{}/calibration/ensemble_posterior_dist_sample.pkl'.format(family_name)),
                  'wb') as file:
            pk.dump(ensemble_sample_calib_val, file, protocol=pk.HIGHEST_PROTOCOL)

    # # plot original vs calibrated cdf
    # sample_id = 50
    # plt.plot(np.sort(ensemble_sample_val[:, sample_id]),
    #          orig_prob_pred)
    # plt.plot(np.sort(ensemble_sample_val[:, sample_id]),
    #          calib_prob_pred)

    """ 7.5.1. visualize: ensemble posterior reliability """
    with open(os.path.join(_SAVE_ADDR_PREFIX,
                           '{}/calibration/ensemble_posterior_dist_sample.pkl'.format(family_name)), 'rb') as file:
        ensemble_sample_calib_val = pk.load(file)

    os.makedirs(os.path.join(_SAVE_ADDR_PREFIX,
                             "{}/calibration/".format(family_name)),
                exist_ok=True)

    posterior_dist_mu = np.nanmean(ensemble_sample_calib_val, axis=0)
    posterior_dist_cov = np.nanvar(ensemble_sample_calib_val, axis=0)

    posterior_dist_median = np.nanmedian(ensemble_sample_calib_val, axis=0)
    posterior_dist_quantiles = [
        np.percentile(ensemble_sample_calib_val,
                      [100 - (100 - q) / 2, (100 - q) / 2], axis=0)
        for q in [68, 95, 99, 99.9]
    ]

    visual_util.gpr_1d_visual(posterior_dist_mu,
                              pred_cov=posterior_dist_cov,
                              X_train=X_test, y_train=y_test,
                              X_test=X_valid, y_test=y_valid,
                              rmse_id=calib_sample_id,
                              title="Posterior Predictive, {}".format(family_name_full),
                              save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                     "{}/calibration/ensemble_posterior_full_gpr.png".format(
                                                         family_name))
                              )

    visual_util.gpr_1d_visual(posterior_dist_median,
                              pred_cov=None,
                              pred_quantiles=posterior_dist_quantiles,
                              X_train=X_test, y_train=y_test,
                              X_test=X_valid, y_test=y_valid,
                              rmse_id=calib_sample_id,
                              title="Posterior Predictive Quantiles, {}".format(family_name_full),
                              save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                     "{}/calibration/ensemble_posterior_full_quantile_gpr.png".format(
                                                         family_name))
                              )
    visual_util.gpr_1d_visual(pred_mean=None, pred_cov=None, pred_quantiles=[],
                              pred_samples=list(ensemble_sample_calib_val)[:150],
                              X_train=X_test, y_train=y_test,
                              X_test=X_valid, y_test=y_valid,
                              rmse_id=calib_sample_id,
                              X_induce=X_valid[calib_train_id],
                              title="Posterior Predictive Samples, {}".format(family_name_full),
                              save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                     "{}/calibration/ensemble_posterior_full_sample_gpr.png".format(
                                                         family_name))
                              )

    """ 7.5.2. visualize: ensemble posterior reliability """
    # training
    y_calib = y_valid[calib_train_id]
    y_sample_calib = ensemble_sample_calib_val[:, calib_train_id].T

    visual_util.prob_calibration_1d(
        y_calib, y_sample_calib,
        title="Train, {}".format(family_name_full),
        save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                               "{}/calibration/gpr_calibration_prob_train.png".format(family_name)))

    visual_util.coverage_index_1d(
        y_calib, y_sample_calib,
        title="Train, {}".format(family_name_full),
        save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                               "{}/calibration/gpr_credible_coverage_train.png".format(family_name)))

    # testing
    y_calib = y_valid[calib_test_id]
    y_sample_calib = ensemble_sample_calib_val[:, calib_test_id].T

    visual_util.prob_calibration_1d(
        y_calib, y_sample_calib,
        title="Test, {}".format(family_name_full),
        save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                               "{}/calibration/gpr_calibration_prob_test.png".format(family_name)))

    visual_util.coverage_index_1d(
        y_calib, y_sample_calib,
        title="Test, {}".format(family_name_full),
        save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                               "{}/calibration/gpr_credible_coverage_test.png".format(family_name)))

    # overall
    y_calib = y_valid[calib_sample_id]
    y_sample_calib = ensemble_sample_calib_val[:, calib_sample_id].T

    visual_util.prob_calibration_1d(
        y_calib, y_sample_calib,
        title="Calibrated, {}".format(family_name_full),
        save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                               "{}/calibration/gpr_calibration_prob_all.png".format(family_name)))

    visual_util.coverage_index_1d(
        y_calib, y_sample_calib,
        title="Calibrated, {}".format(family_name_full),
        save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                               "{}/calibration/gpr_credible_coverage_all.png".format(family_name)))
