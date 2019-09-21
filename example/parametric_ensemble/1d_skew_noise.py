"""Calibre (Parametric Ensemble) with flat model structure. """
import os
import sys
import time

from scipy import interpolate

import functools
from importlib import reload

import pickle as pk
import pandas as pd

import numpy as np
import scipy.stats as stats

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

import calibre.util.data as data_util
import calibre.util.metric as metric_util
import calibre.util.visual as visual_util
import calibre.util.gp_flow as gp_util
import calibre.util.calibration as calib_util
import calibre.util.experiment_data as experiment_util

from calibre.util.model import sparse_softmax
from calibre.util.inference import make_value_setter

import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

tfd = tfp.distributions

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DEFAULT_LOG_LS_RESID = np.log(0.2).astype(np.float32)

_SAVE_ADDR_PREFIX = "./result/parametric_ensemble/1d_skew_noise"
_FIT_BASE_MODELS = False
_FIT_VI_MODELS = False
_FIT_AUG_VI_MODELS = False
_FIT_CALIB_MODELS = True
_PLOT_LOCAL_CDF_COMPARISON = False

"""""""""""""""""""""""""""""""""
# 1. Generate data
"""""""""""""""""""""""""""""""""
# data with skewed (weibull (a=1.1)) noise
N_train = 20
N_test = 80
N_calib = 100
N_valid = 500

data_gen_func = data_util.sin_cos_curve_skew_noise_1d
data_gen_func_x = data_util.gaussian_mix
data_gen_func_x_test = functools.partial(data_util.gaussian_mix, sd_scale=2.)

(X_train, y_train,
 X_test, y_test,
 X_valid, y_valid_sample, calib_sample_id) = experiment_util.generate_data_1d(
    N_train=N_train, N_test=N_test, N_calib=N_calib, N_valid=N_valid,
    noise_sd=None,
    data_gen_func=data_gen_func,
    data_gen_func_x=data_gen_func_x,
    data_gen_func_x_test=data_gen_func_x_test,
    data_range=(-6., 6.), valid_range=(-6., 6.),
    seed_train=1000, seed_test=2500, seed_calib=500)

y_valid_mean = np.mean(y_valid_sample, axis=1)
y_valid = y_valid_sample[:, 0]

# # compute data mode
# y_valid_eval = np.linspace(np.min(y_valid_sample), np.max(y_valid_sample), 1000)
# y_valid_map = [y_valid_eval[np.argmax(stats.gaussian_kde(sample)(y_valid_eval))]
#                for sample in y_valid_sample]


# plot ground truth
plt.figure(figsize=(12, 6))
plt.scatter(np.repeat(X_valid, 100),
            y_valid_sample[:, :100].flatten(), marker=".", s=1)
plt.savefig("{}/data".format(_SAVE_ADDR_PREFIX))
plt.close()

plt.scatter(X_train, y_train, marker="o", s=5.)
plt.savefig("{}/data_train".format(_SAVE_ADDR_PREFIX))
plt.close()

plt.scatter(X_test, y_test, marker="o", s=5.)
plt.savefig("{}/data_test".format(_SAVE_ADDR_PREFIX))
plt.close()

# # inspect data's empirical cdf
# ecdf_val, ecdf_sample = metric_util.local_ecdf_1d(x_eval=[0],
#                                                   X=X_train, y=y_train,
#                                                   eval_window=0.05,
#                                                   return_sample=True)
# plt.plot(ecdf_val[0])
# sns.distplot(ecdf_sample)

""" 1.1. Build base GP models using GPflow """
if _FIT_BASE_MODELS:
    gp_util.fit_base_gp_models(X_train, y_train,
                               X_test, y_test,
                               X_valid, y_valid_mean,
                               kern_func_dict=gp_util.DEFAULT_KERN_FUNC_DICT_GPY,
                               n_valid_sample=1000,
                               save_addr_prefix="{}/base".format(_SAVE_ADDR_PREFIX),
                               y_range=[-2.5, 2.5])

"""""""""""""""""""""""""""""""""
# 2. Fit data
"""""""""""""""""""""""""""""""""

_ADD_MFVI_MIXTURE = False
_N_INFERENCE_SAMPLE = 20
_N_MFVI_MIXTURE = 5

with open(os.path.join(_SAVE_ADDR_PREFIX, 'base/base_test_pred.pkl'), 'rb') as file:
    base_test_pred = pk.load(file)

with open(os.path.join(_SAVE_ADDR_PREFIX, 'base/base_valid_pred.pkl'), 'rb') as file:
    base_valid_pred = pk.load(file)

model_names = [  # "poly_3",
    "rbf_0.25", "rbf_1", "period1.5"]
base_test_pred = {key: value for key, value in base_test_pred.items() if
                  (key in model_names)}
base_valid_pred = {key: value for key, value in base_valid_pred.items()
                   if key in model_names}

""" 3.1. basic data/algorithm config"""

n_final_sample = 5000  # number of samples to collect from variational family
max_steps = 20000  # number of training iterations

X_induce_mean = KMeans(n_clusters=int(N_test * 0.9), random_state=100).fit(
    X_test).cluster_centers_.astype(np.float32)
X_induce = KMeans(n_clusters=int(N_test * 0.5), random_state=100).fit(
    X_test).cluster_centers_.astype(np.float32)

# X_induce_mean = np.expand_dims(np.linspace(np.min(X_test),
#                                            np.max(X_test), 50), 1).astype(np.float32)
# X_induce = np.expand_dims(np.linspace(np.min(X_test),
#                                       np.max(X_test), 20), 1).astype(np.float32)

family_name_list = [  # "mfvi",
    # "sgpr",
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

        """ 3.4. prediction and posterior sampling """

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

        # compute residual noise
        ensemble_noise_valid_sample = np.random.normal(
            loc=0, scale=np.exp(np.mean(sigma_sample_val)),
            size=ensemble_resid_valid_sample.shape)

        # compute posterior samples for ensemble weight and outcome
        base_weight_sample_val = list(weight_sample_dict_val.values())
        temp_sample_val = np.squeeze(temp_sample_val)
        with tf.Session() as sess:
            #
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

            ensemble_mean_corrected_val = ensemble_mean_val + ensemble_resid_valid_sample
            ensemble_sample_val = ensemble_mean_corrected_val + ensemble_noise_valid_sample

        with open(os.path.join(_SAVE_ADDR_PREFIX,
                               '{}/ensemble_posterior_mean_sample.pkl'.format(family_name)), 'wb') as file:
            pk.dump(ensemble_mean_val, file, protocol=pk.HIGHEST_PROTOCOL)

        with open(os.path.join(_SAVE_ADDR_PREFIX,
                               '{}/ensemble_posterior_mean_corrected_sample.pkl'.format(family_name)), 'wb') as file:
            pk.dump(ensemble_mean_corrected_val, file, protocol=pk.HIGHEST_PROTOCOL)

        with open(os.path.join(_SAVE_ADDR_PREFIX,
                               '{}/ensemble_posterior_dist_sample.pkl'.format(family_name)), 'wb') as file:
            pk.dump(ensemble_sample_val, file, protocol=pk.HIGHEST_PROTOCOL)

    """ 3.5. visualization """

    """ 3.5.1. visualize: base prediction """
    with open(os.path.join(_SAVE_ADDR_PREFIX,
                           '{}/ensemble_posterior_mean_sample.pkl'.format(family_name)), 'rb') as file:
        ensemble_mean_val = pk.load(file)

    with open(os.path.join(_SAVE_ADDR_PREFIX,
                           '{}/ensemble_posterior_mean_corrected_sample.pkl'.format(family_name)), 'rb') as file:
        ensemble_mean_corrected_val = pk.load(file)

    with open(os.path.join(_SAVE_ADDR_PREFIX,
                           '{}/ensemble_posterior_dist_sample.pkl'.format(family_name)), 'rb') as file:
        ensemble_sample_val = pk.load(file)

    ensemble_resid_valid_sample = ensemble_mean_corrected_val - ensemble_mean_val

    ensemble_model_names = list(base_valid_pred.keys())
    if 'ensemble_model_names' in globals():
        base_pred_dict = {key: value for key, value in base_valid_pred.items()
                          if key in ensemble_model_names}

        visual_util.plot_base_prediction(base_pred=base_pred_dict,
                                         model_names=ensemble_model_names,
                                         X_valid=X_valid, y_valid=y_valid_mean,
                                         X_train=X_train, y_train=y_train,
                                         X_test=X_test, y_test=y_test,
                                         y_range=[-2.5, 2.5],
                                         save_addr=os.path.join(
                                             _SAVE_ADDR_PREFIX,
                                             "{}/ensemble_base_model_fit.png".format(family_name)))

        visual_util.plot_base_prediction(base_pred=base_pred_dict,
                                         model_names=ensemble_model_names,
                                         X_valid=X_valid, y_valid=y_valid_mean,
                                         title_size=16, legend_size=12,
                                         y_range=[-2.5, 2.5],
                                         save_addr=os.path.join(
                                             _SAVE_ADDR_PREFIX,
                                             "{}/ensemble_base_model_fit_no_data.png".format(family_name)))

    """ 3.5.2. visualize: ensemble posterior predictive mean """

    posterior_mean_mu = np.nanmean(ensemble_mean_val, axis=0)
    posterior_mean_cov = np.nanvar(ensemble_mean_val, axis=0)
    posterior_resid_cov = np.nanvar(ensemble_resid_valid_sample, axis=0)

    posterior_dist_cov = np.nanvar(ensemble_sample_val, axis=0) - posterior_resid_cov

    posterior_mean_median = np.nanmedian(ensemble_mean_val, axis=0)
    posterior_mean_quantiles = [
        np.percentile(ensemble_mean_val,
                      [100 - (100 - q) / 2, (100 - q) / 2], axis=0)
        for q in [68, 95, 99]
    ]

    visual_util.gpr_1d_visual(posterior_mean_mu,
                              pred_cov=posterior_mean_cov,
                              X_train=X_test, y_train=y_test,
                              X_test=X_valid, y_test=y_valid_mean,
                              rmse_id=calib_sample_id,
                              title="Posterior Mean, {}".format(family_name_full),
                              save_addr=os.path.join(
                                  _SAVE_ADDR_PREFIX,
                                  "{}/ensemble_posterior_mean.png".format(family_name)),
                              y_range=[-2.5, 2.5]
                              )

    visual_util.gpr_1d_visual(posterior_mean_median,
                              pred_cov=None,
                              pred_quantiles=posterior_mean_quantiles,
                              X_train=X_test, y_train=y_test,
                              X_test=X_valid, y_test=y_valid_mean,
                              rmse_id=calib_sample_id,
                              title="Posterior Mean Quantiles, {}".format(family_name_full),
                              save_addr=os.path.join(
                                  _SAVE_ADDR_PREFIX,
                                  "{}/ensemble_posterior_mean_quantile.png".format(family_name)),
                              y_range=[-2.5, 2.5]
                              )

    visual_util.gpr_1d_visual(posterior_mean_mu,
                              pred_quantiles=[
                                  (posterior_mean_mu + 3 * np.sqrt(posterior_dist_cov),
                                   posterior_mean_mu - 3 * np.sqrt(posterior_dist_cov)),
                                  (posterior_mean_mu + 3 * np.sqrt(posterior_mean_cov),
                                   posterior_mean_mu - 3 * np.sqrt(posterior_mean_cov)),
                              ],
                              quantile_colors=[
                                  visual_util.UNC_COLOR_PALETTE["alea"],
                                  visual_util.UNC_COLOR_PALETTE["para"],
                              ],
                              quantile_alpha=.4,
                              X_train=X_test, y_train=y_test,
                              X_test=X_valid, y_test=y_valid_mean,
                              rmse_id=calib_sample_id,
                              title="Uncertainty Decomposition, Original Model",
                              fontsize=18,
                              quantile_shade_legend=["Aleatoric",
                                                     "Parametric"],
                              save_addr=os.path.join(
                                  _SAVE_ADDR_PREFIX,
                                  "{}/ensemble_posterior_unc_decomp_mean.png".format(family_name)),
                              y_range=[-2.5, 2.5],
                              figsize=(12, 6)
                              )

    visual_util.gpr_1d_visual(pred_mean=None, pred_cov=None, pred_quantiles=[],
                              pred_samples=list(ensemble_mean_val)[:2500],
                              X_train=X_test, y_train=y_test,
                              X_test=X_valid, y_test=y_valid_mean,
                              rmse_id=calib_sample_id,
                              title="Posterior Samples, {}".format(family_name_full),
                              save_addr=os.path.join(
                                  _SAVE_ADDR_PREFIX,
                                  "{}/ensemble_posterior_sample.png".format(family_name)),
                              y_range=[-2.5, 2.5]
                              )

    """ 3.5.3. visualize: ensemble residual """

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
                              X_test=X_valid, y_test=y_valid_mean,
                              rmse_id=calib_sample_id,
                              title="Posterior Residual, {}".format(family_name_full),
                              save_addr=os.path.join(
                                  _SAVE_ADDR_PREFIX,
                                  "{}/ensemble_posterior_resid.png".format(family_name)),
                              y_range=[-2.5, 2.5]
                              )

    visual_util.gpr_1d_visual(posterior_resid_median,
                              pred_cov=None,
                              pred_quantiles=posterior_resid_quantiles,
                              X_train=X_test, y_train=y_test,
                              X_test=X_valid, y_test=y_valid_mean,
                              rmse_id=calib_sample_id,
                              title="Posterior Residual Quantiles, {}".format(family_name_full),
                              save_addr=os.path.join(
                                  _SAVE_ADDR_PREFIX,
                                  "{}/ensemble_posterior_resid_quantile.png".format(family_name)),
                              y_range=[-2.5, 2.5]
                              )

    visual_util.gpr_1d_visual(pred_mean=None, pred_cov=None, pred_quantiles=[],
                              pred_samples=list(ensemble_resid_valid_sample)[:2500],
                              X_train=X_test, y_train=y_test,
                              X_test=X_valid, y_test=y_valid_mean,
                              rmse_id=calib_sample_id,
                              title="Posterior Residual Samples, {}".format(family_name_full),
                              save_addr=os.path.join(
                                  _SAVE_ADDR_PREFIX,
                                  "{}/ensemble_posterior_resid_sample.png".format(family_name)),
                              y_range=[-2.5, 2.5]
                              )

    """ 3.5.4. visualize: ensemble posterior full """
    posterior_dist_mu = np.nanmean(ensemble_sample_val, axis=0)
    posterior_dist_cov = np.nanvar(ensemble_sample_val, axis=0)

    posterior_mean_adj_cov = np.nanvar(ensemble_mean_corrected_val, axis=0)

    posterior_dist_median = np.nanmedian(ensemble_sample_val, axis=0)
    posterior_dist_quantiles = [
        np.percentile(ensemble_sample_val,
                      [100 - (100 - q) / 2, (100 - q) / 2], axis=0)
        for q in [68, 95, 99]
    ]

    visual_util.gpr_1d_visual(posterior_dist_mu,
                              pred_cov=posterior_dist_cov,
                              X_train=X_test, y_train=y_test,
                              X_test=X_valid, y_test=y_valid_mean,
                              rmse_id=calib_sample_id,
                              title="{}".format("Ours"),
                              fontsize=18,
                              save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                     "{}/ensemble_posterior_full.png".format(family_name)),
                              y_range=[-2.5, 2.5]
                              )

    visual_util.gpr_1d_visual(posterior_dist_median,
                              pred_cov=None,
                              pred_quantiles=posterior_dist_quantiles,
                              X_train=X_test, y_train=y_test,
                              X_test=X_valid, y_test=y_valid_mean,
                              rmse_id=calib_sample_id,
                              title="Posterior Predictive Quantiles, {}".format(family_name_full),
                              save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                     "{}/ensemble_posterior_full_quantile.png".format(family_name)),
                              y_range=[-2.5, 2.5]
                              )
    visual_util.gpr_1d_visual(pred_mean=None, pred_cov=None, pred_quantiles=[],
                              pred_samples=list(ensemble_sample_val)[:2500],
                              X_train=X_test, y_train=y_test,
                              X_test=X_valid, y_test=y_valid_mean,
                              rmse_id=calib_sample_id,
                              title="Posterior Predictive Samples, {}".format(family_name_full),
                              save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                     "{}/ensemble_posterior_full_sample.png".format(family_name)),
                              y_range=[-2.5, 2.5]
                              )

    visual_util.gpr_1d_visual(pred_mean=posterior_dist_median,
                              pred_cov=None,
                              pred_quantiles=[
                                  [posterior_dist_median - 3 * np.sqrt(posterior_dist_cov),
                                   posterior_dist_median + 3 * np.sqrt(posterior_dist_cov)],
                                  [posterior_dist_median - 3 * np.sqrt(posterior_mean_adj_cov),
                                   posterior_dist_median + 3 * np.sqrt(posterior_mean_adj_cov)],
                                  [posterior_dist_median - 3 * np.sqrt(posterior_mean_cov),
                                   posterior_dist_median + 3 * np.sqrt(posterior_mean_cov)],

                              ],
                              X_train=X_test, y_train=y_test,
                              X_test=X_valid, y_test=y_valid_mean,
                              rmse_id=None,
                              quantile_colors=[
                                  visual_util.UNC_COLOR_PALETTE["alea"],
                                  visual_util.UNC_COLOR_PALETTE["str_system"],
                                  visual_util.UNC_COLOR_PALETTE["para"],
                              ],
                              quantile_shade_legend=["Aleatoric",
                                                     "Structural, System Comp",
                                                     "Parametric"],
                              quantile_alpha=0.4,
                              title="Uncertainty Decomposition, System Comp Only",
                              fontsize=18,
                              save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                     "{}/ensemble_posterior_unc_decomp.png".format(family_name)),
                              y_range=[-2.5, 2.5],
                              figsize=(12, 6)
                              )

    """ 3.5.5. visualize: impact of model mis-specification """
    posterior_resid_mu = np.nanmean(ensemble_resid_valid_sample, axis=0)
    posterior_resid_cov = np.nanvar(ensemble_resid_valid_sample, axis=0)

    posterior_resid_median = np.nanmedian(ensemble_resid_valid_sample, axis=0)

    posterior_resid_zero_quantile = np.nanmean(
        ensemble_resid_valid_sample > 0, axis=0).flatten()
    posterior_resid_quantiles = [
        np.percentile(ensemble_resid_valid_sample,
                      [100 - (100 - q) / 2, (100 - q) / 2], axis=0)
        for q in [68, 95, 99]
    ]

    # interpolate
    f = interpolate.interp1d(x=X_valid.flatten(),
                             y=posterior_resid_zero_quantile,
                             kind='cubic')
    X_valid_new = np.linspace(np.min(X_valid),
                              np.max(X_valid), 2000)
    posterior_resid_zero_quantile = f(X_valid_new)

    posterior_resid_zero_quantile_abs = np.abs(
        posterior_resid_zero_quantile - 0.5
    )

    # mean misspecification, posterior belief
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    visual_util.gpr_1d_visual(posterior_resid_zero_quantile_abs * 2,
                              pred_cov=None,
                              pred_quantiles=[[
                                  np.zeros(posterior_resid_zero_quantile_abs.size),
                                  posterior_resid_zero_quantile_abs * 2
                              ]],
                              X_train=None, y_train=None,
                              X_test=X_valid_new, y_test=None,
                              rmse_id=None,
                              title=r"Error in Predictive Mean due to System Comp Misspecification",
                              fontsize=18,
                              save_addr=None,
                              y_range=[0., 1.1],
                              quantile_colors=posterior_resid_zero_quantile,
                              quantile_colors_norm=visual_util.SIGNIFICANT_NORM,
                              pred_mean_color='white',
                              pred_mean_alpha=0.,
                              smooth_quantile=True,
                              ax=ax1)

    # save color bar
    # _, ax = plt.subplots(figsize=(3, 10))
    # ax.tick_params(labelsize='18')
    visual_util.add_color_bar(color_data=np.linspace(0, 1, 500),
                              norm=visual_util.SIGNIFICANT_NORM,
                              ax=ax2, orientation="horizontal")
    save_addr = os.path.join(
        _SAVE_ADDR_PREFIX,
        "{}/impact_mean_misspec.pdf".format(family_name))
    plt.savefig(save_addr)
    plt.close()

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

    """ 3.5.7. visualize: ensemble posterior cdf, localized """
    if _PLOT_LOCAL_CDF_COMPARISON:
        y_eval_grid = np.linspace(np.min(y_valid_sample),
                                  np.max(y_valid_sample), 1000)

        visual_util.compare_local_cdf_1d(X_pred=X_valid,
                                         y_post_sample=ensemble_sample_val.T,
                                         y_true_sample=y_valid_sample,
                                         x_eval_window=0.05,
                                         y_eval_grid=None,
                                         save_addr=os.path.join(
                                             _SAVE_ADDR_PREFIX,
                                             "{}/compare_cdf/".format(family_name))
                                         )

# call calibration
if _FIT_CALIB_MODELS:
    exec(open(
        "./example/parametric_ensemble/"
        "1d_skew_noise_local_calibration.py").read())
