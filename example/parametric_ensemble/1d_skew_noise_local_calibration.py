"""Local calibration for parametric ensemble. """
import os
import sys
import time

import tqdm

import functools
from importlib import reload

import pickle as pk
import pandas as pd

import numpy as np
from scipy import stats

from scipy import interpolate
import scipy.signal as signal

from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed

import gpflowSlim as gpf

from calibre.model import gaussian_process as gp
from calibre.model import gp_regression as gpr
from calibre.model import gp_regression_calibration as gpr_calib
from calibre.model import parametric_ensemble

import calibre.util.data as data_util
import calibre.util.model as model_util
import calibre.util.metric as metric_util
import calibre.util.visual as visual_util
import calibre.util.gp_flow as gp_util
import calibre.util.calibration as calib_util
import calibre.util.experiment_data as experiment_util

from calibre.util.model import sparse_softmax
from calibre.util.inference import make_value_setter

import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

tfd = tfp.distributions

sys.path.extend([os.getcwd()])
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.warnings.filterwarnings('ignore')

_DEFAULT_LOG_LS_SCALE = np.log(0.25).astype(np.float32)

_SAVE_ADDR_PREFIX = "./result/parametric_ensemble/1d_skew_noise"

_VISUALIZE_CALIB_DATA = False
_FIT_CALIB_MODEL = False
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
data_gen_func_x_valid = functools.partial(data_util.gaussian_mix, sd_scale=5.)

(X_train_orig, y_train_orig,
 X_test_orig, y_test_orig,
 X_valid, y_valid_sample, calib_sample_id) = experiment_util.generate_data_1d(
    N_train=N_train, N_test=N_test, N_calib=N_calib, N_valid=N_valid,
    noise_sd=None,
    data_gen_func=data_gen_func,
    data_gen_func_x=data_gen_func_x,
    data_gen_func_x_test=data_gen_func_x_test,
    data_range=(-6., 6.), valid_range=(-6., 6.),
    seed_train=1000, seed_test=2500, seed_calib=500)

y_valid_mean = np.mean(y_valid_sample, axis=1)

# extract calibration sample
y_valid = y_valid_sample[:, 0]

"""""""""""""""""""""""""""""""""""""""""""""
# 2. Visualize (Ideal) Calibration Dataset
"""""""""""""""""""""""""""""""""""""""""""""
family_name = "dgpr"
family_name_full = "Decoupled Gaussian Process Regression"

os.makedirs(os.path.join(_SAVE_ADDR_PREFIX,
                         '{}/calibration_local/'.format(family_name)), exist_ok=True)

with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/ensemble_posterior_dist_sample.pkl'.format(family_name)), 'rb') as file:
    ensemble_sample_val = pk.load(file)

# generate "ideal" dataset
n_x_eval = 1000
n_cdf_eval = 100

y_eval_grid = np.linspace(np.min(y_valid_sample),
                          np.max(y_valid_sample), n_cdf_eval)

(ecdf_diff, ecdf_true, ecdf_modl,
 X_eval, y_eval_grid, X_pred, y_true_sample) = (
    metric_util.ecdf_l1_dist(X_pred=X_valid,
                             y_post_sample=ensemble_sample_val.T,
                             y_true_sample=y_valid_sample,
                             n_x_eval=n_x_eval,
                             n_cdf_eval=n_cdf_eval,
                             x_eval_window=0.05,
                             y_eval_grid=y_eval_grid,
                             n_max_sample=100,
                             return_addtional_data=True))

# prepare calibration features and labels
cdf_eval_subset_id = np.arange(0, n_cdf_eval, step=2)

feature_1 = ecdf_modl
feature_2 = np.tile(X_eval[:, np.newaxis], (1, n_cdf_eval))
label = ecdf_true

if _VISUALIZE_CALIB_DATA:
    # scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = feature_1[:, cdf_eval_subset_id].squeeze()
    y = feature_2[:, cdf_eval_subset_id].squeeze()
    z = label[:, cdf_eval_subset_id].squeeze()

    ax.scatter(x, y, z, c='r', marker='.', s=0.5)

    ax.set_xlabel('Model CDF')
    ax.set_ylabel('X')
    ax.set_zlabel('True CDF')

    plt.show()

    # surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = feature_1[:, cdf_eval_subset_id]
    y = feature_2[:, cdf_eval_subset_id]
    z = label[:, cdf_eval_subset_id]

    ax.plot_surface(x, y, z,
                    cmap='viridis', edgecolor='none')

    ax.set_xlabel('Model CDF')
    ax.set_ylabel('X')
    ax.set_zlabel('True CDF')

    plt.show()

"""""""""""""""""""""""""""""""""""""""""""""
# 3. Prepare Real Calibration Dataset
"""""""""""""""""""""""""""""""""""""""""""""
n_x_eval = 50
n_cdf_eval = 5

""" 3.1. compute empirical cdfs. """
#
n_valid_datasets = 100
calib_sample_id = np.arange(len(X_valid))

y_post_sample = ensemble_sample_val.T[calib_sample_id]
y_true_sample = y_valid_sample[calib_sample_id, :n_valid_datasets]

# choose locations to evaluate data
X_pred = np.squeeze(X_valid)[calib_sample_id]
X_ecdf_eval = np.linspace(np.min(X_pred), np.max(X_pred), n_x_eval)

# choose range to evaluate cdf
y_eval_grid = np.linspace(np.min(y_post_sample),
                          np.max(y_post_sample), n_cdf_eval)

# adjust shape of X_pred to match y_sample
X_pred_modl = np.repeat(X_pred, y_post_sample.shape[1])
X_pred_true = np.repeat(X_pred, y_true_sample.shape[1])

# compute training cdfs


ecdf_modl, _, y_eval_grid = metric_util.local_ecdf_1d(x_eval=X_ecdf_eval,
                                                      X=X_pred_modl,
                                                      y=y_post_sample,
                                                      n_cdf_eval=n_cdf_eval,
                                                      return_sample=True,
                                                      x_eval_window=0.1,
                                                      y_eval_grid=None
                                                      )

ecdf_true = metric_util.local_ecdf_1d(x_eval=X_ecdf_eval,
                                      X=X_pred_true,
                                      y=y_true_sample,
                                      n_cdf_eval=n_cdf_eval,
                                      return_sample=False,
                                      x_eval_window=0.1,
                                      y_eval_grid=y_eval_grid,
                                      )

ecdf_modl_train = np.asarray(ecdf_modl)
ecdf_true_train = np.asarray(ecdf_true)

""" 3.2. prepare data for visualization. """
# prepare training data
feature_1 = ecdf_modl_train
feature_2 = np.tile(X_ecdf_eval[:, np.newaxis], (1, n_cdf_eval))
label = ecdf_true_train

""" 3.3. visualization. """
if _VISUALIZE_CALIB_DATA:
    # scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(feature_1.squeeze(),
               feature_2.squeeze(),
               label.squeeze(), c='r', marker='.')

    ax.set_xlabel('Model CDF')
    ax.set_ylabel('X')
    ax.set_zlabel('True CDF')

    plt.show()

    # surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(feature_1, feature_2, label,
                    cmap='viridis', edgecolor='none')

    ax.set_xlabel('Model CDF')
    ax.set_ylabel('X')
    ax.set_zlabel('True CDF')

    plt.show()

"""""""""""""""""""""""""""""""""""""""""""""
# 4. Conduct Local Calibration
"""""""""""""""""""""""""""""""""""""""""""""
X_train = np.asarray([feature_1.flatten(),
                      feature_2.flatten()]).T
y_train = label.flatten()

X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)

""" 4.1. Set up the computational graph """
_ADD_MFVI_MIXTURE = False
_N_MFVI_MIXTURE = 10
_N_INFERENCE_SAMPLE = 100
_N_POSTERIOR_SAMPLE = 1000

N_induce_mean = int(len(X_train))
N_induce_cov = int(len(X_train)) // 2

X_induce_mean = KMeans(n_clusters=N_induce_mean, random_state=100).fit(
    X_train).cluster_centers_.astype(np.float32)
X_induce_cov = KMeans(n_clusters=N_induce_cov, random_state=100).fit(
    X_train).cluster_centers_.astype(np.float32)

vi_graph = tf.Graph()
with vi_graph.as_default():
    # sample from variational family
    q_f, q_sig, qf_mean, qf_vcov, mixture_par_list = (
        gpr.variational_dgpr(X=X_train,
                             Zm=X_induce_mean,
                             Zs=X_induce_cov,
                             ls=np.exp(_DEFAULT_LOG_LS_SCALE),
                             mfvi_mixture=_ADD_MFVI_MIXTURE,
                             n_mixture=_N_MFVI_MIXTURE)
    )

    # compute the expected predictive log-likelihood
    with ed.tape() as model_tape:
        with ed.interception(make_value_setter(gp_f=q_f, sigma=q_sig)):
            y, gp_f, sigma, log_ls, = gpr.model(X=X_train,
                                                log_ls=_DEFAULT_LOG_LS_SCALE,
                                                ridge_factor=1e-3)

    log_likelihood = y.distribution.log_prob(y_train)

    # compute the KL divergence
    if _ADD_MFVI_MIXTURE:
        # compute MC approximation
        qf_sample = q_f.distribution.sample(_N_INFERENCE_SAMPLE)
        qsig_sample = q_sig.distribution.sample(_N_INFERENCE_SAMPLE)

        kl = (q_f.distribution.log_prob(qf_sample) -
              gp_f.distribution.log_prob(qf_sample)) + (
                     q_sig.distribution.log_prob(qsig_sample) -
                     sigma.distribution.log_prob(qsig_sample))

    else:
        # compute analytical form
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

    vi_graph.finalize()

""" 4.2. execute optimization and sampling """
max_steps = 50000  # number of training iterations

if _FIT_CALIB_MODEL:
    with tf.Session(graph=vi_graph) as sess:
        start_time = time.time()

        sess.run(init_op)
        for step in range(max_steps):
            _, elbo_value = sess.run([train_op, elbo])

            if step % 500 == 0:
                duration = time.time() - start_time
                print("Step: {:>3d} Loss: {:.3f} ({:.3f} min)".format(
                    step, elbo_value, duration / 60.))
        qf_mean_val, qf_vcov_val = sess.run([qf_mean, qf_vcov])

        if _ADD_MFVI_MIXTURE:
            mixture_par_list_val = sess.run(mixture_par_list)
        else:
            mixture_par_list_val = []

        sess.close()

    with tf.Session() as sess:
        f_samples = gpr.variational_dgpr_sample(
            n_sample=_N_POSTERIOR_SAMPLE,
            qf_mean=qf_mean_val, qf_vcov=qf_vcov_val,
            mfvi_mixture=_ADD_MFVI_MIXTURE,
            mixture_par_list=mixture_par_list_val)

        f_samples_val = sess.run(f_samples)

    with open(os.path.join(_SAVE_ADDR_PREFIX,
                           '{}/calibration_local/calibration_cdf_sample.pkl'.format(family_name)),
              'wb') as file:
        pk.dump(f_samples_val, file, protocol=pk.HIGHEST_PROTOCOL)

    """ 4.3. prediction at testing locations """
    with open(os.path.join(_SAVE_ADDR_PREFIX,
                           '{}/calibration_local/calibration_cdf_sample.pkl'.format(family_name)),
              'rb') as file:
        f_samples_val = pk.load(file)
    with open(os.path.join(_SAVE_ADDR_PREFIX,
                           '{}/ensemble_posterior_dist_sample.pkl'.format(family_name)),
              'rb') as file:
        ensemble_sample_val = pk.load(file)
    with open(os.path.join(_SAVE_ADDR_PREFIX,
                           '{}/ensemble_posterior_mean_corrected_sample.pkl'.format(family_name)), 'rb') as file:
        ensemble_mean_corrected_val = pk.load(file)
    with open(os.path.join(_SAVE_ADDR_PREFIX,
                           '{}/ensemble_posterior_mean_sample.pkl'.format(family_name)), 'rb') as file:
        ensemble_mean_val = pk.load(file)

    #
    n_test_cdf_eval = 100

    """ 4.4. prediction at testing locations """
    # prepare test datasets

    uniform_quantile = np.linspace(0, 1, num=n_test_cdf_eval)
    uniform_quantile = uniform_quantile.reshape((uniform_quantile.size, 1))
    X_valid = X_valid.reshape((X_valid.size, 1))

    N_test, _ = X_valid.shape

    feature_1_test = np.repeat(uniform_quantile, repeats=N_test, axis=-1).T
    feature_2_test = np.tile(X_valid, reps=(1, n_test_cdf_eval))
    X_test = np.stack([feature_1_test, feature_2_test], axis=-1)

    # predict calibrated quantiles at test dataset
    predicted_quantiles = []

    for test_x_id in tqdm.tqdm(range(N_test)):
        X_test_val = X_test[test_x_id]

        f_test_val = gp.sample_posterior_full(X_new=X_test_val, X=X_train,
                                              f_sample=f_samples_val.T,
                                              ls=np.exp(_DEFAULT_LOG_LS_SCALE),
                                              kernel_func=gp.rbf)
        predicted_quantiles.append(f_test_val)

    # dimension n_x_eval, n_cdf_eval, n_sample
    predicted_quantiles = np.asarray(predicted_quantiles)

    with open(os.path.join(_SAVE_ADDR_PREFIX,
                           '{}/calibration_local/ensemble_posterior_predicted_quantiles.pkl'.format(family_name)),
              'wb') as file:
        pk.dump(predicted_quantiles, file, protocol=pk.HIGHEST_PROTOCOL)

""" 4.5. sample from calibrated predictive posterior """
# load data
with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/ensemble_posterior_dist_sample.pkl'.format(family_name)),
          'rb') as file:
    ensemble_sample_val = pk.load(file)
with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/ensemble_posterior_mean_corrected_sample.pkl'.format(family_name)), 'rb') as file:
    ensemble_mean_corrected_val = pk.load(file)
with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/ensemble_posterior_mean_sample.pkl'.format(family_name)), 'rb') as file:
    ensemble_mean_val = pk.load(file)

with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/calibration_local/ensemble_posterior_predicted_quantiles.pkl'.format(family_name)),
          'rb') as file:
    predicted_quantiles = pk.load(file)

predicted_quantiles_mean = np.mean(predicted_quantiles, axis=-1)

# sample from calibrated distribution, full sample
ensemble_sample_calib_val = (
    calib_util.resample_ecdf_batch(n_sample=1000,
                                   base_sample_batch=ensemble_sample_val.T,
                                   quantile_batch=predicted_quantiles_mean,
                                   # y_range=(np.min(y_valid), np.max(y_valid))
                                   ))

# sample from calibrated distribution, posterior mean
ensemble_mean_calib_val = (
    calib_util.resample_ecdf_batch(n_sample=1000,
                                   base_sample_batch=ensemble_mean_val.T,
                                   quantile_batch=predicted_quantiles_mean, ))

ensemble_mean_corrected_calib_val = (
    calib_util.resample_ecdf_batch(n_sample=1000,
                                   base_sample_batch=ensemble_mean_corrected_val.T,
                                   quantile_batch=predicted_quantiles_mean, ))

# sample from calibrated distribution, posterior mean E(Y|G) and variance Var(Y|G)
# using Darth Vader Formula E( g(Y) ) = int { g'(Y) S(Y) } dY

original_quantiles = np.linspace(0, 1,
                                 num=predicted_quantiles.shape[1])[None, :, None]
original_y_values = np.percentile(ensemble_sample_val,  # shape n_x_eval, n_quantile
                                  q=100 * original_quantiles.squeeze(), axis=0).T

# shape (dim n_x_eval, n_sample)
ensemble_mean_orig_sample = np.mean(ensemble_sample_val, axis=0)[:, None]
ensemble_mean_diff_sample = np.nanmean(
    (original_quantiles - predicted_quantiles), axis=1)
ensemble_mean_calib_sample = (ensemble_mean_orig_sample + ensemble_mean_diff_sample)

# sample from calibrated distribution, posterior variance Var(Y|G)

# shape (dim n_x_eval, n_sample)
ensemble_moment2_orig_sample = np.nanmean(original_y_values[:, :, None] ** 2, axis=1)
ensemble_moment2_diff_sample = np.nanmean(
    2 * original_y_values[:, :, None] *
    (original_quantiles - predicted_quantiles), axis=1)
ensemble_moment2_calib_sample = ensemble_moment2_orig_sample + ensemble_moment2_diff_sample

ensemble_var_orig_sample = (
        ensemble_moment2_orig_sample - ensemble_mean_orig_sample ** 2)
ensemble_var_calib_sample = (
        ensemble_moment2_calib_sample - ensemble_mean_calib_sample ** 2)

ensemble_var_diff_sample = (
        ensemble_var_calib_sample - ensemble_var_orig_sample).T

# sample from calibrated distribution, posterior centered skewness
# skewness = E[ (Y - E(Y)) ] =
#          = (E[ Y**3 ] - 3*E[ Y ]E[ Y**2 ] - E[ Y ]**3)
ensemble_moment3_orig_sample = np.nanmean(
    original_y_values[:, :, None] ** 3, axis=1)
ensemble_moment3_diff_sample = np.nanmean(
    3 * original_y_values[:, :, None] ** 2 *
    (original_quantiles - predicted_quantiles), axis=1)
ensemble_moment3_calib_sample = (ensemble_moment3_orig_sample +
                                 ensemble_moment3_diff_sample)

skew_nom = (ensemble_moment3_calib_sample -
            3 * ensemble_mean_calib_sample *
            ensemble_moment2_calib_sample -
            ensemble_mean_calib_sample ** 3)
skew_denom = ensemble_var_calib_sample
skew_denom[skew_denom < 0] = 1e-3
skew_denom = np.sqrt(skew_denom) ** 3

ensemble_skew_calib_sample = skew_nom / skew_denom

with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/calibration_local/ensemble_posterior_dist_sample.pkl'.format(family_name)),
          'wb') as file:
    pk.dump(ensemble_sample_calib_val, file, protocol=pk.HIGHEST_PROTOCOL)
with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/calibration_local/ensemble_posterior_mean_corrected_sample.pkl'.format(family_name)),
          'wb') as file:
    pk.dump(ensemble_mean_corrected_calib_val, file, protocol=pk.HIGHEST_PROTOCOL)
with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/calibration_local/ensemble_posterior_mean_sample.pkl'.format(family_name)),
          'wb') as file:
    pk.dump(ensemble_mean_calib_val, file, protocol=pk.HIGHEST_PROTOCOL)

with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/calibration_local/ensemble_posterior_mean_calib_sample.pkl'.format(family_name)),
          'wb') as file:
    pk.dump(ensemble_mean_calib_sample, file, protocol=pk.HIGHEST_PROTOCOL)
with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/calibration_local/ensemble_posterior_var_calib_sample.pkl'.format(family_name)),
          'wb') as file:
    pk.dump(ensemble_var_calib_sample, file, protocol=pk.HIGHEST_PROTOCOL)
with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/calibration_local/ensemble_posterior_skew_calib_sample.pkl'.format(family_name)),
          'wb') as file:
    pk.dump(ensemble_skew_calib_sample, file, protocol=pk.HIGHEST_PROTOCOL)

"""""""""""""""""""""""""""""""""""""""""""""
# 5. Visualizations
"""""""""""""""""""""""""""""""""""""""""""""
with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/calibration_local/ensemble_posterior_predicted_quantiles.pkl'.format(family_name)),
          'rb') as file:
    predicted_quantiles = pk.load(file)

with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/ensemble_posterior_dist_sample.pkl'.format(family_name)),
          'rb') as file:
    ensemble_sample_val_orig = pk.load(file)

with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/calibration_local/ensemble_posterior_dist_sample.pkl'.format(family_name)),
          'rb') as file:
    ensemble_sample_calib_val = pk.load(file)
with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/calibration_local/ensemble_posterior_mean_corrected_sample.pkl'.format(family_name)),
          'rb') as file:
    ensemble_mean_corrected_val = pk.load(file)

with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/ensemble_posterior_mean_sample.pkl'.format(family_name)), 'rb') as file:
    ensemble_mean_val = pk.load(file).T

with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/calibration_local/ensemble_posterior_mean_calib_sample.pkl'.format(family_name)),
          'rb') as file:
    ensemble_mean_calib_sample = pk.load(file)
with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/calibration_local/ensemble_posterior_var_calib_sample.pkl'.format(family_name)),
          'rb') as file:
    ensemble_var_calib_sample = pk.load(file)
with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/calibration_local/ensemble_posterior_skew_calib_sample.pkl'.format(family_name)),
          'rb') as file:
    ensemble_skew_calib_sample = pk.load(file)

""" 5.5.1 visualize: ensemble posterior full """
y_valid_sample_plot = y_valid_sample[:, :100]

# compute statistics, empirical distribution
true_dist_median = np.nanmedian(y_valid_sample_plot.T, axis=0)
true_dist_quantiles = [
    np.percentile(y_valid_sample_plot.T,
                  [100 - (100 - q) / 2, (100 - q) / 2], axis=0)
    for q in [68, 95, 99]
]

# compute statistics, uncalibrated predictive distribution
posterior_dist_median_orig = np.nanmedian(ensemble_sample_val_orig, axis=0)

posterior_dist_quantiles_orig = [
    np.percentile(ensemble_sample_val_orig,
                  [100 - (100 - q) / 2, (100 - q) / 2], axis=0)
    for q in [68, 95, 99]
]

# compute statistics, calibrated predictive distribution
posterior_mean_median = np.nanmedian(ensemble_mean_val.T, axis=0)
posterior_mean_cov = np.nanvar(ensemble_mean_val.T, axis=0)

posterior_mean_adj_median = np.nanmedian(ensemble_mean_corrected_val.T, axis=0)
posterior_mean_adj_cov = np.nanvar(ensemble_mean_corrected_val.T, axis=0)

posterior_dist_mu = np.nanmean(ensemble_sample_calib_val.T, axis=0)
posterior_dist_median = np.nanmedian(ensemble_sample_calib_val.T, axis=0)
posterior_dist_cov = np.nanvar(ensemble_sample_calib_val.T, axis=0)

posterior_dist_quantiles = [
    np.percentile(ensemble_sample_calib_val.T,
                  [100 - (100 - q) / 2, (100 - q) / 2], axis=0)
    for q in [68, 95, 99]
]

# compute statistics, additional variance due to G
posterior_dist_additional_sd = np.std(ensemble_mean_calib_sample, axis=1)

uncn_decomp_quantiles = [
    [posterior_dist_median - 3 * np.sqrt(posterior_dist_cov),
     posterior_dist_median + 3 * np.sqrt(posterior_dist_cov)],
    [posterior_dist_median - 3 * np.sqrt(posterior_mean_adj_cov),
     posterior_dist_median + 3 * np.sqrt(posterior_mean_adj_cov)],
    [posterior_dist_median - 3 * np.sqrt(posterior_mean_cov),
     posterior_dist_median + 3 * np.sqrt(posterior_mean_cov)],
]

uncn_decomp_quantiles_cond = [
    np.percentile(ensemble_sample_calib_val.T, [1, 99], axis=0),  # all
    np.percentile(ensemble_mean_corrected_val.T, [10, 90], axis=0) +
    posterior_dist_median - posterior_mean_adj_median,  # ensemble + resid
    np.percentile(ensemble_mean_val.T, [10, 90], axis=0) +
    posterior_dist_median - posterior_mean_median,  # ensemble
]

uncn_decomp_quantiles_all = [
    np.percentile(ensemble_sample_calib_val.T, [1, 99], axis=0) +
    np.stack([-posterior_dist_additional_sd,
              posterior_dist_additional_sd]) * 2,  # all
    np.percentile(ensemble_mean_corrected_val.T, [10, 90], axis=0) +
    np.stack([-posterior_dist_additional_sd,
              posterior_dist_additional_sd]) * 2 +
    posterior_dist_median - posterior_mean_adj_median,  # ensemble + resid + G
    np.percentile(ensemble_mean_corrected_val.T, [10, 90], axis=0) +
    posterior_dist_median - posterior_mean_adj_median,  # ensemble + resid
    np.array([
        posterior_dist_median - 1.5 * np.sqrt(posterior_mean_cov),
        posterior_dist_median + 1.5 * np.sqrt(posterior_mean_cov)])  # ensemble
]

# empirical distribution
visual_util.gpr_1d_visual(true_dist_median,
                          pred_cov=None,
                          pred_quantiles=true_dist_quantiles,
                          X_train=X_test_orig, y_train=y_test_orig,
                          X_test=X_valid, y_test=y_valid_mean,
                          rmse_id=calib_sample_id,
                          title="Posterior Predictive Quantiles, Ground Truth",
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                 "{}/calibration_local/ensemble_posterior_quantile_true.png".format(
                                                     family_name)),
                          y_range=[-2.5, 2.5]
                          )

# calibrated, fix G at its mean
visual_util.gpr_1d_visual(posterior_dist_median,
                          pred_cov=None,
                          pred_quantiles=posterior_dist_quantiles,
                          X_train=X_test_orig, y_train=y_test_orig,
                          X_test=X_valid, y_test=y_valid_mean,
                          rmse_id=calib_sample_id,
                          title="Posterior Predictive Quantiles, Calibrated VI",
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                 "{}/calibration_local/ensemble_posterior_quantile_calibrated.png".format(
                                                     family_name)),
                          y_range=[-2.5, 2.5],
                          smooth_mean=True
                          )

# uncalibrated (i.e. G = Identity).
visual_util.gpr_1d_visual(posterior_dist_median_orig,
                          pred_cov=None,
                          pred_quantiles=posterior_dist_quantiles_orig,
                          X_train=X_test_orig, y_train=y_test_orig,
                          X_test=X_valid, y_test=y_valid_mean,
                          rmse_id=calib_sample_id,
                          title="Posterior Predictive Quantiles, VI",
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                 "{}/calibration_local/ensemble_posterior_quantile_orig.png".format(
                                                     family_name)),
                          y_range=[-2.5, 2.5]
                          )

# overall decomposition, fixing G at its mean
visual_util.gpr_1d_visual(pred_mean=posterior_dist_median,
                          pred_cov=None,
                          pred_quantiles=uncn_decomp_quantiles_cond,
                          X_train=X_test_orig, y_train=y_test_orig,
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
                          quantile_alpha=.4,
                          title="Conditional Posterior y|E(G), Uncertainty Decomposition",
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                 "{}/calibration_local/ensemble_posterior_unc_decomp_cond.png".format(
                                                     family_name)),
                          y_range=[-2.5, 2.5],
                          smooth_mean=True
                          )

# overall decomposition, G ~ Posterior
visual_util.gpr_1d_visual(pred_mean=posterior_dist_median,
                          pred_cov=None,
                          pred_quantiles=uncn_decomp_quantiles_all,
                          X_train=X_test_orig, y_train=y_test_orig,
                          X_test=X_valid, y_test=y_valid_mean,
                          rmse_id=None,
                          quantile_colors=[
                              visual_util.UNC_COLOR_PALETTE["alea"],
                              visual_util.UNC_COLOR_PALETTE["str_random"],
                              visual_util.UNC_COLOR_PALETTE["str_system"],
                              visual_util.UNC_COLOR_PALETTE["para"],
                          ],
                          quantile_shade_legend=["Aleatoric",
                                                 "Structural, Random Comp",
                                                 "Structural, System Comp",
                                                 "Parametric"],
                          quantile_alpha=[.4, .5, .4, .4],
                          title="Uncertainty Decomposition, Full Model",
                          fontsize=18,
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                 "{}/calibration_local/unc_decomp_all.png".format(
                                                     family_name)),
                          y_range=[-2.5, 2.5],
                          smooth_mean=True,
                          figsize=(12, 6)
                          )

""" 5.5.3.1. visualize: L1 distance between predictive and empirical CDF"""
original_quantiles = np.linspace(0, 1,
                                 num=predicted_quantiles.shape[1])[None, :, None]
ensemble_dist_diff_sample = np.nanmean(
    np.abs(original_quantiles - predicted_quantiles), axis=1).T

posterior_resid_mu = np.nanmean(ensemble_dist_diff_sample, axis=0)
posterior_resid_cov = np.nanvar(ensemble_dist_diff_sample, axis=0)

posterior_resid_median = np.nanmedian(ensemble_dist_diff_sample, axis=0)

posterior_resid_quantiles = [
    np.percentile(ensemble_dist_diff_sample,
                  [100 - (100 - q) / 2, (100 - q) / 2], axis=0)
    for q in [68, 95, 99]
]

# random component misspecification, posterior distribution
visual_util.gpr_1d_visual(posterior_resid_mu,
                          pred_cov=posterior_resid_cov,
                          pred_quantiles=None,
                          X_train=None, y_train=None,
                          X_test=X_valid, y_test=None,
                          rmse_id=None,
                          title=r"Structural Uncertainty, Random Component ($P( Diff > 0 )$)",
                          save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                                 "{}/calibration_local/l1_random_misspecifcation_dist.png".format(
                                                     family_name)),
                          y_range=[0., 0.65],
                          add_reference=True,
                          ax=None)

""" 5.5.3.2 impact of random component mis-specification: predictive mean"""
# compute mean difference
original_quantiles = np.linspace(0, 1,
                                 num=predicted_quantiles.shape[1])[None, :, None]
ensemble_dist_diff_sample = np.nanmean(
    original_quantiles - predicted_quantiles, axis=1).T

# compute summary statistics
posterior_resid_mu = np.nanmean(ensemble_dist_diff_sample, axis=0)
posterior_resid_median = np.nanmedian(ensemble_dist_diff_sample, axis=0)

posterior_resid_cov = np.nanvar(ensemble_dist_diff_sample, axis=0)
posterior_resid_quantiles = [
    np.percentile(ensemble_dist_diff_sample,
                  [100 - (100 - q) / 2, (100 - q) / 2], axis=0)
    for q in [68, 95, 99]
]

posterior_resid_zero_quantile = np.nanmean(
    ensemble_dist_diff_sample > 0, axis=0).flatten()

# specify color norm
quantile_color_norm = visual_util.make_color_norm(
    [np.linspace(0, 0.1, 35),
     np.linspace(0.05, 0.95, 30),
     np.linspace(0.9, 1, 35)],
    method="percentile")

# interpolate
f = interpolate.interp1d(x=X_valid.flatten(),
                         y=posterior_resid_zero_quantile,
                         kind='cubic')
X_valid_new = np.linspace(np.min(X_valid),
                          np.max(X_valid), 2000)
posterior_mean_zero_quantile = f(X_valid_new)

posterior_mean_impact = posterior_mean_zero_quantile - 0.5
posterior_mean_impact_abs = np.abs(posterior_mean_impact)

# initialize graph
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(2, 2, width_ratios=[20, 1])
ax1 = plt.subplot(gs[0])
ax3 = plt.subplot(gs[2])
ax4 = plt.subplot(gs[3])

# random component misspecification, posterior distribution
visual_util.gpr_1d_visual(posterior_resid_mu,
                          pred_cov=posterior_resid_cov,
                          X_train=None, y_train=None,
                          X_test=X_valid, y_test=None,
                          rmse_id=None,
                          title=r"Impact on Predictive Mean, Structural Uncertainty",
                          save_addr=None,
                          y_range=None,
                          add_reference=True,
                          ax=ax1)

# random component misspecification, posterior belief
visual_util.gpr_1d_visual(posterior_mean_impact_abs * 2,
                          pred_cov=None,
                          pred_quantiles=[[
                              np.zeros(posterior_mean_impact_abs.size),
                              posterior_mean_impact_abs * 2
                          ]],
                          X_train=None, y_train=None,
                          X_test=X_valid_new, y_test=None,
                          rmse_id=None,
                          title=r"Posterior Confidence, $P( Diff > 0 )$",
                          save_addr=None,
                          y_range=[0., 1.1],
                          quantile_colors=posterior_mean_zero_quantile,
                          quantile_colors_norm=visual_util.SIGNIFICANT_NORM,
                          ax=ax3,
                          pred_mean_color='white',
                          pred_mean_alpha=0.,
                          smooth_quantile=True)

visual_util.add_color_bar(color_data=np.linspace(0, 1, 500),
                          norm=visual_util.SIGNIFICANT_NORM,
                          ax=ax4)

save_addr = os.path.join(
    _SAVE_ADDR_PREFIX,
    "{}/calibration_local/impact_random_misspecifcation_mean_abs.png".format(family_name))
plt.savefig(save_addr)
plt.close()

""" 5.5.3.3 impact of random component mis-specification: predictive variance"""
# compute difference in variance, dim n_x_eval, n_quantile, n_sample
original_quantiles = np.linspace(0, 1,
                                 num=predicted_quantiles.shape[1])[None, :, None]
original_y_values = np.percentile(ensemble_sample_val_orig,
                                  q=100 * original_quantiles.squeeze(), axis=0).T

ensemble_EY_orig_sample = np.mean(ensemble_sample_val_orig, axis=0)[:, None]
ensemble_EY2_orig_sample = np.nanmean(original_y_values[:, :, None] ** 2, axis=1)
ensemble_var_orig_sample = (
        ensemble_EY2_orig_sample - ensemble_EY_orig_sample ** 2)
ensemble_var_diff_sample = (
        ensemble_var_calib_sample - ensemble_var_orig_sample).T

# compute summary statistics
posterior_ensemble_var_mu = np.nanmean(ensemble_var_diff_sample, axis=0)
posterior_ensemble_var_cov = np.nanvar(ensemble_var_diff_sample, axis=0)

posterior_ensemble_var_median = np.nanmedian(ensemble_var_diff_sample, axis=0)

posterior_ensemble_var_quantiles = [
    np.percentile(ensemble_var_diff_sample,
                  [100 - (100 - q) / 2, (100 - q) / 2], axis=0)
    for q in [68, 95, 99]
]

# zero quantile
posterior_resid_zero_quantile = np.nanmean(
    ensemble_var_diff_sample > 0, axis=0).flatten()
f = interpolate.interp1d(x=X_valid.flatten(),
                         y=posterior_resid_zero_quantile,
                         kind='cubic')
X_valid_new = np.linspace(np.min(X_valid),
                          np.max(X_valid), 2000)
posterior_var_zero_quantile = f(X_valid_new)

posterior_var_impact = np.abs(
    posterior_var_zero_quantile - 0.5
)

# random component misspecification, posterior distribution

fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(2, 2, width_ratios=[20, 1])
ax1 = plt.subplot(gs[0])
ax3 = plt.subplot(gs[2])
ax4 = plt.subplot(gs[3])

# random component misspecification, posterior distribution
visual_util.gpr_1d_visual(posterior_ensemble_var_median,
                          pred_cov=None,
                          pred_quantiles=posterior_ensemble_var_quantiles,
                          X_train=None, y_train=None,
                          X_test=X_valid, y_test=None,
                          rmse_id=None,
                          title=r"Model Over-confidence due to Random Component Misspecification",
                          save_addr=None,
                          y_range=None,
                          add_reference=True,
                          ax=ax1)

# random component misspecification, posterior belief
visual_util.gpr_1d_visual(posterior_var_impact * 2,
                          pred_cov=None,
                          pred_quantiles=[[
                              np.zeros(posterior_var_impact.size),
                              posterior_var_impact * 2
                          ]],
                          X_train=None, y_train=None,
                          X_test=X_valid_new, y_test=None,
                          rmse_id=None,
                          title=r"Posterior Confidence, $P( Diff > 0 )$",
                          save_addr=None,
                          y_range=[0., 1.1],
                          quantile_colors=posterior_var_zero_quantile,
                          quantile_colors_norm=visual_util.SIGNIFICANT_NORM,
                          ax=ax3,
                          pred_mean_color='white',
                          pred_mean_alpha=0.,
                          smooth_quantile=True)

visual_util.add_color_bar(color_data=np.linspace(0, 1, 500),
                          norm=visual_util.SIGNIFICANT_NORM,
                          ax=ax4)

save_addr = os.path.join(
    _SAVE_ADDR_PREFIX,
    "{}/calibration_local/impact_random_misspecifcation_var.png".format(family_name))
plt.savefig(save_addr)
plt.close()

""" 5.5.3.3. impact of random component mis-specification: predictive skewness"""
# compute difference in skewness (cubic root), shape dim n_x_eval, n_sample
original_y_values = np.percentile(ensemble_sample_val_orig,
                                  q=100 * original_quantiles.squeeze(), axis=0).T

ensemble_skew_orig_sample = np.nanmean(
    (original_y_values - np.mean(original_y_values, axis=1, keepdims=True)) /
    (np.std(original_y_values, axis=1, keepdims=True) ** 3), axis=1)

# compute difference in cubic root
ensemble_skew_diff_sample = (np.cbrt(ensemble_skew_calib_sample) -
                             np.cbrt(ensemble_skew_orig_sample)[:, None])

# compute summary statistics using difference in skewness,
posterior_ensemble_skew_mu = np.nanmean(ensemble_skew_diff_sample.T, axis=0)
posterior_ensemble_skew_median = np.nanmedian(ensemble_skew_diff_sample.T, axis=0)

posterior_ensemble_skew_cov = np.nanvar(ensemble_skew_diff_sample.T, axis=0)
posterior_ensemble_skew_quantiles = [
    np.percentile(ensemble_skew_diff_sample.T,
                  [100 - (100 - q) / 2, (100 - q) / 2], axis=0)
    for q in [68, 95, 99]
]

# zero quantile
ensemble_skew_diff_quantile = np.nanmean(
    ensemble_skew_diff_sample.T > 0, axis=0).flatten()
f = interpolate.interp1d(x=X_valid.flatten(),
                         y=ensemble_skew_diff_quantile,
                         kind='cubic')
X_valid_new = np.linspace(np.min(X_valid),
                          np.max(X_valid), 2000)
posterior_skew_zero_quantile = f(X_valid_new)

posterior_skew_impact = np.abs(
    posterior_skew_zero_quantile - 0.5
)

# random component misspecification, posterior distribution
fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(2, 2, width_ratios=[20, 1])
ax1 = plt.subplot(gs[0])
ax3 = plt.subplot(gs[2])
ax4 = plt.subplot(gs[3])

# random component misspecification, posterior distribution
visual_util.gpr_1d_visual(posterior_ensemble_skew_median,
                          pred_cov=None,
                          pred_quantiles=posterior_ensemble_skew_quantiles,
                          X_train=None, y_train=None,
                          X_test=X_valid, y_test=None,
                          rmse_id=None,
                          title=r"Error in Skewness due to Random Component Misspecification",
                          save_addr=None,
                          y_range=None,
                          add_reference=True,
                          smooth_mean=True,
                          smooth_quantile=True,
                          ax=ax1)

# random component misspecification, posterior belief
visual_util.gpr_1d_visual(posterior_skew_impact * 2,
                          pred_cov=None,
                          pred_quantiles=[[
                              np.zeros(posterior_skew_impact.size),
                              posterior_skew_impact * 2
                          ]],
                          X_train=None, y_train=None,
                          X_test=X_valid_new, y_test=None,
                          rmse_id=None,
                          title=r"Posterior Confidence, $P( Diff > 0 )$",
                          save_addr=None,
                          y_range=[0., 1.1],
                          quantile_colors=posterior_skew_zero_quantile,
                          quantile_colors_norm=visual_util.SIGNIFICANT_NORM,
                          ax=ax3,
                          pred_mean_color='white',
                          pred_mean_alpha=0.,
                          smooth_quantile=True)

visual_util.add_color_bar(color_data=np.linspace(0, 1, 500),
                          norm=visual_util.SIGNIFICANT_NORM,
                          ax=ax4)

save_addr = os.path.join(
    _SAVE_ADDR_PREFIX,
    "{}/calibration_local/impact_random_misspecifcation_skew.png".format(family_name))
plt.savefig(save_addr)
plt.close()

""" 5.5.3.4 impact of random component mis-specification: predictive mean & variance"""
_, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
ax1.tick_params(labelsize='5')
ax2.tick_params(labelsize='5')

# random component misspecification, posterior belief
visual_util.gpr_1d_visual(posterior_mean_impact_abs * 2,
                          pred_cov=None,
                          pred_quantiles=[[
                              np.zeros(posterior_mean_impact.size),
                              posterior_mean_impact_abs * 2
                          ]],
                          X_train=None, y_train=None,
                          X_test=X_valid_new, y_test=None,
                          rmse_id=None,
                          title=r"Error in Predictive Mean due to Random Comp Misspecification",
                          fontsize=18,
                          save_addr=None,
                          y_range=[0, 1.1],
                          quantile_colors=posterior_mean_zero_quantile,
                          quantile_colors_norm=visual_util.SIGNIFICANT_NORM,
                          ax=ax1,
                          pred_mean_color='white',
                          pred_mean_alpha=0.,
                          smooth_quantile=True,
                          )

visual_util.gpr_1d_visual(posterior_var_impact * 2,
                          pred_cov=None,
                          pred_quantiles=[[
                              np.zeros(posterior_var_impact.size),
                              posterior_var_impact * 2
                          ]],
                          X_train=None, y_train=None,
                          X_test=X_valid_new, y_test=None,
                          rmse_id=None,
                          title=r"Error in Predictive Variance due to Random Comp Misspecification",
                          fontsize=18,
                          save_addr=None,
                          y_range=[0., 1.1],
                          quantile_colors=posterior_var_zero_quantile,
                          quantile_colors_norm=visual_util.SIGNIFICANT_NORM,
                          ax=ax2,
                          pred_mean_color='white',
                          pred_mean_alpha=0.,
                          smooth_quantile=True)

save_addr = os.path.join(
    _SAVE_ADDR_PREFIX,
    "{}/calibration_local/impact_random_misspec.png".format(family_name))
plt.savefig(save_addr)
plt.close()

""" 5.5.4. visualize: ensemble posterior reliability """
# overall
y_calib = y_valid[calib_sample_id]
y_sample_calib = ensemble_sample_calib_val[:, calib_sample_id].T

visual_util.prob_calibration_1d(
    y_calib, y_sample_calib,
    title="Calibrated, {}".format(family_name_full),
    save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                           "{}/calibration_local/gpr_calibration_prob_all.png".format(family_name)))

visual_util.coverage_index_1d(
    y_calib, y_sample_calib,
    title="Calibrated, {}".format(family_name_full),
    save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                           "{}/calibration_local/gpr_credible_coverage_all.png".format(family_name)))

""" 5.5.4. visualize: ensemble posterior cdf, localized """
if _PLOT_LOCAL_CDF_COMPARISON:
    y_eval_grid = np.linspace(np.min(ensemble_sample_calib_val),
                              np.max(ensemble_sample_calib_val), 1000)

    # original
    visual_util.compare_local_cdf_1d(X_pred=X_valid,
                                     y_post_sample=ensemble_sample_val_orig.T,
                                     y_true_sample=y_valid_sample_plot,
                                     x_eval_window=0.05,
                                     y_eval_grid=y_eval_grid,
                                     save_addr=os.path.join(
                                         _SAVE_ADDR_PREFIX,
                                         "{}/compare_cdf/".format(family_name))
                                     )

    # calibrated
    visual_util.compare_local_cdf_1d(X_pred=X_valid,
                                     y_post_sample=ensemble_sample_calib_val,
                                     y_true_sample=y_valid_sample_plot,
                                     x_eval_window=0.05,
                                     y_eval_grid=y_eval_grid,
                                     save_addr=os.path.join(
                                         _SAVE_ADDR_PREFIX,
                                         "{}/calibration_local/compare_cdf/".format(family_name))
                                     )
