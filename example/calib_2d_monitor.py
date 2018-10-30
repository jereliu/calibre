"""Calibre and other methods on Annual 2011 Data"""
import os
import time

from importlib import reload

import pickle as pk
import pandas as pd

import numpy as np

from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed

# sys.path.extend([os.getcwd()])

from calibre.model import gaussian_process as gp
from calibre.model import tailfree_process as tail_free
from calibre.model import gp_regression_monotone as gpr_mono
from calibre.model import adaptive_ensemble

from calibre.calibration import score

import calibre.util.misc as misc_util
import calibre.util.metric as metric_util
import calibre.util.visual as visual_util
import calibre.util.matrix as matrix_util
import calibre.util.ensemble as ensemble_util
import calibre.util.calibration as calib_util
import calibre.util.experiment as experiment_util

from calibre.util.inference import make_value_setter

import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

_NUM_CV_FOLD = 20

_PLOT_COMPOSITION = False
_FIT_ALT_MODELS = True
_FIT_MCMC_MODELS = True
_FIT_VI_MODELS = True

_DATA_ADDR_PREFIX = "./example/data"
_SAVE_ADDR_PREFIX = "./result/calibre_2d_annual_pm25"

_MODEL_DICTIONARY = {"root": ["IK", "QD", "AV"]}

DEFAULT_LOG_LS_WEIGHT = np.log(0.35).astype(np.float32)
DEFAULT_LOG_LS_RESID = np.log(0.1).astype(np.float32)

"""""""""""""""""""""""""""""""""
# 0. Prepare data
"""""""""""""""""""""""""""""""""
os.makedirs(os.path.join(_SAVE_ADDR_PREFIX, 'base'), exist_ok=True)

""" 0. prepare testing data dictionary """
y_obs = pd.read_csv("{}/NEdta_Ensemble.csv".format(_DATA_ADDR_PREFIX))
y_obs_nona = y_obs[y_obs.measured.notnull()]
y_obs_2011 = y_obs_nona[y_obs_nona.year == 2011]

X_test = np.asarray(y_obs_2011[["longitude", "latitude"]].values.tolist()).astype(np.float32)
y_test = np.asarray(y_obs_2011["measured"].tolist()).astype(np.float32)

""" 1. prepare validation data dictionary """
base_valid_feat = dict()
base_valid_pred = dict()
for model_name in tail_free.get_leaf_model_names(_MODEL_DICTIONARY):
    data_pd = pd.read_csv("{}/{}_2011_align.csv".format(
        _DATA_ADDR_PREFIX, model_name))
    base_valid_feat[model_name] = np.asarray(data_pd[["lon", "lat"]].values.tolist()).astype(np.float32)
    base_valid_pred[model_name] = np.asarray(data_pd["pm25"].tolist()).astype(np.float32)

X_valid = base_valid_feat[model_name]

""" 2. prepare testing data dictionary """
base_test_feat = dict()
base_test_pred = dict()
for model_name in tail_free.get_leaf_model_names(_MODEL_DICTIONARY):
    test_idx = misc_util.find_nearest(base_valid_feat[model_name], X_test)
    base_test_feat[model_name] = base_valid_feat[model_name][test_idx]
    base_test_pred[model_name] = base_valid_pred[model_name][test_idx]

calib_sample_id = test_idx

with open(os.path.join(_SAVE_ADDR_PREFIX, 'base/base_test_feat.pkl'), 'wb') as file:
    pk.dump(base_test_feat, file, protocol=pk.HIGHEST_PROTOCOL)
with open(os.path.join(_SAVE_ADDR_PREFIX, 'base/base_test_pred.pkl'), 'wb') as file:
    pk.dump(base_test_pred, file, protocol=pk.HIGHEST_PROTOCOL)

with open(os.path.join(_SAVE_ADDR_PREFIX, 'base/base_valid_feat.pkl'), 'wb') as file:
    pk.dump(base_valid_feat, file, protocol=pk.HIGHEST_PROTOCOL)
with open(os.path.join(_SAVE_ADDR_PREFIX, 'base/base_valid_pred.pkl'), 'wb') as file:
    pk.dump(base_valid_pred, file, protocol=pk.HIGHEST_PROTOCOL)

""" 3. standardize data """
# standardize
X_centr = np.mean(X_valid, axis=0)
X_scale = np.max(X_valid, axis=0) - np.min(X_valid, axis=0)

X_valid = (X_valid - X_centr) / X_scale
X_test = (X_test - X_centr) / X_scale

"""""""""""""""""""""""""""""""""
# 1. Other Methods
"""""""""""""""""""""""""""""""""
os.makedirs(os.path.join(_SAVE_ADDR_PREFIX, "other"), exist_ok=True)

with open(os.path.join(_SAVE_ADDR_PREFIX, 'base/base_test_pred.pkl'), 'rb') as file:
    base_test_pred = pk.load(file)
with open(os.path.join(_SAVE_ADDR_PREFIX, 'base/base_valid_pred.pkl'), 'rb') as file:
    base_valid_pred = pk.load(file)

base_test_pred = {model_name: base_test_pred[model_name]
                  for model_name in
                  tail_free.get_leaf_model_names(_MODEL_DICTIONARY)}
base_valid_pred = {model_name: base_valid_pred[model_name]
                   for model_name in
                   tail_free.get_leaf_model_names(_MODEL_DICTIONARY)}

ens_model_list = {"avg": ensemble_util.AveragingEnsemble(),
                  "exp": ensemble_util.ExpWeighting(),
                  "cvs": ensemble_util.CVStacking(),
                  "gam": ensemble_util.GAMEnsemble(),
                  "lnr": ensemble_util.GAMEnsemble(residual_process=False),
                  "nlr": ensemble_util.GAMEnsemble(nonlinear_ensemble=True,
                                                   residual_process=False)}
if _FIT_ALT_MODELS:
    for ens_name, ens_model in ens_model_list.items():
        # define family name
        family_name = ens_name
        family_name_full = ens_model.name

        # perform 10-fold cross-validation
        kf = KFold(n_splits=_NUM_CV_FOLD, random_state=100)

        y_valid = []
        y_valid_pred = []
        ensemble_sample_val = []

        print("Estimating {}".format(family_name_full), end="")
        for train_index, test_index in kf.split(X_test):
            print(".", end="")
            # prepare features
            X_train_fold, X_valid_fold = X_test[train_index], X_test[test_index]
            y_train_fold, y_valid_fold = y_test[train_index], y_test[test_index]

            base_test_pred_fold = {
                model_name: base_test_pred[model_name][train_index]
                for model_name in base_test_pred.keys()}
            base_valid_pred_fold = {
                model_name: base_test_pred[model_name][test_index]
                for model_name in base_test_pred.keys()}

            # train model
            ens_model.train(X_train_fold, y_train_fold, base_test_pred_fold)
            y_valid_pred_fold, y_valid_pred_var_fold = (
                ens_model.predict(X_valid_fold, base_valid_pred_fold))

            y_valid.append(y_valid_fold)
            y_valid_pred.append(y_valid_pred_fold)

            # if prob prediction, examine uncertainty quantification
            if y_valid_pred_var_fold is not None:
                with tf.Session() as sess:
                    ensemble_sample_fold = gp.variational_mfvi_sample(
                        n_sample=1000,
                        qf_mean=y_valid_pred_fold,
                        qf_sdev=np.sqrt(y_valid_pred_var_fold))
                    ensemble_sample_fold_val = sess.run(ensemble_sample_fold)
                    sess.close()

                ensemble_sample_val.append(ensemble_sample_fold_val)

        y_valid = np.concatenate(y_valid)
        y_valid_pred = np.concatenate(y_valid_pred)

        # compute quality metric
        rmse_mean, rmse_std, rmse_sample = metric_util.boot_sample(
            y_valid, y_valid_pred, metric_func=metric_util.rmse)
        rsqr_mean, rsqr_std, rsqr_sample = metric_util.boot_sample(
            y_valid, y_valid_pred, metric_func=metric_util.rsqure)

        print("rmse={:.4f}, sd={:.4f}".format(rmse_mean, rmse_std))

        # if prob prediction, examine uncertainty quantification
        if ensemble_sample_val:
            ensemble_sample_val = np.concatenate(ensemble_sample_val, axis=-1)
            # measure calibration
            y_calib = y_test
            y_sample_calib = ensemble_sample_val.T

            visual_util.prob_calibration_1d(
                y_calib, y_sample_calib,
                title="Ensemble, {}".format(family_name_full),
                save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                       "other/ensemble_calibration_prob_{}.png".format(family_name)))

            visual_util.coverage_index_1d(
                y_calib, y_sample_calib,
                title="Ensemble, {}".format(family_name_full),
                save_addr=os.path.join(_SAVE_ADDR_PREFIX,
                                       "other/ensemble_credible_coverage_{}.png".format(family_name)))

        with open(os.path.join(_SAVE_ADDR_PREFIX,
                               'other/ensemble_y_valid_{}.pkl'.format(family_name)), 'wb') as file:
            pk.dump(y_valid, file, protocol=pk.HIGHEST_PROTOCOL)
        with open(os.path.join(_SAVE_ADDR_PREFIX,
                               'other/ensemble_pred_{}.pkl'.format(family_name)), 'wb') as file:
            pk.dump(y_valid_pred, file, protocol=pk.HIGHEST_PROTOCOL)
        with open(os.path.join(_SAVE_ADDR_PREFIX,
                               'other/ensemble_sample_{}.pkl'.format(family_name)), 'wb') as file:
            pk.dump(ensemble_sample_val, file, protocol=pk.HIGHEST_PROTOCOL)

        with open(os.path.join(_SAVE_ADDR_PREFIX,
                               'other/ensemble_rmse_sample_{}_{:.4f}_{:4f}.pkl'.format(
                                   family_name, rmse_mean, rmse_std)), 'wb') as file:
            pk.dump(rmse_sample, file, protocol=pk.HIGHEST_PROTOCOL)
        with open(os.path.join(_SAVE_ADDR_PREFIX,
                               'other/ensemble_rsqr_sample_{}_{:.4f}_{:4f}.pkl'.format(
                                   family_name, rsqr_mean, rsqr_std)), 'wb') as file:
            pk.dump(rsqr_sample, file, protocol=pk.HIGHEST_PROTOCOL)

"""""""""""""""""""""""""""""""""
# 2. MCMC
"""""""""""""""""""""""""""""""""
family_name = "hmc"
family_name_full = "Hamilton MC"

os.makedirs("{}/{}".format(_SAVE_ADDR_PREFIX, family_name),
            exist_ok=True)

with open(os.path.join(_SAVE_ADDR_PREFIX, 'base/base_test_pred.pkl'), 'rb') as file:
    base_test_pred = pk.load(file)

with open(os.path.join(_SAVE_ADDR_PREFIX, 'base/base_valid_pred.pkl'), 'rb') as file:
    base_valid_pred = pk.load(file)

"""2.1. sampler basic config"""
family_tree_dict = _MODEL_DICTIONARY

num_results = 1000
num_burnin_steps = 5000

"""2.2. cross-validation procedure"""

if _FIT_MCMC_MODELS:
    kf = KFold(n_splits=_NUM_CV_FOLD, random_state=100)
    y_valid = []
    ensemble_sample_val = []
    ensemble_weights_val = []

    for fold_id, (train_index, test_index) in enumerate(kf.split(X_test)):
        print("Estimating Fold {}".format(fold_id))

        # prepare features and base-model dictionaries
        X_train_fold, X_valid_fold = X_test[train_index], X_test[test_index]
        y_train_fold, y_valid_fold = y_test[train_index], y_test[test_index]

        base_test_pred_fold = {
            model_name: base_test_pred[model_name][train_index]
            for model_name in base_test_pred.keys()}
        base_valid_pred_fold = {
            model_name: base_test_pred[model_name][test_index]
            for model_name in base_test_pred.keys()}

        N_fold = X_train_fold.shape[0]

        # define mcmc computation graph
        mcmc_graph = tf.Graph()
        with mcmc_graph.as_default():
            # build likelihood explicitly
            log_joint = ed.make_log_joint_fn(adaptive_ensemble.model_tailfree)

            # aggregate node-specific variable names
            cond_weight_temp_names = ['temp_{}'.format(model_name) for
                                      model_name in
                                      tail_free.get_parent_node_names(family_tree_dict)]
            node_weight_names = ['base_weight_{}'.format(model_name) for
                                 model_name in
                                 tail_free.get_nonroot_node_names(family_tree_dict)]
            node_specific_varnames = cond_weight_temp_names + node_weight_names


            def target_log_prob_fn(sigma,
                                   ensemble_resid,
                                   *node_specific_positional_args):
                """Unnormalized target density as a function of states."""
                # build kwargs for base model weight using positional args
                node_specific_kwargs = dict(zip(node_specific_varnames,
                                                node_specific_positional_args))

                return log_joint(X=X_train_fold,
                                 base_pred=base_test_pred_fold,
                                 family_tree=family_tree_dict,
                                 y=y_train_fold.squeeze(),
                                 log_ls_weight=DEFAULT_LOG_LS_WEIGHT,
                                 log_ls_resid=DEFAULT_LOG_LS_RESID,
                                 sigma=sigma,
                                 ensemble_resid=ensemble_resid,
                                 **node_specific_kwargs)


            # set up state container
            initial_state = [
                                tf.constant(0.1, name='init_sigma'),
                                # tf.constant(-1., name='init_ls_weight'),
                                # tf.constant(-1., name='init_ls_resid'),
                                tf.random_normal([N_fold], stddev=0.01,
                                                 name='init_ensemble_resid'),
                            ] + [
                                tf.random_normal([], stddev=0.01,
                                                 name='init_{}'.format(var_name)) for
                                var_name in cond_weight_temp_names
                            ] + [
                                tf.random_normal([N_fold], stddev=0.01,
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

            (sigma_sample,
             # ls_weight_sample,
             # ls_resid_sample,
             ensemble_resid_sample) = state[:2]
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
                # ls_weight_sample_val,
                # ls_resid_sample_val,
                temp_sample_val,
                resid_sample_val,
                weight_sample_val,
                is_accepted_,
            ] = sess.run(
                [
                    sigma_sample,
                    # ls_weight_sample,
                    # ls_resid_sample,
                    temp_sample,
                    ensemble_resid_sample,
                    weight_sample,
                    kernel_results.is_accepted,
                ])
            print('Acceptance Rate: {}'.format(np.mean(is_accepted_)))
            sess.close()

        # DEFAULT_LOG_LS_WEIGHT = np.median(ls_weight_sample_val)
        # DEFAULT_LOG_LS_RESID = np.median(ls_resid_sample_val)

        """ 2.3.1. prediction """
        # compute GP prediction for weight GP and residual GP
        model_weight_valid_sample = []
        for model_weight_sample in weight_sample_val:
            model_weight_valid_sample.append(
                gp.sample_posterior_full(X_new=X_valid_fold, X=X_train_fold,
                                         f_sample=model_weight_sample.T,
                                         ls=np.exp(DEFAULT_LOG_LS_WEIGHT),
                                         kernel_func=gp.rbf).T.astype(np.float32)
            )

        ensemble_resid_valid_sample = (
            gp.sample_posterior_full(X_new=X_valid_fold, X=X_train_fold,
                                     f_sample=resid_sample_val.T,
                                     ls=np.exp(DEFAULT_LOG_LS_RESID),
                                     kernel_func=gp.rbf).T
        )

        # compute sample for posterior mean
        raw_weights_dict = dict(zip(tail_free.get_nonroot_node_names(family_tree_dict),
                                    model_weight_valid_sample))
        parent_temp_dict = dict(zip(tail_free.get_parent_node_names(family_tree_dict),
                                    temp_sample_val))

        (ensemble_sample_val_fold, ensemble_mean_val,
         ensemble_weights_val_fold, cond_weights_dict_val, ensemble_model_names) = (
            adaptive_ensemble.sample_posterior_tailfree(X=X_valid_fold,
                                                        base_pred_dict=base_valid_pred_fold,
                                                        family_tree=family_tree_dict,
                                                        weight_gp_dict=raw_weights_dict,
                                                        temp_dict=parent_temp_dict,
                                                        resid_gp_sample=ensemble_resid_valid_sample,
                                                        log_ls_weight=DEFAULT_LOG_LS_WEIGHT))
        # collect fold data
        y_valid.append(y_valid_fold)
        ensemble_sample_val.append(ensemble_sample_val_fold)
        ensemble_weights_val.append(ensemble_weights_val_fold)

        # compute rmse
        test_rmse_fold = metric_util.rmse(
            y_valid_fold, np.mean(ensemble_sample_val_fold, axis=0))

        print("Estimated ls_weight {:.4f}, ls_resid {:.4f}".format(
            np.exp(DEFAULT_LOG_LS_WEIGHT), np.exp(DEFAULT_LOG_LS_RESID)
        ))
        print("Fold Test RMSE: {:.4f}".format(test_rmse_fold))

    y_valid = np.concatenate(y_valid, axis=0)
    ensemble_sample_val = np.concatenate(ensemble_sample_val, axis=-1)

    # compute rmse
    rmse_mean, rmse_std, rmse_sample = metric_util.boot_sample(
        y_valid, np.mean(ensemble_sample_val, axis=0),
        metric_func=metric_util.rmse)
    print("Test RMSE={:.4f}, sd={:.4f}".format(rmse_mean, rmse_std))

    with open(os.path.join(_SAVE_ADDR_PREFIX,
                           '{}/ensemble_y_valid.pkl'.format(family_name)), 'wb') as file:
        pk.dump(y_valid, file, protocol=pk.HIGHEST_PROTOCOL)
    with open(os.path.join(_SAVE_ADDR_PREFIX,
                           '{}/ensemble_posterior_dist_sample.pkl'.format(family_name)), 'wb') as file:
        pk.dump(ensemble_sample_val, file, protocol=pk.HIGHEST_PROTOCOL)
    with open(os.path.join(_SAVE_ADDR_PREFIX,
                           '{}/ensemble_posterior_model_weights.pkl'.format(family_name)), 'wb') as file:
        pk.dump(ensemble_weights_val, file, protocol=pk.HIGHEST_PROTOCOL)

""" 2.2. quality metrics """

with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/ensemble_y_valid.pkl'.format(family_name)), 'rb') as file:
    y_valid = pk.load(file)

with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/ensemble_posterior_dist_sample.pkl'.format(family_name)), 'rb') as file:
    ensemble_sample_val = pk.load(file)

with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/ensemble_posterior_model_weights.pkl'.format(family_name)), 'rb') as file:
    ensemble_weights_val = pk.load(file)

# compute quality metrics
y_valid_pred = np.mean(ensemble_sample_val, axis=0)

rmse_mean, rmse_std, rmse_sample = metric_util.boot_sample(
    y_valid, y_valid_pred, metric_func=metric_util.rmse)
rsqr_mean, rsqr_std, rsqr_sample = metric_util.boot_sample(
    y_valid, y_valid_pred, metric_func=metric_util.rsqure)

with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/ensemble_rmse_{:.4f}_{:.4f}.pkl'.format(
                           family_name, rmse_mean, rmse_std)), 'wb') as file:
    pk.dump(rmse_sample, file, protocol=pk.HIGHEST_PROTOCOL)
with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/ensemble_rsqr_{:.4f}_{:.4f}.pkl'.format(
                           family_name, rsqr_mean, rsqr_std)), 'wb') as file:
    pk.dump(rsqr_sample, file, protocol=pk.HIGHEST_PROTOCOL)

""" 2.3. visualize: ensemble posterior reliability """
y_calib = y_valid
y_sample_calib = ensemble_sample_val.T

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
DEFAULT_LOG_LS_WEIGHT = np.log(0.35).astype(np.float32)
DEFAULT_LOG_LS_RESID = np.log(0.1).astype(np.float32)

with open(os.path.join(_SAVE_ADDR_PREFIX, 'base/base_test_pred.pkl'), 'rb') as file:
    base_test_pred = pk.load(file)

with open(os.path.join(_SAVE_ADDR_PREFIX, 'base/base_valid_pred.pkl'), 'rb') as file:
    base_valid_pred = pk.load(file)

""" 3.1. basic data/algorithm config"""

family_tree_dict = _MODEL_DICTIONARY

n_inference_sample = 100
n_final_sample = 1000  # number of samples to collect from variational family
max_steps = 20000  # number of training iterations

for family_name in ["mfvi", "sgpr"]:
    os.makedirs(os.path.join(_SAVE_ADDR_PREFIX, family_name), exist_ok=True)

    if family_name == "mfvi":
        family_name_full = "Mean-field VI"
        ensemble_variational_family = adaptive_ensemble.variational_mfvi
        ensemble_variational_family_sample = adaptive_ensemble.variational_mfvi_sample
    elif family_name == "sgpr":
        family_name_full = "Sparse Gaussian Process"
        ensemble_variational_family = adaptive_ensemble.variational_sgpr
        ensemble_variational_family_sample = adaptive_ensemble.variational_sgpr_sample

    if _FIT_VI_MODELS:
        kf = KFold(n_splits=_NUM_CV_FOLD, random_state=100)
        y_valid = []
        ensemble_sample_val = []
        ensemble_weights_val = []

        for fold_id, (train_index, test_index) in enumerate(kf.split(X_test)):
            print("Estimating Fold {}".format(fold_id))

            # prepare features and base-model dictionaries
            X_train_fold, X_valid_fold = X_test[train_index], X_test[test_index]
            y_train_fold, y_valid_fold = y_test[train_index], y_test[test_index]

            base_test_pred_fold = {
                model_name: base_test_pred[model_name][train_index]
                for model_name in base_test_pred.keys()}
            base_valid_pred_fold = {
                model_name: base_test_pred[model_name][test_index]
                for model_name in base_test_pred.keys()}

            N_fold = X_train_fold.shape[0]
            X_induce = KMeans(n_clusters=20, random_state=100).fit(
                X_train_fold).cluster_centers_.astype(np.float32)

            if _FIT_VI_MODELS:
                """ 3.2. Set up the computational graph """
                vi_graph = tf.Graph()

                with vi_graph.as_default():
                    # sample from variational family
                    (weight_gp_dict, resid_gp, temp_dict, sigma, _, _,  # variational RVs
                     weight_gp_mean_dict, weight_gp_vcov_dict,  # variational parameters, weight GP
                     resid_gp_mean, resid_gp_vcov,  # resid GP variational parameters
                     temp_mean_dict, temp_sdev_dict,  # temperature variational parameters
                     sigma_mean, sigma_sdev,  # variational parameters, resid GP
                     ) = ensemble_variational_family(X=X_train_fold,
                                                     Z=X_induce,
                                                     base_pred=base_test_pred_fold,
                                                     family_tree=family_tree_dict,
                                                     log_ls_weight=DEFAULT_LOG_LS_WEIGHT,
                                                     log_ls_resid=DEFAULT_LOG_LS_RESID,
                                                     kernel_func=gp.rbf,
                                                     ridge_factor=1e-3)

                    # assemble kwargs for make_value_setter
                    variational_rv_dict = {"ensemble_resid": resid_gp, "sigma": sigma, }
                    variational_rv_dict.update(temp_dict)
                    variational_rv_dict.update(weight_gp_dict)

                    # compute the expected predictive log-likelihood
                    with ed.tape() as model_tape:
                        with ed.interception(make_value_setter(**variational_rv_dict)):
                            y = adaptive_ensemble.model_tailfree(X=X_train_fold,
                                                                 base_pred=base_test_pred_fold,
                                                                 family_tree=family_tree_dict,
                                                                 log_ls_weight=DEFAULT_LOG_LS_WEIGHT,
                                                                 log_ls_resid=DEFAULT_LOG_LS_RESID,
                                                                 kernel_func=gp.rbf,
                                                                 ridge_factor=1e-3)

                    log_likelihood = y.distribution.log_prob(y_train_fold)

                    # compute the KL divergence
                    kl = 0.
                    for rv_name, variational_rv in variational_rv_dict.items():
                        kl += tf.reduce_sum(
                            variational_rv.distribution.kl_divergence(
                                model_tape[rv_name].distribution)
                        )

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
                        start_time = time.time()
                        _, elbo_value, = sess.run([train_op, elbo])
                        if step % 500 == 0:
                            duration = time.time() - start_time
                            print("Step: {:>3d} ELBO: {:.3f}, ({:.3f} sec)".format(
                                step, elbo_value, duration))

                    (weight_gp_mean_dict_val, weight_gp_vcov_dict_val,
                     resid_gp_mean_val, resid_gp_vcov_val,
                     temp_mean_dict_val, temp_sdev_dict_val,
                     sigma_mean_val, sigma_sdev_val) = sess.run([
                        weight_gp_mean_dict, weight_gp_vcov_dict,
                        resid_gp_mean, resid_gp_vcov,
                        temp_mean_dict, temp_sdev_dict,  # temperature variational parameters
                        sigma_mean, sigma_sdev])

                    sess.close()

                with tf.Session() as sess:
                    (weight_gp_sample_dict, temp_sample_dict,
                     resid_gp_sample, sigma_sample, _, _) = (
                        ensemble_variational_family_sample(
                            n_final_sample,
                            weight_gp_mean_dict_val, weight_gp_vcov_dict_val,
                            temp_mean_dict_val, temp_sdev_dict_val,
                            resid_gp_mean_val, resid_gp_vcov_val,
                            sigma_mean_val, sigma_sdev_val,
                            log_ls_weight_mean=DEFAULT_LOG_LS_WEIGHT,
                            log_ls_weight_sdev=.01,
                            log_ls_resid_mean=DEFAULT_LOG_LS_WEIGHT,
                            log_ls_resid_sdev=.01
                        ))

                    (weight_gp_sample_dict_val, temp_sample_dict_val,
                     resid_gp_sample_val, sigma_sample_val) = sess.run([
                        weight_gp_sample_dict, temp_sample_dict,
                        resid_gp_sample, sigma_sample])

                """ 3.5. prediction and posterior sampling """
                # compute GP prediction for weight GP and residual GP
                raw_weights_dict = dict()

                for model_name, model_weight_sample in weight_gp_sample_dict_val.items():
                    # extract node name and verify correctness
                    node_name = model_name.replace("{}_".format(tail_free.BASE_WEIGHT_NAME_PREFIX), "")
                    assert node_name in tail_free.get_nonroot_node_names(family_tree_dict)

                    raw_weights_dict[node_name] = (
                        gp.sample_posterior_full(X_new=X_valid_fold, X=X_train_fold,
                                                 f_sample=model_weight_sample.T,
                                                 ls=np.exp(DEFAULT_LOG_LS_WEIGHT),
                                                 kernel_func=gp.rbf).T.astype(np.float32))

                ensemble_resid_valid_sample = (
                    gp.sample_posterior_full(X_new=X_valid_fold, X=X_train_fold,
                                             f_sample=resid_gp_sample_val.T,
                                             ls=np.exp(DEFAULT_LOG_LS_RESID),
                                             kernel_func=gp.rbf).T
                )

                # prepare temperature dictionary
                parent_temp_dict = dict()
                for model_name, parent_temp_sample in temp_sample_dict_val.items():
                    # extract node name and verify correctness
                    node_name = model_name.replace("{}_".format(tail_free.TEMP_NAME_PREFIX), "")
                    assert node_name in tail_free.get_parent_node_names(family_tree_dict)

                    parent_temp_dict[node_name] = parent_temp_sample

                # compute sample for posterior mean
                (ensemble_sample_val_fold, ensemble_mean_val,
                 ensemble_weights_val_fold, cond_weights_dict_val, ensemble_model_names) = (
                    adaptive_ensemble.sample_posterior_tailfree(X=X_valid_fold,
                                                                base_pred_dict=base_valid_pred_fold,
                                                                family_tree=family_tree_dict,
                                                                weight_gp_dict=raw_weights_dict,
                                                                temp_dict=parent_temp_dict,
                                                                resid_gp_sample=ensemble_resid_valid_sample,
                                                                log_ls_weight=DEFAULT_LOG_LS_WEIGHT))

                # collect fold data
                y_valid.append(y_valid_fold)
                ensemble_sample_val.append(ensemble_sample_val_fold)
                ensemble_weights_val.append(ensemble_weights_val_fold)

                test_rmse_fold = metric_util.rmse(
                    y_valid_fold, np.mean(ensemble_sample_val_fold, axis=0))
                print("Fold Test RMSE: {:.4f}".format(test_rmse_fold))

        y_valid = np.concatenate(y_valid, axis=0)
        ensemble_sample_val = np.concatenate(ensemble_sample_val, axis=-1)

        # compute quality metrics
        rmse_mean, rmse_std, rmse_sample = metric_util.boot_sample(
            y_valid, y_valid_pred, metric_func=metric_util.rmse)
        print("Test RMSE={:.4f}, sd={:.4f}".format(rmse_mean, rmse_std))

        with open(os.path.join(_SAVE_ADDR_PREFIX,
                               '{}/ensemble_y_valid.pkl'.format(family_name)), 'wb') as file:
            pk.dump(y_valid, file, protocol=pk.HIGHEST_PROTOCOL)

        with open(os.path.join(_SAVE_ADDR_PREFIX,
                               '{}/ensemble_posterior_dist_sample.pkl'.format(family_name)), 'wb') as file:
            pk.dump(ensemble_sample_val, file, protocol=pk.HIGHEST_PROTOCOL)

        with open(os.path.join(_SAVE_ADDR_PREFIX,
                               '{}/ensemble_posterior_model_weights.pkl'.format(family_name)), 'wb') as file:
            pk.dump(ensemble_weights_val, file, protocol=pk.HIGHEST_PROTOCOL)

    """ 3.2. quality metrics """

    with open(os.path.join(_SAVE_ADDR_PREFIX,
                           '{}/ensemble_y_valid.pkl'.format(family_name)), 'rb') as file:
        y_valid = pk.load(file)

    with open(os.path.join(_SAVE_ADDR_PREFIX,
                           '{}/ensemble_posterior_dist_sample.pkl'.format(family_name)), 'rb') as file:
        ensemble_sample_val = pk.load(file)

    # compute quality metrics
    y_valid_pred = np.mean(ensemble_sample_val, axis=0)

    rmse_mean, rmse_std, rmse_sample = metric_util.boot_sample(
        y_valid, y_valid_pred, metric_func=metric_util.rmse)
    rsqr_mean, rsqr_std, rsqr_sample = metric_util.boot_sample(
        y_valid, y_valid_pred, metric_func=metric_util.rsqure)

    with open(os.path.join(_SAVE_ADDR_PREFIX,
                           '{}/ensemble_rmse_{:.4f}_{:.4f}.pkl'.format(
                               family_name, rmse_mean, rmse_std)), 'wb') as file:
        pk.dump(rmse_sample, file, protocol=pk.HIGHEST_PROTOCOL)
    with open(os.path.join(_SAVE_ADDR_PREFIX,
                           '{}/ensemble_rsqr_{:.4f}_{:.4f}.pkl'.format(
                               family_name, rsqr_mean, rsqr_std)), 'wb') as file:
        pk.dump(rsqr_sample, file, protocol=pk.HIGHEST_PROTOCOL)

    """ 3.3. visualize: ensemble posterior reliability """
    y_calib = y_valid
    y_sample_calib = ensemble_sample_val.T

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
