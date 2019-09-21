"""Parametric Ensemble using MCMC and Calibrated VI.

Perform on Density Estimation problem.

"""

import os
import time

from importlib import reload
from functools import partial

import pickle as pk

import numpy as np
import scipy.stats as stats

from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed

# sys.path.extend([os.getcwd()])

from calibre.model import gaussian_process as gp
from calibre.model import tailfree_process as tail_free
from calibre.model import gp_regression_monotone as gpr_mono
from calibre.model import parametric_ensemble

from calibre.inference import mcmc
from calibre.inference import vi

from calibre.calibration import score

import calibre.util.data as data_util
import calibre.util.metric as metric_util
import calibre.util.visual as visual_util
import calibre.util.matrix as matrix_util
import calibre.util.gp_flow as gpf_util
import calibre.util.ensemble as ensemble_util
import calibre.util.calibration as calib_util
import calibre.util.experiment_data as experiment_util
import calibre.util.experiment_pred as pred_util

from calibre.util.model import sparse_softmax
from calibre.util.inference import make_value_setter
from calibre.util.gp_flow import DEFAULT_KERN_FUNC_DICT_RBF

import matplotlib.pyplot as plt
import seaborn as sns

tfd = tfp.distributions

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DEFAULT_LOG_LS_WEIGHT = np.log(0.1).astype(np.float32)
DEFAULT_LOG_LS_RESID = np.log(0.1).astype(np.float32)

_FIT_BASE_MODELS = False
_PLOT_COMPOSITION = False
_FIT_ALT_MODELS = False
_FIT_MAP_MODELS = False  # if true, then default_ls_weight will be replaced
_FIT_MCMC_MODELS = True
_FIT_VI_MODELS = False
_FIT_AUG_VI_MODELS = False
_FIT_CALIB_MODELS = True

"""""""""""""""""""""""""""""""""
# 0. Generate data
"""""""""""""""""""""""""""""""""

N_model = 3
N_test = 20
N_valid = 500

seed_model = 500
seed_test = 1000
seed_valid = 1500

_SAVE_ADDR_PREFIX = "./result/parametric_ensemble/0d_density"


def data_gen_func(size, intercept=5., shape=1.5, scale=2.):
    return np.random.gamma(shape=shape, scale=scale, size=size) + intercept


# generate y_test and y_valid
np.random.seed(seed_test)
y_test = data_gen_func(size=N_test)
np.random.seed(seed_valid)
y_valid = data_gen_func(size=N_valid)

""" 0.1. Prepare base prediction """
# generate base_pred
np.random.seed(seed_model)

base_pred_name = ['model_{}'.format(i) for i in range(N_model + 1)]
base_pred_list = [data_gen_func(size=N_test).astype(np.float32)
                  for i in range(N_model)]
base_pred_list = [np.ones(N_test, dtype=np.float32)] + base_pred_list
base_pred_dict = dict(zip(base_pred_name, base_pred_list))

with open(os.path.join(_SAVE_ADDR_PREFIX, 'base/base_test_pred.pkl'), 'wb') as file:
    pk.dump(base_pred_dict, file, protocol=pk.HIGHEST_PROTOCOL)

""" 0.2. Plot data """
# ground truth
sns.distplot(y_valid, hist=False,
             kde_kws={'shade': False, 'linewidth': 2,
                      'linestyle': '--', 'color': 'grey'})

## plot posterior mean
# add_vertical_segment(np.mean(y_valid), data_den_func(np.mean(y_valid)),
#                      lw=2, c='#A9A9A9')

# base models
for pred_val in np.concatenate(list(base_pred_dict.values())[1:]):
    plt.plot([pred_val, pred_val], [0, 0.01], lw=3, c='black')

plt.savefig("{}/data.png".format(_SAVE_ADDR_PREFIX))
plt.close()

"""""""""""""""""""""""""""""""""
# 2. Fit Ensemble (using MCMC)
"""""""""""""""""""""""""""""""""
family_name = "hmc"
family_name_full = "Hamilton MC"

_SAVE_ADDR = _SAVE_ADDR_PREFIX
os.makedirs('{}/{}'.format(_SAVE_ADDR_PREFIX, family_name), exist_ok=True)

"""2.0. prepare data"""
with open(os.path.join(_SAVE_ADDR_PREFIX, 'base/base_test_pred.pkl'), 'rb') as file:
    base_test_pred = pk.load(file)

y_test = y_test.squeeze()
# create empty placeholder for X since it is requred by model.
X_test = np.zeros((len(y_test), 2))

"""2.1. sampler basic config"""
num_results = 10000
num_burnin_steps = 5000

if _FIT_MCMC_MODELS:
    # define mcmc computation graph
    mcmc_graph = tf.Graph()
    with mcmc_graph.as_default():
        log_joint = ed.make_log_joint_fn(parametric_ensemble.model)

        ensemble_model_names = list(base_test_pred.keys())
        base_weight_names = ['base_weight_{}'.format(model_name) for
                             model_name in ensemble_model_names]
        model_specific_varnames = base_weight_names


        def target_log_prob_fn(sigma, temp,
                               *model_specific_positional_args):
            """Unnormalized target density as a function of states."""
            # build kwargs for base model weight using positional args
            model_specific_kwargs = dict(zip(model_specific_varnames,
                                             model_specific_positional_args))

            return log_joint(X=X_test,
                             base_pred=base_test_pred,
                             log_ls_resid=DEFAULT_LOG_LS_RESID,
                             y=y_test.squeeze(),
                             add_resid=False,
                             sigma=sigma,
                             temp=temp,
                             **model_specific_kwargs)


        # set up state container
        initial_state_scales = [
            tf.constant(0.1, name='init_sigma'),
            tf.constant(0.1, name='init_temp'),
        ]
        initial_state_base_weight = [
            tf.constant(1., name='init_base_weight_{}'.format(model_name)) for
            model_name in base_weight_names
        ]

        initial_state = initial_state_scales + initial_state_base_weight

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

        sigma_sample, temp_sample = state[:len(initial_state_scales)]
        base_weight_sample = state[len(initial_state_scales):]

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
            base_weight_sample_val,
            is_accepted_,
        ] = sess.run(
            [
                sigma_sample,
                temp_sample,
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
                           '{}/ensemble_ls_weight_sample.pkl'.format(family_name)), 'wb') as file:
        pk.dump(base_weight_sample_val, file, protocol=pk.HIGHEST_PROTOCOL)

""" 2.3. predictive sample"""
with tf.Session() as sess:
    W_ensemble = parametric_ensemble.sample_posterior_weight(
        base_weight_sample_val,
        temp_sample_val,
        link_func=sparse_softmax)

    F = np.asarray(list(base_test_pred.values())[1:]).T  # (N_obs, K)
    FW_sample = tf.matmul(F, W_ensemble[:, 1:], transpose_b=True)
    ensemble_mean = tf.transpose(FW_sample, name="f_ensemble")

    ensemble_mean_with_intercept = parametric_ensemble.sample_posterior_mean(
        base_test_pred,
        weight_sample=base_weight_sample_val,
        temp_sample=temp_sample_val,
        link_func=sparse_softmax)

    ensemble_resid_valid_sample = np.random.multivariate_normal(
        mean=np.zeros(len(sigma_sample_val)),
        cov=np.diag(np.exp(sigma_sample_val)),
        size=len(y_test)).T

    ensemble_mean_val, ensemble_mean_with_intercept_val, W_ensemble_val = sess.run([
        ensemble_mean, ensemble_mean_with_intercept, W_ensemble])

    ensemble_sample_val = ensemble_mean_with_intercept_val + ensemble_resid_valid_sample

with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/ensemble_posterior_mean_sample.pkl'.format(family_name)), 'wb') as file:
    pk.dump(ensemble_mean_val, file, protocol=pk.HIGHEST_PROTOCOL)

with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/ensemble_posterior_dist_sample.pkl'.format(family_name)), 'wb') as file:
    pk.dump(ensemble_sample_val, file, protocol=pk.HIGHEST_PROTOCOL)

""" 2.4. visualize"""
with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/ensemble_posterior_mean_sample.pkl'.format(family_name)), 'rb') as file:
    ensemble_mean_val = pk.load(file)

with open(os.path.join(_SAVE_ADDR_PREFIX,
                       '{}/ensemble_posterior_dist_sample.pkl'.format(family_name)), 'rb') as file:
    ensemble_sample_val = pk.load(file)

ensemble_mean_arr = np.concatenate(ensemble_mean_val)
ensemble_sample_arr = np.concatenate(ensemble_sample_val)


def save_fig_and_close(file_name, family_name):
    save_addr = os.path.join(_SAVE_ADDR_PREFIX,
                             "{}/{}.png".format(family_name, file_name))
    plt.savefig(save_addr, bbox_inches='tight')
    plt.close()


def plot_true_density(add_mean=False):
    # plot target
    ax, x, y = visual_util.scaled_1d_kde_plot(y_valid, shade=False, linewidth=2,
                                              linestyle='--', color='grey',
                                              density_scale=None)
    if add_mean:
        mean_height = y[np.argmin(abs(x - np.mean(y_valid)))]
        visual_util.add_vertical_segment(np.mean(y_valid), mean_height,
                                         c='grey', alpha=0.8, linestyle='--')
    # plot observed model predictions
    for pred_val in np.concatenate(list(base_pred_dict.values())[1:]):
        plt.plot([pred_val, pred_val], [0, 0.01], lw=3, c='black')


def plot_posterior_sample(add_mean=False):
    # plot uncalibrated posterior prediction
    ax, x, y = visual_util.scaled_1d_kde_plot(ensemble_sample_arr,
                                              shade=True, color='grey', linewidth=0,
                                              density_scale=None)
    if add_mean:
        mean_height = y[np.argmin(abs(x - np.mean(ensemble_mean_arr)))]
        visual_util.add_vertical_segment(np.mean(ensemble_mean_arr), mean_height,
                                         c='grey', alpha=0.8)
    return ax


def plot_posterior_mean(add_mean=False):
    # plot posterior mean
    ax, x, y = visual_util.scaled_1d_kde_plot(ensemble_mean_arr,
                                              shade=True, color='red', linewidth=0,
                                              bandwidth=0.25,
                                              density_scale=0.19)
    return ax


# produce plots
plot_true_density()
save_fig_and_close("data", family_name="hmc")

plot_true_density()
ax = plot_posterior_mean()
ax.set_xlim(2, 18.5)
ax.set_ylim(0, 0.24)
save_fig_and_close("ensemble_posterior_mean", family_name="hmc")

plot_true_density()
plot_posterior_sample()
ax = plot_posterior_mean()
ax.set_xlim(2, 18.5)
ax.set_ylim(0, 0.24)
save_fig_and_close("ensemble_posterior_sample", family_name="hmc")

""" 2.5. visualize: ensemble posterior reliability """

y_calib = y_valid
y_sample_calib = ensemble_sample_val.reshape(
    (len(y_calib), ensemble_sample_val.size // len(y_calib)))

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
# 3. Calibration
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
    os.makedirs(os.path.join(_SAVE_ADDR_PREFIX,
                             "{}/calibration/".format(family_name)),
                exist_ok=True)

    # load estimates
    with open(os.path.join(_SAVE_ADDR_PREFIX,
                           '{}/ensemble_posterior_dist_sample.pkl'.format(family_name)), 'rb') as file:
        ensemble_sample_val = pk.load(file)

    """ 7.2. build calibration dataset """
    y_calib = y_valid
    y_sample_calib_all = ensemble_sample_val.reshape(
        (len(y_calib), ensemble_sample_val.size // len(y_calib)))

    sample_size = y_sample_calib_all.shape[1]
    calib_sample_id = np.arange(sample_size)

    calib_train_id = np.random.choice(
        calib_sample_id, int(calib_sample_id.size / 2), replace=False)
    calib_test_id = np.asarray(list(set(calib_sample_id) -
                                    set(calib_train_id)))

    y_calib_train = y_valid[calib_train_id]
    y_calib_test = y_valid[calib_test_id]
    y_calib_sample_train = y_sample_calib_all[:, calib_train_id].T
    y_calib_sample_test = y_sample_calib_all[:, calib_test_id].T

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

    # scale x
    orig_prob_train = (orig_prob_train - 0.5) * 2
    orig_prob_test = (orig_prob_test - 0.5) * 2
    orig_prob_derv = (orig_prob_derv - 0.5) * 2
    orig_prob_pred = (orig_prob_pred - 0.5) * 2

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
                ridge_factor=1e-2,
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
            (gpf_sample, gpf_deriv_sample, sigma_sample,) = state

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
        # prediction over a regular grid between [0, 1]
        df_pred_val = gp.sample_posterior_full(X_new=orig_prob_pred,
                                               X=orig_prob_derv,
                                               f_sample=f_deriv_samples_val.T,
                                               ls=DEFAULT_LS_CALIB_VAL,
                                               kernel_func=gpr_mono.rbf_hess_1d,
                                               ridge_factor=5e-2,
                                               return_mean=False)

        calib_prob_pred_val = gp.sample_posterior_full(X_new=orig_prob_pred,
                                                       X=orig_prob_train,
                                                       f_sample=f_samples_val.T,
                                                       ls=DEFAULT_LS_CALIB_VAL,
                                                       kernel_func=gp.rbf,
                                                       ridge_factor=5e-2,
                                                       return_mean=False,
                                                       return_vcov=False)

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

        """ 7.4. produce calibrated posterior sample"""
        # re-sample observations based on newly obtained cdf
        calib_prob_pred = np.mean(calib_prob_pred_val, axis=1)
        calib_prob_pred[calib_prob_pred > 1.] = 1.
        calib_prob_pred[calib_prob_pred < 0.] = 0.

        ensemble_sample_calib_val = [
            calib_util.sample_ecdf(n_sample=1000,
                                   base_sample=base_sample,
                                   quantile=calib_prob_pred) for
            base_sample in y_sample_calib_all.T
        ]
        ensemble_sample_calib_val = np.asarray(ensemble_sample_calib_val).T

        # alternatively, re-sample by sampling calibrated cdf
        sample_list = np.random.choice(calib_prob_pred_val.shape[1], 500,
                                     replace=False)
        ensemble_sample_calib_val_sample = []
        for sample_id in sample_list:
            # calib_prob_pred = np.mean(calib_prob_pred_val, axis=1)
            calib_prob_pred = calib_prob_pred_val[:, sample_id]

            calib_prob_pred[calib_prob_pred > 1.] = 1.
            calib_prob_pred[calib_prob_pred < 0.] = 0.

            ensemble_sample_calib_val = [
                calib_util.sample_ecdf(n_sample=50,
                                       base_sample=base_sample,
                                       quantile=calib_prob_pred) for
                base_sample in y_sample_calib_all.T
            ]
            ensemble_sample_calib_val_sample.append(
                np.asarray(ensemble_sample_calib_val).T)

        with open(os.path.join(_SAVE_ADDR_PREFIX,
                               '{}/calibration/ensemble_posterior_dist_sample_ecdf_mean.pkl'.format(family_name)),
                  'wb') as file:
            pk.dump(ensemble_sample_calib_val, file, protocol=pk.HIGHEST_PROTOCOL)
        with open(os.path.join(_SAVE_ADDR_PREFIX,
                               '{}/calibration/ensemble_posterior_dist_sample_ecdf_sample.pkl'.format(family_name)),
                  'wb') as file:
            pk.dump(ensemble_sample_calib_val_sample, file, protocol=pk.HIGHEST_PROTOCOL)

    # # plot original vs calibrated cdf
    # sample_id = 50
    # plt.plot(np.sort(ensemble_sample_val[:, sample_id]),
    #          orig_prob_pred)
    # plt.plot(np.sort(ensemble_sample_val[:, sample_id]),
    #          calib_prob_pred)

    """ 7.5.1 visualize: ensemble posterior reliability """
    with open(os.path.join(_SAVE_ADDR_PREFIX,
                           '{}/calibration/ensemble_posterior_dist_sample_ecdf_mean.pkl'.format(family_name)), 'rb') as file:
        ensemble_sample_calib_val = pk.load(file)

    os.makedirs(os.path.join(_SAVE_ADDR_PREFIX,
                             "{}/calibration/".format(family_name)),
                exist_ok=True)

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

    """ 7.5.2. visualize: density of calibrated sample """


    def save_fig_and_close(file_name, family_name):
        save_addr = os.path.join(_SAVE_ADDR_PREFIX,
                                 "{}/calibration/{}.png".format(family_name, file_name))
        plt.savefig(save_addr, bbox_inches='tight')
        plt.close()


    def plot_true_density(add_mode=False, **kwargs):
        # plot target
        ax, x, y = visual_util.scaled_1d_kde_plot(y_valid, shade=False, linewidth=2,
                                                  linestyle='--', color='grey',
                                                  density_scale=None,
                                                  **kwargs)
        if add_mode:
            median_val = x[np.argmax(y)]
            median_height = y[np.argmin(abs(x - median_val))]
            visual_util.add_vertical_segment(median_val, median_height,
                                             c='grey', alpha=0.8, linestyle='--')
        # plot observed model predictions
        for pred_val in np.concatenate(list(base_pred_dict.values())[1:]):
            plt.plot([pred_val, pred_val], [0, 0.01], lw=3, c='black')


    def plot_posterior_sample(add_mode=False):
        # plot uncalibrated posterior prediction
        ensemble_sample_arr = np.concatenate(ensemble_sample_val)
        ax, x, y = visual_util.scaled_1d_kde_plot(ensemble_sample_arr,
                                                  shade=True, color='grey', linewidth=0,
                                                  density_scale=None)
        if add_mode:
            median_val = x[np.argmax(y)]
            median_height = y[np.argmin(abs(x - median_val))]
            visual_util.add_vertical_segment(median_val, median_height,
                                             c='grey', alpha=0.8)
        return ax


    def plot_posterior_mean(**kwargs):
        # plot posterior mean
        ensemble_mean_arr = np.concatenate(ensemble_mean_val)
        ax, x, y = visual_util.scaled_1d_kde_plot(ensemble_mean_arr,
                                                  shade=True, color='red', linewidth=0,
                                                  bandwidth=0.25,
                                                  density_scale=0.19,
                                                  **kwargs)
        return ax


    def plot_posterior_calib_sample(add_mode=False, **kwargs):
        # plot uncalibrated posterior prediction
        y_sample_calib_arr = np.concatenate(ensemble_sample_calib_val)
        ax, x, y = visual_util.scaled_1d_kde_plot(y_sample_calib_arr,
                                                  shade=True, color='blue', linewidth=0,
                                                  density_scale=None, **kwargs)
        if add_mode:
            median_val = x[np.argmax(y)]
            median_height = y[np.argmin(abs(x - median_val))]
            visual_util.add_vertical_segment(median_val, median_height,
                                             c='blue', alpha=0.8)
        return ax


    def plot_posterior_calib_uncertainty(scale=1.):
        # visualize posterior uncertainty
        std_val = np.std(calib_prob_pred_val, axis=1)
        std_val = std_val/np.max(std_val)

        y_sample_calib_arr = y_calib
        sample_percentile = np.percentile(y_sample_calib_arr,
                                          np.squeeze(orig_prob_pred/2+0.5)*100)

        plt.plot(sample_percentile, std_val * scale)


    # produce plots
    plot_true_density(add_mode=True)
    ax = plot_posterior_sample(add_mode=True)
    ax.set_xlim(2, 18.5)
    ax.set_ylim(0, 0.24)
    save_fig_and_close("ensemble_posterior_sample_with_mode", family_name="hmc")

    plot_true_density(add_mode=True)
    plot_posterior_sample()
    ax = plot_posterior_mean()
    plot_posterior_calib_sample(add_mode=True, bandwidth=0.5)
    # plot_posterior_calib_uncertainty(scale=.1)
    ax.set_xlim(2, 18.5)
    ax.set_ylim(0, 0.24)
    save_fig_and_close("ensemble_posterior_sample_calibrated", family_name="hmc")


    plot_true_density(add_mode=True)
    plot_posterior_sample(add_mode=True)
    ax = plot_posterior_calib_sample(add_mode=True, bandwidth=0.5)
    # plot_posterior_calib_uncertainty(scale=.1)
    ax.set_xlim(2, 18.5)
    ax.set_ylim(0, 0.24)
    save_fig_and_close("ensemble_posterior_sample_calibrated_with_mode", family_name="hmc")
