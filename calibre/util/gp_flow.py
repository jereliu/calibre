"""API functions to GPflow-Slim package"""

import os
import sys
import pathlib

import pickle as pk

import numpy as np

import tensorflow as tf

import GPy as gpy
import gpflowSlim as gpf

sys.path.extend([os.getcwd()])

import calibre.util.visual as visual_util

# Example dictionary of kernel functions to fit. """
DEFAULT_KERN_FUNC_DICT_GPFLOW = {
    "poly_1": {'kernel': gpf.kernels.Polynomial,
               'param': {'degree': 1.,
                         'train_kernel_params': False}},
    "poly_2": {'kernel': gpf.kernels.Polynomial,
               'param': {'degree': 2.,
                         'train_kernel_params': False}},
    "poly_3": {'kernel': gpf.kernels.Polynomial,
               'param': {'degree': 3.,
                         'train_kernel_params': False}},
    "rquad1_0.1": {'kernel': gpf.kernels.RatQuad,
                   'param': {'lengthscales': .1, 'alpha': 1.,
                             'train_kernel_params': False}},
    "rquad1_0.2": {'kernel': gpf.kernels.RatQuad,
                   'param': {'lengthscales': .2, 'alpha': 1.,
                             'train_kernel_params': False}},
    "rquad1_0.5": {'kernel': gpf.kernels.RatQuad,
                   'param': {'lengthscales': .5, 'alpha': 1.,
                             'train_kernel_params': False}},
    "rquad2_0.1": {'kernel': gpf.kernels.RatQuad,
                   'param': {'lengthscales': .1, 'alpha': 2.,
                             'train_kernel_params': False}},
    "rquad2_0.2": {'kernel': gpf.kernels.RatQuad,
                   'param': {'lengthscales': .2, 'alpha': 2.,
                             'train_kernel_params': False}},
    "rquad2_0.5": {'kernel': gpf.kernels.RatQuad,
                   'param': {'lengthscales': .5, 'alpha': 2.,
                             'train_kernel_params': False}},
    "rquad_auto": {'kernel': gpf.kernels.RatQuad,
                   'param': {'lengthscales': .5, 'alpha': 2.,
                             'train_kernel_params': True}},
    "period0.5_0.15": {'kernel': gpf.kernels.Periodic,
                       'param': {'lengthscales': .15, 'period': .5,
                                 'train_kernel_params': False}},
    "period1_0.15": {'kernel': gpf.kernels.Periodic,
                     'param': {'lengthscales': .15, 'period': 1.,
                               'train_kernel_params': False}},
    "period1.5_0.15": {'kernel': gpf.kernels.Periodic,
                       'param': {'lengthscales': .15, 'period': 1.5,
                                 'train_kernel_params': False}},
    "period_auto": {'kernel': gpf.kernels.Periodic,
                    'param': {'lengthscales': .15, 'period': 1.5,
                              'train_kernel_params': True}},
    "matern12_auto": {'kernel': gpf.kernels.Matern12,
                      'param': {'lengthscales': .15,
                                'train_kernel_params': True}},
    "matern32_auto": {'kernel': gpf.kernels.Matern32,
                      'param': {'lengthscales': .15,
                                'train_kernel_params': True}},
    "matern52_auto": {'kernel': gpf.kernels.Matern52,
                      'param': {'lengthscales': .15,
                                'train_kernel_params': True}},
    "rbf_1": {'kernel': gpf.kernels.RBF,
              'param': {'lengthscales': 1.,
                        'train_kernel_params': False}},
    "rbf_0.5": {'kernel': gpf.kernels.RBF,
                'param': {'lengthscales': .5,
                          'train_kernel_params': False}},
    "rbf_0.2": {'kernel': gpf.kernels.RBF,
                'param': {'lengthscales': .2,
                          'train_kernel_params': False}},
    "rbf_0.05": {'kernel': gpf.kernels.RBF,
                 'param': {'lengthscales': .05,
                           'train_kernel_params': False}},
    "rbf_0.01": {'kernel': gpf.kernels.RBF,
                 'param': {'lengthscales': .01,
                           'train_kernel_params': False}},
    "rbf_auto": {'kernel': gpf.kernels.RBF,
                 'param': {'lengthscales': 1.,
                           'train_kernel_params': True}},
}

DEFAULT_KERN_FUNC_DICT_GPY = {
    "poly_1": {'kernel': gpy.kern.Poly,
               'param': {'order': 1.}},
    "poly_2": {'kernel': gpy.kern.Poly,
               'param': {'order': 2.}},
    "poly_3": {'kernel': gpy.kern.Poly,
               'param': {'order': 3.}},
    "rquad1": {'kernel': gpy.kern.RatQuad,
               'param': {'lengthscale': None, 'power': 1.}},
    "rquad2": {'kernel': gpy.kern.RatQuad,
               'param': {'lengthscale': None, 'power': 2.}},
    "period0.5": {'kernel': gpy.kern.StdPeriodic,
                  'param': {'lengthscale': None, 'period': .5}},
    "period1": {'kernel': gpy.kern.StdPeriodic,
                'param': {'lengthscale': None, 'period': 1.}},
    "period1.5": {'kernel': gpy.kern.StdPeriodic,
                  'param': {'lengthscale': None, 'period': 1.5}},
    "period_auto": {'kernel': gpy.kern.StdPeriodic,
                    'param': {'lengthscale': .15, 'period': None}},
    "matern12": {'kernel': gpy.kern.OU,
                 'param': {'lengthscale': None}},
    "matern32": {'kernel': gpy.kern.Matern32,
                 'param': {'lengthscale': None}},
    "matern52": {'kernel': gpy.kern.Matern52,
                 'param': {'lengthscale': None}},
    "rbf_1": {'kernel': gpf.kernels.RBF,
              'param': {'lengthscales': 1.,
                        'train_kernel_params': False}},
    "rbf_0.5": {'kernel': gpf.kernels.RBF,
                'param': {'lengthscales': .5,
                          'train_kernel_params': False}},
    "rbf_0.3": {'kernel': gpf.kernels.RBF,
                'param': {'lengthscales': .3,
                          'train_kernel_params': False}},
    "rbf_0.25": {'kernel': gpf.kernels.RBF,
                'param': {'lengthscales': .25,
                          'train_kernel_params': False}},
    "rbf_0.2": {'kernel': gpf.kernels.RBF,
                'param': {'lengthscales': .2,
                          'train_kernel_params': False}},
    "rbf_0.05": {'kernel': gpf.kernels.RBF,
                 'param': {'lengthscales': .05,
                           'train_kernel_params': False}},
    "rbf_auto": {'kernel': gpy.kern.RBF,
                 'param': {'lengthscale': None}},
    "mlp_auto": {'kernel': gpy.kern.MLP,
                 'param': {'weight_variance': 1.}},

}

DEFAULT_KERN_FUNC_DICT_RBF = {
    "rbf_0.5": {'kernel': gpf.kernels.RBF,
                'param': {'lengthscales': .5,
                          'train_kernel_params': False}},
    "rbf_0.4": {'kernel': gpf.kernels.RBF,
                 'param': {'lengthscales': .4,
                           'train_kernel_params': False}},
    "rbf_0.35": {'kernel': gpf.kernels.RBF,
                'param': {'lengthscales': .35,
                          'train_kernel_params': False}},
    "rbf_0.3": {'kernel': gpf.kernels.RBF,
                 'param': {'lengthscales': .3,
                           'train_kernel_params': False}},
    "rbf_0.275": {'kernel': gpf.kernels.RBF,
                 'param': {'lengthscales': .275,
                           'train_kernel_params': False}},
    "rbf_0.25": {'kernel': gpf.kernels.RBF,
                 'param': {'lengthscales': .25,
                           'train_kernel_params': False}},
    "rbf_0.2": {'kernel': gpf.kernels.RBF,
                'param': {'lengthscales': .2,
                          'train_kernel_params': False}},
    "rbf_0.15": {'kernel': gpf.kernels.RBF,
                 'param': {'lengthscales': .15,
                           'train_kernel_params': False}},
    "rbf_0.1": {'kernel': gpf.kernels.RBF,
                'param': {'lengthscales': .1,
                          'train_kernel_params': False}},
    "rbf_0.075": {'kernel': gpf.kernels.RBF,
                  'param': {'lengthscales': .075,
                            'train_kernel_params': False}},
    "rbf_0.05": {'kernel': gpf.kernels.RBF,
                 'param': {'lengthscales': .05,
                           'train_kernel_params': False}},
    "rbf_0.04": {'kernel': gpf.kernels.RBF,
                 'param': {'lengthscales': .04,
                           'train_kernel_params': False}},
    "rbf_0.03": {'kernel': gpf.kernels.RBF,
                 'param': {'lengthscales': .03,
                           'train_kernel_params': False}},
    "rbf_0.025": {'kernel': gpf.kernels.RBF,
                 'param': {'lengthscales': .025,
                           'train_kernel_params': False}},
    "rbf_0.02": {'kernel': gpf.kernels.RBF,
                 'param': {'lengthscales': .02,
                           'train_kernel_params': False}},
    "rbf_0.01": {'kernel': gpf.kernels.RBF,
                 'param': {'lengthscales': .01,
                           'train_kernel_params': False}},
}

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Predictive functions, GPflow Implementation """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def fit_gpflow(X_train, y_train,
               X_test, X_valid,
               kern_func=None, n_step=10000,
               **kwargs):
    """Fits GP regression using GPflow

    Args:
        X_train: (np.ndarray of float32) Training data (N_train, D).
        y_train: (np.ndarray of float32) Training labels (N_train, D).
        X_test: (np.ndarray of float32) Testintg features (N_test, D).
        X_valid: (np.ndarray of float32) Validation features (N_test, D).
        kern_func: (gpflow.kernels) GPflow kernel function.
        n_step: (int) number of optimization iterations.
        kwargs: Additional arguments passed to kern_func.

    Returns::
        mu, var: (np.ndarray) Posterior predictive mean/variance.
        par_val: (list of np.ndarray) List of model parameter values
        m: (gpflow.models.gpr) gpflow model object.
        k: (gpflow.kernels) gpflow kernel object.
    """
    if y_train.ndim == 1:
        y_train = np.expand_dims(y_train, 1)

    # define computation graph
    gpr_graph = tf.Graph()
    with gpr_graph.as_default():

        # define model
        if not kern_func:
            k = gpf.kernels.RBF(input_dim=X_train.shape[1], ARD=True)
        else:
            k = kern_func(input_dim=X_train.shape[1], **kwargs)

        m = gpf.models.GPR(X_train, y_train, kern=k)

        # define optimization
        objective = m.objective
        param_dict = {par.name: par.value for par in m.parameters}
        optimizer = tf.train.AdamOptimizer(5e-3)
        train_op = optimizer.minimize(objective)

        # define prediction
        pred_mu_test, pred_cov_test = m.predict_f(X_test)
        pred_mu_valid, pred_cov_valid = m.predict_f(X_valid)

        init_op = tf.global_variables_initializer()

        gpr_graph.finalize()

    # execute training
    with tf.Session(graph=gpr_graph) as sess:
        sess.run(init_op)
        for step in range(n_step):
            _, obj = sess.run([train_op, objective])

            if step % 1000 == 0:
                print('Iter {}: Loss = {}'.format(step, obj))

                # evaluate
                (mu_test, var_test,
                 mu_valid, var_valid, par_dict) = sess.run(
                    [pred_mu_test, pred_cov_test,
                     pred_mu_valid, pred_cov_valid, param_dict])

                mu_test, var_test = mu_test.squeeze(), var_test.squeeze()
                mu_valid, var_valid = mu_valid.squeeze(), var_valid.squeeze()

        sess.close()

    return mu_test, var_test, mu_valid, var_valid,


def fit_gpy(X_train, y_train,
            X_test, X_valid,
            kern_func=None, n_step=10000,
            **kwargs):
    """Fits GP regression using GPflow

    Args:
        X_train: (np.ndarray of float32) Training data (N_train, D).
        y_train: (np.ndarray of float32) Training labels (N_train, D).
        X_test: (np.ndarray of float32) Testintg features (N_test, D).
        X_valid: (np.ndarray of float32) Validation features (N_test, D).
        kern_func: (gpflow.kernels) GPflow kernel function.
        n_step: (int) number of optimization iterations.
        kwargs: Additional arguments passed to kern_func.

    Returns::
        mu, var: (np.ndarray) Posterior predictive mean/variance.
        par_val: (list of np.ndarray) List of model parameter values
        m: (gpflow.models.gpr) gpflow model object.
        k: (gpflow.kernels) gpflow kernel object.
    """
    if y_train.ndim == 1:
        y_train = np.expand_dims(y_train, 1)

    # define model
    if not kern_func:
        k = gpy.kern.RBF(input_dim=X_train.shape[1], ARD=True)
    else:
        k = kern_func(input_dim=X_train.shape[1], **kwargs)

    m = gpy.models.GPRegression(X_train, y_train, kernel=k)

    # define optimization
    m.optimize_restarts(num_restarts=10)

    # define prediction
    mu_test, var_test = m.predict(X_test)
    mu_valid, var_valid = m.predict(X_valid)

    return mu_test, var_test, mu_valid, var_valid,


def fit_base_gp_models(X_train, y_train,
                       X_test, y_test, X_valid, y_valid,
                       y_valid_rmse_id=None,
                       kern_func_dict=DEFAULT_KERN_FUNC_DICT_GPY,
                       n_valid_sample=5000,
                       n_train_step=20000,
                       save_addr_prefix="./plot/calibre/base",
                       **visual_kwargs):
    """Run GPflow-Slim GPR according to list of supplied kernel functions.

    Args:
        X_train: (np.ndarray of float32) Training data (N_train, D).
        y_train: (np.ndarray of float32) Training labels (N_train, D).
        X_test: (np.ndarray of float32) Testintg features (N_test, D).
        y_test: (np.ndarray of float32) Testing labels (N_train, D).
        X_valid: (np.ndarray of float32) Validation features (N_test, D).
        y_valid: (np.ndarray of float32) Validation labels (N_train, D).
        y_valid_rmse_id: (np.ndarray of int) Sample id in y_valid to compute
            RMSE over. If None then use entire y_valid
        kern_func_dict: (dict) Dictionary of kernel functions and kernel kwargs
            to pass to fit_gpflow.
            For example see calibre.util.gp_flow.DEFAULT_KERN_FUNC_DICT
        n_valid_sample: (int) Number of samples to draw from posterior predictive
            for validation predictions.
        n_train_step: (int) Number of training step.
        save_addr_prefix: (str) Prefix to save address for plots and
            model prediction/samples.
        **visual_kwargs: Additional keyword args to pass to gpr_1d_visual or
            gpr_2d_visual
    """
    pathlib.Path(save_addr_prefix).mkdir(parents=True, exist_ok=True)

    num_valid_obs = X_valid.shape[0]

    # obtain base model predictions
    test_pred_list = dict()
    valid_pred_list = dict()
    valid_samp_list = dict()

    for kern_name, kern_obj in kern_func_dict.items():
        print("\n\nFitting {}".format(kern_name))
        kern_func, kern_pars = kern_obj.values()
        # decide kernel function
        if _get_module_name(kern_func) == "GPy":
            fit_gp_func = fit_gpy
        elif _get_module_name(kern_func) == "gpflowSlim":
            fit_gp_func = fit_gpflow
        else:
            raise ValueError("supplied kernel function must from either "
                             "'GPy' or 'GPFlowSlim' ")

        (mu_test, var_test, mu_valid, var_valid,
         ) = fit_gp_func(X_train, y_train,
                         X_test, X_valid,
                         n_step=n_train_step,
                         kern_func=kern_func, **kern_pars)

        # store variable names
        mu_test = mu_test.squeeze().astype(np.float32)
        mu_valid = mu_valid.squeeze().astype(np.float32)

        test_pred_list[kern_name] = mu_test.squeeze()
        valid_pred_list[kern_name] = mu_valid.squeeze()
        valid_samp_list[kern_name] = (
                np.expand_dims(mu_valid.squeeze(), -1) +
                np.random.normal(size=(num_valid_obs, n_valid_sample)) *
                np.expand_dims(np.sqrt(var_valid.squeeze()), -1)
        )

        # visualization
        if X_train.size == y_train.size:
            visual_util.gpr_1d_visual(
                pred_mean=mu_valid, pred_cov=var_valid,
                X_train=X_train, y_train=y_train,
                X_test=X_valid, y_test=y_valid,
                compute_rmse=True, rmse_id=y_valid_rmse_id,
                title=kern_name,
                save_addr="{}/fit/{}.png".format(save_addr_prefix, kern_name),
                **visual_kwargs)
        elif X_train.squeeze().ndim == 2:
            visual_util.gpr_2d_visual(
                pred_mean=mu_valid, pred_cov=var_valid,
                X_train=X_train, y_train=y_train,
                X_test=X_valid, y_test=y_valid,
                title=kern_name,
                save_addr="{}/fit/{}.png".format(save_addr_prefix, kern_name),
                **visual_kwargs)

        visual_util.prob_calibration_1d(
            y_valid, valid_samp_list[kern_name],
            title=kern_name,
            save_addr="{}/reliability/{}_prob.png".format(
                save_addr_prefix, kern_name))

        visual_util.coverage_index_1d(
            y_valid, valid_samp_list[kern_name],
            title=kern_name,
            save_addr="{}/reliability/{}_coverage.png".format(
                save_addr_prefix, kern_name))

    # save test/validation prediction, and also validation samples
    with open('{}/base_test_pred.pkl'.format(save_addr_prefix), 'wb') as file:
        pk.dump(test_pred_list, file, protocol=pk.HIGHEST_PROTOCOL)

    with open('{}/base_valid_pred.pkl'.format(save_addr_prefix), 'wb') as file:
        pk.dump(valid_pred_list, file, protocol=pk.HIGHEST_PROTOCOL)

    with open('{}/base_valid_sample.pkl'.format(save_addr_prefix), 'wb') as file:
        pk.dump(valid_samp_list, file, protocol=pk.HIGHEST_PROTOCOL)


def _get_module_name(function):
    """Extracts module signature of a function."""
    full_name = function.__module__
    return full_name.split(".")[0]
