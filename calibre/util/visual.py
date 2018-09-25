"""Utility functions for visualization"""
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import calibre.util.evaluation as eval_util


def gpr_1d_visual(pred_mean_test, pred_cov_test,
                  X_train=None, y_train=None, X_test=None, y_test=None,
                  X_induce=None, title="", save_addr=""):
    """Plots the GP posterior predictive mean and uncertainty.

    Args:
        pred_mean_test: (np.ndarray) posterior predictive mean at X_test
        pred_cov_test: (np.ndarray) posterior predictive variance at X_test
        X_train: (np.ndarray) X values in training dataset.
        y_train: (np.ndarray) y values in training dataset.
        X_test: (np.ndarray) X values in test dataset.
        y_test: (np.ndarray) y values in test dataset.
        X_induce: (np.ndarray)  X values marking the position of inducing points.
        title: (str) Title of the image.
        save_addr: (str) Address to save image to.
    """
    if save_addr:
        plt.ioff()

    fig, ax = plt.subplots()
    if isinstance(X_test, np.ndarray):
        ax.plot(X_test.squeeze(), y_test.squeeze(), c='black')
    if isinstance(X_train, np.ndarray):
        ax.plot(X_train.squeeze(), y_train.squeeze(), 'o', c='red',
                markeredgecolor='black')
    if isinstance(X_induce, np.ndarray):
        for x_vertical in X_induce:
            plt.axvline(x=x_vertical, c='black', alpha=.2)

    ax.plot(X_test, pred_mean_test, c='blue', alpha=.5)
    ax.fill_between(X_test.squeeze(),
                    pred_mean_test + np.sqrt(pred_cov_test),
                    pred_mean_test - np.sqrt(pred_cov_test),
                    color='black', alpha=.1, edgecolor=None, linewidth=0.0)
    ax.fill_between(X_test.squeeze(),
                    pred_mean_test + 2 * np.sqrt(pred_cov_test),
                    pred_mean_test - 2 * np.sqrt(pred_cov_test),
                    color='black', alpha=.1, edgecolor=None, linewidth=0.0)
    ax.fill_between(X_test.squeeze(),
                    pred_mean_test + 3 * np.sqrt(pred_cov_test),
                    pred_mean_test - 3 * np.sqrt(pred_cov_test),
                    color='black', alpha=.1, edgecolor=None, linewidth=0.0)

    plt.title(title)
    plt.ylim([-4.5, 4.5])

    if save_addr:
        plt.savefig(save_addr)
        plt.close()
        plt.ion()


def plot_base_prediction(base_pred, X_valid, y_valid=None,
                         X_train=None, y_train=None, save_addr="", **kwargs):
    if save_addr:
        plt.ioff()

    # prepare for plotting predictions
    sns_data = pd.DataFrame(
        {"x": np.tile(X_valid.squeeze(), reps=len(base_pred)),
         "y": np.concatenate(list(base_pred.values())),
         "model": np.repeat(list(base_pred.keys()), repeats=X_valid.shape[0])})

    # plot baseline predictions.
    if isinstance(X_train, np.ndarray):
        plt.plot(X_train.squeeze(), y_train.squeeze(),
                 'o', c='red', markeredgecolor='black')
    if isinstance(y_valid, np.ndarray):
        plt.plot(X_valid, y_valid, c='black')

    sns.lineplot(x="x", y="y", hue="model", alpha=0.7,
                 data=sns_data, **kwargs)
    plt.ylim(-4, 4)
    plt.title("Base Model Predictions")
    plt.legend(loc='lower left')

    if save_addr:
        plt.savefig(save_addr)
        plt.close()
        plt.ion()


def plot_ensemble_weight_1d(X, weight_sample, model_names="",
                            save_addr_prefix=""):
    """Plots the posterior mean and median of weight sample for K models.

    Args:
        X: (np.ndarray of float32) A 1D array of feature values, dimension (N_obs, )
        weight_sample: (np.ndarray of float32) Sample of model ensemble weights
            dimension (N_sample, N_obs, num_models).
        model_names: (list of str) list of model names, dimension (num_models, ).
        save_addr_prefix: (str) Prefix for save address.
    """
    _, _, num_models = weight_sample.shape

    weight_mean = np.nanmean(weight_sample, axis=0)
    weight_median = np.nanpercentile(weight_sample, q=50, axis=0)
    weight_lower = np.nanpercentile(weight_sample, q=25, axis=0)
    weight_upper = np.nanpercentile(weight_sample, q=75, axis=0)

    # plot posterior median
    if save_addr_prefix:
        plt.ioff()

    fig_med, ax_med = plt.subplots(1, 1)
    for k in range(num_models):
        # plot median
        ax_med.plot(X.squeeze(), weight_median[:, k],
                    label=model_names[k] if model_names else "")
        # plot 50% confidence interval
        ax_med.fill_between(X.squeeze(),
                            y1=weight_lower[:, k], y2=weight_upper[:, k],
                            alpha=0.1)

    plt.ylim(-0.05, 1.05)
    plt.title("Ensemble Weights, Posterior Mode")
    if model_names:
        plt.legend(loc='upper left')
    if save_addr_prefix:
        plt.savefig("{}_weight_median.png".format(save_addr_prefix))
        plt.close()
        plt.ion()

    # plot posterior mean
    if save_addr_prefix:
        plt.ioff()

    fig_mean, ax_mean = plt.subplots(1, 1)
    for k in range(num_models):
        ax_mean.plot(X.squeeze(), weight_mean[:, k],
                     label=model_names[k] if model_names else "")

    plt.ylim(-0.05, 1.05)
    plt.title("Ensemble Weights, Posterior Mean")
    if model_names:
        plt.legend(loc='upper left')
    if save_addr_prefix:
        plt.savefig("{}_weight_mean.png".format(save_addr_prefix))
        plt.close()
        plt.ion()


def plot_reliability_diagram_1d(Y_obs, Y_sample,
                                title="", save_addr=""):
    """Plots the reliability diagram (i.e. CDF for F^{-1}(y) ) for 1D prediction.

    Args:
        Y_obs: (np.ndarray of float32) N observations of dim (N, 1)
        Y_sample: (np.ndarray of float32) Samples of size M corresponding
        to the N observations. dim (N, M)
        title: (str) Title of the image.
        save_addr: (str) Address to save image to.
    """
    # TODO(jereliu): extend to multivariate setting.

    if save_addr:
        plt.ioff()

    ecdf_sample = eval_util.ecdf_eval(Y_obs, Y_sample)
    ecdf_func = eval_util.make_empirical_cdf_1d(ecdf_sample)

    ecdf_eval = np.linspace(0, 1, 1000)
    plt.plot(ecdf_eval, ecdf_eval, c="black")
    plt.plot(ecdf_eval, ecdf_func(ecdf_eval))
    plt.title("Reliability diagram, {}".format(title))

    if save_addr:
        plt.savefig(save_addr)
        plt.close()
        plt.ion()
