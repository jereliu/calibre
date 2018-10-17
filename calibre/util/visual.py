"""Utility functions for visualization"""
import os
import pathlib

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import calibre.util.metric as metric_util


def gpr_1d_visual(pred_mean,
                  pred_cov=None, pred_quantiles=[],
                  pred_samples=None,
                  X_train=None, y_train=None,
                  X_test=None, y_test=None, X_induce=None,
                  quantile_colors=None, quantile_alpha=0.1,
                  y_range=[-4.5, 4.5], add_reference=False,
                  title="", save_addr="",
                  ):
    """Plots the GP posterior predictive mean and uncertainty.

    Args:
        pred_mean: (np.ndarray) posterior predictive mean at X_test
        pred_cov: (np.ndarray or None) posterior predictive variance at X_test
        pred_quantiles: (list of tuples) list of tuples of (upper, lower)
            of np.ndarrays for the predictive quantiles.
            Ignored if pred_cov is not None.
        pred_samples: (list of np.ndarray) list of np.ndarray of samples from posterior.
        X_train: (np.ndarray) X values in training dataset.
        y_train: (np.ndarray) y values in training dataset.
        X_test: (np.ndarray) X values in test dataset.
        y_test: (np.ndarray) y values in test dataset.
        X_induce: (np.ndarray)  X values marking the position of inducing points.
        title: (str) Title of the image.
        save_addr: (str) Address to save image to.

    Raises:
        (ValueError) If y_test is not multiple of X_test.
    """
    # TODO(jereliu): Write a save function decorator.
    if save_addr:
        pathlib.Path(save_addr).parent.mkdir(parents=True, exist_ok=True)
        plt.ioff()

    fig, ax = plt.subplots()

    # plot predictions:
    X_test = np.unique(X_test, axis=0)

    # posterior predictive
    if isinstance(pred_mean, np.ndarray):
        pred_mean = pred_mean.squeeze()[:len(X_test)]
        ax.plot(X_test.squeeze(), pred_mean.squeeze(),
                c='blue', alpha=.5)

    # posterior confidence interval based on std
    if isinstance(pred_cov, np.ndarray):
        pred_cov = pred_cov.squeeze()[:len(X_test)]
        # compute the three sets of predictive quantiles (mean +\- 3*sd)
        pred_quantiles = [(pred_mean + np.sqrt(pred_cov),
                           pred_mean - np.sqrt(pred_cov)),
                          (pred_mean + 2 * np.sqrt(pred_cov),
                           pred_mean - 2 * np.sqrt(pred_cov)),
                          (pred_mean + 3 * np.sqrt(pred_cov),
                           pred_mean - 3 * np.sqrt(pred_cov))]

    # posterior quantile
    if isinstance(pred_quantiles, list):
        if quantile_colors is None:
            quantile_colors = ["black"] * len(pred_quantiles)

        for id, (upper, lower) in enumerate(pred_quantiles):
            upper = upper.squeeze()[:len(X_test)]
            lower = lower.squeeze()[:len(X_test)]
            ax.fill_between(X_test.squeeze(), upper, lower,
                            color=quantile_colors[id],
                            alpha=quantile_alpha, edgecolor=None, linewidth=0.0)

    # posterior samples
    if isinstance(pred_samples, list):
        for pred_sample in pred_samples:
            pred_sample = pred_sample.squeeze()[:len(X_test)]
            ax.plot(X_test.squeeze(), pred_sample,
                    color='teal', alpha=.01, linewidth=2)

    # plot ground truth
    if isinstance(X_test, np.ndarray):
        y_X_ratio = len(y_test) / len(X_test)
        if y_X_ratio.is_integer():
            y_X_ratio = int(y_X_ratio)
            for fold_index in range(y_X_ratio):
                index_start = int(fold_index * len(X_test))
                index_end = int((fold_index + 1) * len(X_test))
                y_test_plot = y_test.squeeze()[index_start:index_end]
                ax.plot(X_test.squeeze(), y_test_plot, c='black')
        else:
            raise ValueError("y_test must be multiple of X_test.")

    if isinstance(X_train, np.ndarray):
        ax.plot(X_train.squeeze(), y_train.squeeze(), 'o', c='red',
                markeredgecolor='black')
    if isinstance(X_induce, np.ndarray):
        for x_vertical in X_induce:
            plt.axvline(x=x_vertical, c='black', alpha=.2)

    plt.title(title)
    if y_range is not None:
        plt.ylim(y_range)

    if add_reference:
        plt.axhline(y=0, c='black')

    if save_addr:
        plt.savefig(save_addr)
        plt.close()
        plt.ion()


def gpr_2d_visual(pred_mean, pred_cov,
                  X_train, y_train, X_test, y_test,
                  title="", save_addr=""):
    if save_addr:
        pathlib.Path(save_addr).parent.mkdir(parents=True, exist_ok=True)
        plt.ioff()

    # prediction surface
    n_reshape = int(np.sqrt(pred_mean.size))
    pred_mean_plot = pred_mean.reshape(n_reshape, n_reshape)
    X_valid = X_test.reshape(n_reshape, n_reshape, 2)
    x_grid, y_grid = X_valid[:, :, 0], X_valid[:, :, 1]

    ax = plt.axes(projection='3d')
    if isinstance(X_train, np.ndarray):
        ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c="black")
    ax.plot_surface(X=x_grid, Y=y_grid, Z=pred_mean_plot, cmap='inferno')
    ax.set_zlim(np.min(y_test), np.max(y_test))

    # optionally, compute RMSE
    if pred_mean.size == y_test.size:
        rmse = np.mean((pred_mean - y_test) ** 2)
        title = "{}, RMSE={:.4f}".format(title, rmse)

    plt.title(title)

    if save_addr:
        plt.savefig(save_addr)
        plt.close()
        plt.ion()


def plot_base_prediction(base_pred,
                         X_valid, y_valid=None,
                         X_train=None, y_train=None,
                         ax=None,
                         save_addr="", **kwargs):
    if save_addr:
        pathlib.Path(save_addr).parent.mkdir(parents=True, exist_ok=True)
        plt.ioff()

    # prepare for plotting predictions
    sns_data = pd.DataFrame(
        {"x": np.tile(X_valid.squeeze(), reps=len(base_pred)),
         "y": np.concatenate(list(base_pred.values())),
         "model": np.repeat(list(base_pred.keys()), repeats=X_valid.shape[0])})

    # plot baseline predictions.
    if not ax:
        fig, ax = plt.subplots(1, 1)
    if isinstance(X_train, np.ndarray):
        ax.plot(X_train.squeeze(), y_train.squeeze(),
                'o', c='red', markeredgecolor='black')
    if isinstance(y_valid, np.ndarray):
        ax.plot(X_valid, y_valid, c='black')

    sns.lineplot(x="x", y="y", hue="model", alpha=0.7,
                 data=sns_data, ax=ax, **kwargs)
    ax.set_ylim(-4, 4)
    ax.set_title("Base Model Predictions")
    ax.legend(loc='lower left')

    if save_addr:
        plt.savefig(save_addr)
        plt.close()
        plt.ion()


def plot_ensemble_weight_mean_1d(X, weight_sample, model_names="",
                                 ax_mean=None,
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

    # plot posterior mean
    if save_addr_prefix:
        pathlib.Path(save_addr_prefix).mkdir(parents=True, exist_ok=True)
        plt.ioff()

    if not ax_mean:
        fig_mean, ax_mean = plt.subplots(1, 1)

    for k in range(num_models):
        ax_mean.plot(X.squeeze(), weight_mean[:, k],
                     label=model_names[k] if model_names else "")

    ax_mean.set_ylim(-0.05, 1.05)
    ax_mean.set_title("Ensemble Weights, Posterior Mean")
    if model_names:
        ax_mean.legend(loc='upper left')
    if save_addr_prefix:
        plt.savefig("{}_weight_mean.png".format(save_addr_prefix))
        plt.close()
        plt.ion()


def plot_ensemble_weight_median_1d(X, weight_sample, model_names="",
                                   ax_median=None,
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

    weight_median = np.nanpercentile(weight_sample, q=50, axis=0)
    weight_lower = np.nanpercentile(weight_sample, q=25, axis=0)
    weight_upper = np.nanpercentile(weight_sample, q=75, axis=0)

    # plot posterior median
    if save_addr_prefix:
        pathlib.Path(save_addr_prefix).mkdir(parents=True, exist_ok=True)
        plt.ioff()

    if not ax_median:
        fig_med, ax_median = plt.subplots(1, 1)

    for k in range(num_models):
        # plot median
        ax_median.plot(X.squeeze(), weight_median[:, k],
                       label=model_names[k] if model_names else "")
        # plot 50% confidence interval
        ax_median.fill_between(X.squeeze(),
                               y1=weight_lower[:, k], y2=weight_upper[:, k],
                               alpha=0.1)

    ax_median.set_ylim(-0.05, 1.05)
    ax_median.set_title("Ensemble Weights, Posterior Mode")
    if model_names:
        ax_median.legend(loc='upper left')
    if save_addr_prefix:
        plt.savefig("{}_weight_median.png".format(save_addr_prefix))
        plt.close()
        plt.ion()

    # plot posterior mean
    if save_addr_prefix:
        plt.ioff()


def plot_ensemble_weight_mean_2d(X, weight_sample, model_names,
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

    # plot posterior mean
    if save_addr_prefix:
        pathlib.Path("{}/weight_mean/".format(save_addr_prefix)).mkdir(
            parents=True, exist_ok=True)

    for k in range(num_models):
        gpr_2d_visual(weight_mean[:, k], None,
                      None, None, X, np.array([-0.05, 1.05]),
                      title="Ensemble Posterior Mean, {}".format(model_names[k]),
                      save_addr="{}/weight_mean/{}.png".format(
                          save_addr_prefix, model_names[k]))


def prob_calibration_1d(Y_obs, Y_sample, title="", save_addr=""):
    """Plots the reliability diagram (i.e. CDF for F^{-1}(y) ) for 1D prediction.

    Args:
        Y_obs: (np.ndarray of float32) N observations of dim (N, 1)
        Y_sample: (np.ndarray of float32) Samples of size M corresponding
        to the N observations. dim (N, M)
        title: (str) Title of the image.
        save_addr: (str) Address to save image to.
    """

    if save_addr:
        pathlib.Path(save_addr).parent.mkdir(parents=True, exist_ok=True)
        plt.ioff()

    ecdf_sample = metric_util.ecdf_eval(Y_obs, Y_sample)
    ecdf_func = metric_util.make_empirical_cdf_1d(ecdf_sample)

    ecdf_eval = np.linspace(0, 1, 1000)
    ecdf_valu = ecdf_func(ecdf_eval)

    fig, ax = plt.subplots()
    ax.plot(ecdf_eval, ecdf_eval, c="black")
    ax.plot(ecdf_eval, ecdf_valu)
    total_variation = np.mean(np.abs(ecdf_eval - ecdf_valu))
    plt.title("Probabilistic Calibration, {}, Score: {:.3f}".format(
        title, total_variation))

    if save_addr:
        plt.savefig(save_addr)
        plt.close()
        plt.ion()


def marginal_calibration_1d(Y_obs, Y_sample, title="", save_addr=""):
    """Plots the reliability diagram (i.e. CDF for F^{-1}(y) ) for 1D prediction.

    Args:
        Y_obs: (np.ndarray of float32) N observations of dim (N, 1)
        Y_sample: (np.ndarray of float32) Monte Carlo Samples of size M
            corresponding to the N observations. dim (N, M)
        title: (str) Title of the image.
        save_addr: (str) Address to save image to.
    """

    if save_addr:
        pathlib.Path(save_addr).parent.mkdir(parents=True, exist_ok=True)
        plt.ioff()

    ecdf_eval = np.linspace(np.min(Y_obs), np.max(Y_obs), 1000)

    ecdf_obsv = metric_util.make_empirical_cdf_1d(Y_obs)
    ecdf_pred = metric_util.make_empirical_cdf_1d(Y_sample)

    ecdf_sample_obsv = ecdf_obsv(ecdf_eval)
    ecdf_sample_pred = ecdf_pred(ecdf_eval)

    fig, ax = plt.subplots()
    ax.plot((0, 1), (0, 1), c="black")
    ax.plot(ecdf_sample_obsv, ecdf_sample_pred)
    plt.xlabel("Empirical Distribution")
    plt.ylabel("Predictive Distribution")
    plt.title("Marginal Calibration, {}".format(title))

    if save_addr:
        plt.savefig(save_addr)
        plt.close()
        plt.ion()


def corr_matrix(corr_mat, ax=None, model_names="auto", save_addr=""):
    """Visualize correlation matrix."""
    if save_addr:
        pathlib.Path(save_addr).parent.mkdir(parents=True, exist_ok=True)
        plt.ioff()

    if not ax:
        fig, ax = plt.subplots(1, 1)

    # mask upper triangle
    mask = np.zeros_like(corr_mat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    cmap = sns.diverging_palette(240, 10, sep=160, n=256, as_cmap=True)

    sns.heatmap(corr_mat,
                mask=mask, cmap=cmap,
                annot=True, annot_kws={'color': 'white'},
                xticklabels=model_names,
                yticklabels=model_names,
                vmin=-1., vmax=1., center=0,
                square=True, linewidths=.5,
                ax=ax)
    plt.yticks(rotation=0)

    if save_addr:
        plt.savefig(save_addr)
        plt.close()
        plt.ion()


def model_composition_1d(X_value, corr_mat, weight_sample,
                         base_pred, X_valid, y_valid, X_train, y_train,
                         model_names, save_addr=""):
    """Plot aligned graph with base prediction at left and correlation at right."""
    if save_addr:
        pathlib.Path(save_addr).parent.mkdir(parents=True, exist_ok=True)
        plt.ioff()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # First plot: Base Model Fit
    plot_base_prediction(base_pred=base_pred,
                         X_valid=X_valid, y_valid=y_valid,
                         X_train=X_train, y_train=y_train, ax=ax1)
    ax1.axvline(X_value, c='red', alpha=0.5, linewidth=2)
    ax1.set(adjustable='box-forced')

    # Second plot: Mean prediction
    plot_ensemble_weight_mean_1d(X=X_valid,
                                 weight_sample=weight_sample,
                                 model_names=model_names,
                                 ax_mean=ax2)
    ax2.axvline(X_value, c='red', alpha=0.5, linewidth=2)
    ax2.set(adjustable='box-forced')

    corr_matrix(corr_mat, model_names=model_names, ax=ax3)
    ax3.set_title("X={}".format(X_value))
    ax3.set(adjustable='box-forced')

    if save_addr:
        plt.savefig(save_addr)
        plt.close()
        plt.ion()
