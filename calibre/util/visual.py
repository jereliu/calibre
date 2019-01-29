"""Utility functions for visualization"""
import os
import pathlib

import tqdm

import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.signal as signal

import statsmodels.nonparametric.api as smnp

import matplotlib.pyplot as plt
import seaborn as sns

from calibre.calibration import coverage

import calibre.util.metric as metric_util

from matplotlib.colors import BoundaryNorm


def gpr_1d_visual(pred_mean,
                  pred_cov=None, pred_quantiles=[],
                  pred_samples=None,
                  X_train=None, y_train=None,
                  X_test=None, y_test=None, X_induce=None,
                  compute_rmse=True, rmse_id=None,
                  quantile_colors=None, quantile_alpha=0.1,
                  y_range=None, add_reference=False,
                  quantile_shade_legend=None,
                  title="", save_addr="", fontsize=12,
                  quantile_colors_norm=None, ax=None,
                  smooth_mean=False, smooth_quantile=True,
                  pred_mean_color='blue',
                  pred_mean_alpha=0.25, figsize=None):
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
        compute_rmse: (bool) Whether to compute test RMSE.
        rmse_id: (np.ndarray of int or None) Subset of X_test to compute
            rmse on. If None then all X_test are used.
        quantile_shade_legend: (list of str or None) Legend names for quantile shades. If None then no
            legend will be added.
        title: (str) Title of the image.
        save_addr: (str) Address to save image to.
        fontsize: (int) font size for title and axis labels

    Raises:
        (ValueError) If y_test is not multiple of X_test.
    """
    # TODO(jereliu): Write a save function decorator.
    if save_addr:
        pathlib.Path(save_addr).parent.mkdir(parents=True, exist_ok=True)
        plt.ioff()

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # plot predictions:
    X_test = np.unique(X_test, axis=0)

    # posterior predictive
    if isinstance(pred_mean, np.ndarray):
        pred_mean = pred_mean.squeeze()[:len(X_test)]

        if smooth_mean:
            # compute window length for filter
            window_len = len(pred_mean) // 15
            if window_len % 2 == 0:
                # savgol_filter requires odd window size
                window_len = window_len + 1

            pred_mean = signal.savgol_filter(pred_mean, window_len, polyorder=3)

        ax.plot(X_test.squeeze(), pred_mean.squeeze(),
                c=pred_mean_color, alpha=pred_mean_alpha)

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

        shade_list = []

        if isinstance(quantile_alpha, float):
            quantile_alpha = [quantile_alpha]

        if len(quantile_alpha) == 1:
            quantile_alpha = list(quantile_alpha) * len(pred_quantiles)

        for id, (upper, lower) in enumerate(pred_quantiles):
            upper = upper.squeeze()[:len(X_test)]
            lower = lower.squeeze()[:len(X_test)]

            if smooth_quantile:
                # compute window length for filter
                window_len = len(upper) // 8
                if window_len % 2 == 0:
                    # savgol_filter requires odd window size
                    window_len = window_len + 1

                upper = signal.savgol_filter(upper, window_len, polyorder=3)
                lower = signal.savgol_filter(lower, window_len, polyorder=3)

            if isinstance(quantile_colors, np.ndarray):
                quantile_shade = rainbow_fill_between(ax, X_test.squeeze(), upper, lower,
                                                      colors=quantile_colors,
                                                      norm=quantile_colors_norm,
                                                      alpha=quantile_alpha[id])
            else:
                # first wash out previous color
                ax.fill_between(X_test.squeeze(), upper, lower,
                                color="white",
                                edgecolor=None, linewidth=0.0)
                quantile_shade = ax.fill_between(X_test.squeeze(), upper, lower,
                                                 color=quantile_colors[id],
                                                 alpha=quantile_alpha[id],
                                                 edgecolor=None, linewidth=0.0)

            shade_list.append(quantile_shade)

            if quantile_shade_legend:
                plt.legend(shade_list, quantile_shade_legend)

    # plot training data
    if isinstance(X_train, np.ndarray):
        if X_train.size < 50:
            ax.plot(X_train.squeeze(), y_train.squeeze(), 'o',
                    c='red', markeredgecolor='black')
        elif X_train.size < 100:
            ax.plot(X_train.squeeze(), y_train.squeeze(), '.',
                    c='red', alpha=.5)
        else:
            ax.scatter(X_train.squeeze(), y_train.squeeze(), marker='.',
                       c='red', alpha=.5, s=1)

    if isinstance(X_induce, np.ndarray):
        for x_vertical in X_induce:
            plt.axvline(x=x_vertical, c='black', alpha=.05)

    # posterior samples
    if isinstance(pred_samples, list):
        for pred_sample in pred_samples:
            pred_sample = pred_sample.squeeze()[:len(X_test)]
            ax.plot(X_test.squeeze(), pred_sample,
                    color='teal', alpha=.01, linewidth=2)

    # plot ground truth
    if y_test is not None:
        # compute rmse
        if compute_rmse and pred_mean is not None:
            if isinstance(rmse_id, np.ndarray):
                test_rmse = metric_util.rmse(y_test[rmse_id],
                                             pred_mean[rmse_id])
            else:
                test_rmse = metric_util.rmse(y_test, pred_mean)

            title = '{}, RMSE={:.4f}'.format(title, test_rmse)

        # plot y_test
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

    ax.set_title(title, fontsize=fontsize)

    if y_range is not None:
        ax.set_ylim(y_range)

    if add_reference:
        ax.axhline(y=0, c='black')

    if save_addr:
        plt.savefig(save_addr)
        plt.close()
        plt.ion()

    return ax


def gpr_2d_visual(pred_mean, pred_cov,
                  X_train, y_train, X_test, y_test,
                  title="", save_addr="", fontsize=12):
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
        rmse = metric_util.rmse(y_test, pred_mean)
        title = "{}, RMSE={:.4f}".format(title, rmse)

    plt.title(title, fontsize=fontsize)

    if save_addr:
        plt.savefig(save_addr)
        plt.close()
        plt.ion()


def plot_base_prediction(base_pred, model_names,
                         X_valid, y_valid=None,
                         X_train=None, y_train=None,
                         X_test=None, y_test=None,
                         ax=None, y_range=[-4.5, 4.5],
                         save_addr="", title_size=12, legend_size=12,
                         **kwargs):
    if save_addr:
        pathlib.Path(save_addr).parent.mkdir(parents=True, exist_ok=True)
        plt.ioff()

    base_pred_plot = np.asarray([base_pred[model_name]
                                 for model_name in model_names])

    # prepare for plotting predictions
    sns_data = pd.DataFrame(
        {"x": np.tile(X_valid.squeeze(), reps=len(base_pred)),
         "y": np.concatenate(base_pred_plot),
         "model": np.repeat(model_names, repeats=X_valid.shape[0])})

    # plot baseline predictions.
    if not ax:
        fig, ax = plt.subplots(1, 1)

    sns.lineplot(x="x", y="y", hue="model", alpha=0.7,
                 data=sns_data, ax=ax, **kwargs)

    if isinstance(y_train, np.ndarray):
        ax.plot(X_train.squeeze(), y_train.squeeze(),
                'o', c='red', markeredgecolor='black')
    if isinstance(y_test, np.ndarray):
        ax.plot(X_test.squeeze(), y_test.squeeze(),
                'o', c='blue', markeredgecolor='black')
    if isinstance(y_valid, np.ndarray):
        ax.plot(X_valid, y_valid, c='black')

    if y_range is not None:
        ax.set_ylim(y_range)
    ax.set_title("Base Model Predictions", fontsize=title_size)
    ax.legend(loc='lower left', prop={'size': legend_size})

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
        fontsize: (int) font size for title and axis labels
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
    ax_median.set_title("Ensemble Weights, Posterior Median")
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


def prob_calibration_1d(Y_obs, Y_sample, title="", save_addr="", fontsize=12):
    """Plots the reliability diagram (i.e. CDF for F^{-1}(y) ) for 1D prediction.

    Args:
        Y_obs: (np.ndarray of float32) N observations of dim (N, 1)
        Y_sample: (np.ndarray of float32) Samples of size M corresponding
        to the N observations. dim (N, M)
        title: (str) Title of the image.
        save_addr: (str) Address to save image to.
        fontsize: (int) font size for title and axis labels
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
    plt.title("Reliability Index, {}, Score: {:.3f}".format(
        title, total_variation), fontsize=fontsize)
    plt.xlabel(r"Empirical CDF for $\hat{F}(Y_i|X_i)$", fontsize=fontsize)
    plt.ylabel("Expected CDF $Uniform(0, 1)$", fontsize=fontsize)

    if save_addr:
        plt.savefig(save_addr)
        plt.close()
        plt.ion()


def coverage_index_1d(Y_obs, Y_sample, title="", save_addr="", fontsize=12):
    """Plots the reliability diagram (i.e. CDF for F^{-1}(y) ) for 1D prediction.

    Args:
        Y_obs: (np.ndarray of float32) N observations of dim (N_obs, 1)
        Y_sample: (np.ndarray of float32) Samples of size M corresponding
        to the N observations. dim (N_obs, N_sample)
        title: (str) Title of the image.
        save_addr: (str) Address to save image to.
        fontsize: (int) font size for title and axis labels
    """

    if save_addr:
        pathlib.Path(save_addr).parent.mkdir(parents=True, exist_ok=True)
        plt.ioff()

    nom_coverage, obs_coverage = coverage.credible_interval_coverage(
        Y_obs, Y_sample)

    fig, ax = plt.subplots()
    ax.plot(nom_coverage, nom_coverage, c="black")
    ax.plot(nom_coverage, obs_coverage)
    total_variation = np.mean(np.abs(obs_coverage - nom_coverage))
    plt.title("Coverage Index, {}, Score: {:.3f}".format(
        title, total_variation), fontsize=fontsize)
    plt.xlabel("Claimed Credible Interval Coverage", fontsize=fontsize)
    plt.ylabel("Observed Credible Interval Coverage", fontsize=fontsize)

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


def posterior_heatmap_2d(plot_data, X,
                         X_monitor=None,
                         cmap='inferno_r',
                         norm=None, norm_method="percentile",
                         save_addr=''):
    """Plots colored 2d heatmap using scatterplot.

    Args:
        plot_data: (np.ndarray) plot data whose color to visualize over
            2D surface, shape (N, ).
        X: (np.ndarray) locations of the plot data, shape (N, 2).
        X_monitor: (np.ndarray or None) Locations to plot data points to.
        cmap: (str) Name of color map.
        norm: (BoundaryNorm or None) Norm values to adjust color map.
            If None then a new norm will be created according to norm_method.
        norm_method: (str) The name of method to compute norm values.
            See util.visual.make_color_norm for detail.
        save_addr: (str) Address to save image to.

    Returns:
        (matplotlib.colors.BoundaryNorm) A color norm object for color map
            to be passed to a matplotlib.pyplot function.
    """
    if save_addr:
        pathlib.Path(save_addr).parent.mkdir(parents=True, exist_ok=True)
        plt.ioff()

    if not norm:
        norm = make_color_norm(plot_data, method=norm_method)

    # 2d color plot using scatter
    plt.figure(figsize=(10, 8))
    plt.scatter(x=X[:, 0], y=X[:, 1],
                s=3,
                c=plot_data, cmap=cmap, norm=norm)
    cbar = plt.colorbar()

    # plot monitors
    if isinstance(X_monitor, np.ndarray):
        plt.scatter(x=X_monitor[:, 0], y=X_monitor[:, 1],
                    s=10, c='black')

    # adjust plot window
    plt.xlim((np.min(X[:, 0]), np.max(X[:, 0])))
    plt.ylim((np.min(X[:, 1]), np.max(X[:, 1])))

    if save_addr:
        plt.savefig(save_addr, bbox_inches='tight')
        plt.close()
        plt.ion()
    else:
        plt.show()

    return norm


def make_color_norm(color_data, method="percentile"):
    """Makes color palette norm for heatmap plots.

    Args:
        color_data: (np.ndarray or list) Either a single numpy array or
            a list of numpy array that records numeric values to adjust
            color map to.
        method: (str) The name of method to compute norm values:
            percentile: Adjust norm to the raw percentile of color_data.
            residual: Adjust norm to the symmetric range of
                [-min(abs(data)), -max(abs(data))].
                Color norm values will space out evenly in between the range.
            residual_percentile: Similar to 'residual'.
                But color norm values will be adjusted with respect to the
                percentile of abs(data).

    Returns:
        (matplotlib.colors.BoundaryNorm) A color norm object for color map
            to be passed to a matplotlib.pyplot function.
    """
    if isinstance(color_data, list):
        color_data = np.concatenate(color_data)

    if method == "percentile":
        levels = np.percentile(color_data,
                               np.linspace(0, 100, 101))
    elif method == "residual":
        abs_max = np.max(np.abs(color_data))
        levels = np.linspace(-abs_max, abs_max, 101)
    elif method == "residual_percentile":
        abs_levels = np.percentile(np.abs(color_data),
                                   np.linspace(0, 100, 101))
        levels = np.sort(np.concatenate([-abs_levels, abs_levels]))
    else:
        raise ValueError("Method {} is not supported".format(method))

    return BoundaryNorm(levels, 256)


def scaled_1d_kde_plot(data, shade, bandwidth='scott',
                       vertical=False, legend=False, ax=None,
                       density_scale=None, **kwargs):
    """Plot a univariate kernel density estimate on one of the axes.

    Adapted from _univariate_kdeplot from seaborn but allow user to
    scale densityu estimates using  density_scale.
    """
    if ax is None:
        ax = plt.gca()

    # Calculate the KDE
    kde = smnp.KDEUnivariate(data.astype('double'))
    kde.fit(bw=bandwidth)
    x, y = kde.support, kde.density

    if density_scale:
        y = density_scale * y / np.max(y)

    # Make sure the density is nonnegative
    y = np.amax(np.c_[np.zeros_like(y), y], axis=1)

    # Flip the data if the plot should be on the y axis
    if vertical:
        x, y = y, x

    # Check if a label was specified in the call
    label = kwargs.pop("label", None)

    # Otherwise check if the data object has a name
    if label is None and hasattr(data, "name"):
        label = data.name

    # Decide if we're going to add a legend
    legend = label is not None and legend
    label = "_nolegend_" if label is None else label

    # Use the active color cycle to find the plot color
    facecolor = kwargs.pop("facecolor", None)
    line, = ax.plot(x, y, **kwargs)
    color = line.get_color()
    line.remove()
    kwargs.pop("color", None)
    facecolor = color if facecolor is None else facecolor

    # Draw the KDE plot and, optionally, shade
    ax.plot(x, y, color=color, label=label, **kwargs)
    shade_kws = dict(
        facecolor=facecolor,
        alpha=kwargs.get("alpha", 0.25),
        clip_on=kwargs.get("clip_on", True),
        zorder=kwargs.get("zorder", 1),
    )
    if shade:
        if vertical:
            ax.fill_betweenx(y, 0, x, **shade_kws)
        else:
            ax.fill_between(x, 0, y, **shade_kws)

    # Set the density axis minimum to 0
    ax.set_ylim(0, auto=None)

    # Draw the legend here
    handles, labels = ax.get_legend_handles_labels()

    return ax, x, y


def add_vertical_segment(x, height, **kwargs):
    """Adds a vertical segment to plot."""
    plt.plot([x, x], [0, height], **kwargs)


def compare_local_cdf_1d(X_pred, y_post_sample, y_true_sample,
                         n_x_eval=100, n_cdf_eval=1000, n_max_sample=100,
                         y_eval_grid=None,
                         save_addr='', **local_ecdf_kwargs):
    """

    Args:
        X_pred: (np.ndarray) feature locations, size (N, 1)
        y_post_sample: (np.ndarray) y samples from model distribution, size (N, M_post_sample)
        y_true_sample: (np.ndarray) y samples from true distribution. size (N, M_true_sample)
        n_x_eval: (int) Number of locations to compute cdfs at within range of X_eval .
        n_cdf_eval: (int) Number of cdf evaluations.
        n_max_sample: (int) Maximum number of sample to take to compute ecdf.
        save_addr: (str) Parent address to save figures to.

    Raises:
        (ValueError) If save_addr is None
    """
    if not save_addr:
        raise ValueError('save_addr cannot be None.')

    local_ecdf_kwargs['y_eval_grid'] = y_eval_grid

    (ecdf_diff, ecdf_true, ecdf_modl,
     X_eval, y_eval_grid, X_pred, y_true_sample) = (
        metric_util.ecdf_l1_dist(X_pred, y_post_sample, y_true_sample,
                                 n_x_eval=n_x_eval, n_cdf_eval=n_cdf_eval,
                                 n_max_sample=n_max_sample,
                                 return_addtional_data=True,
                                 **local_ecdf_kwargs))

    if save_addr:
        os.makedirs(save_addr, exist_ok=True)
        plt.ioff()

    for x_id in tqdm.tqdm(range(len(X_eval))):
        save_name = os.path.join(save_addr, "{}.png".format(x_id))
        #
        plt.figure(figsize=(14, 6))
        plt.subplot(221)
        plt.scatter(X_pred, y_true_sample, marker='.', s=0.1)
        plt.axvline(x=X_eval[x_id], c='red')

        plt.subplot(223)
        plt.plot(X_eval, ecdf_diff)
        plt.axvline(x=X_eval[x_id], c='red')
        plt.ylim(0, 0.2)
        plt.title("L1 Distance = {:3f}".format(np.mean(ecdf_diff)))

        #
        plt.subplot(122)
        quantile_val = np.linspace(0, 1, n_cdf_eval)
        y_eval_data = y_eval_grid[x_id] if y_eval_grid.ndim > 1 else y_eval_grid
        plt.plot(y_eval_data, ecdf_modl[x_id])
        plt.plot(y_eval_data, ecdf_true[x_id])
        plt.title("x = {:.3f}".format(X_eval[x_id]))
        plt.legend(('Model CDF', 'Data CDF'))
        if save_addr:
            plt.savefig(save_name,
                        bbox_inches='tight', pad_inches=0)
            plt.close()

    if save_addr:
        plt.ion()


""" Helper functions """


# Plot a rectangle
def rect(ax, x, y, w, h, c, **kwargs):
    # Varying only in x
    if len(c.shape) is 1:
        rect = plt.Rectangle((x, y), w, h, color=c, ec=c, **kwargs)
        ax.add_patch(rect)
    # Varying in x and y
    else:
        # Split into a number of bins
        N = c.shape[0]
        hb = h / float(N);
        yl = y
        for i in range(N):
            yl += hb
            rect = plt.Rectangle((x, yl), w, hb,
                                 color=c[i, :], ec=c[i, :], **kwargs)
            ax.add_patch(rect)


# Fill a contour between two lines
def rainbow_fill_between(ax, X, Y1, Y2,
                         colors=None, norm=None,
                         cmap=plt.get_cmap("RdBu_r"), **kwargs):
    plt.plot(X, Y1, lw=0)  # Plot so the axes scale correctly

    dx = X[1] - X[0]
    N = X.size

    # Pad a float or int to same size as x
    if (type(Y2) is float or type(Y2) is int):
        Y2 = np.array([Y2] * N)

    # No colors -- specify linear
    if norm is not None and colors is not None:
        cmap_norm = norm(colors)
        colors = cmap(cmap_norm)

    # if colors is None:
    #     colors = []
    #     for n in range(N):
    #         colors.append(cmap(n / float(N)))
    # # Varying only in x
    # elif len(colors.shape) is 1:
    #     colors = cmap((colors - colors.min())
    #                   / (colors.max() - colors.min()))
    # # Varying only in x and y
    # else:
    #     cnp = np.array(colors)
    #     colors = np.empty([colors.shape[0], colors.shape[1], 4])
    #     for i in range(colors.shape[0]):
    #         for j in range(colors.shape[1]):
    #             colors[i, j, :] = cmap((cnp[i, j] - cnp[:, :].min())
    #                                    / (cnp[:, :].max() - cnp[:, :].min()))

    colors = np.array(colors)

    # Create the patch objects
    for (color, x, y1, y2) in zip(colors, X, Y1, Y2):
        rect(ax, x, y2, dx, y1 - y2, color)

    return ax


def add_color_bar(color_data, norm, cmap=plt.get_cmap("RdBu_r"),
                  h_w_ratio=10, ytick_num=10, ax=None,
                  color_label=None,
                  orientation="vertical"):
    """Plot a color bar to axis according to specified color range."""
    if not ax:
        _, ax = plt.subplots()

    if not color_label:
        color_label = color_data

    N_color_data = color_data.size

    # produce color data
    color_data_norm = norm(color_data)
    colors = cmap(color_data_norm)

    # reshape so it is displayed horizontally/vertically
    if orientation == "vertical":
        colors = np.expand_dims(colors, axis=1)
        colors = np.repeat(colors, N_color_data // h_w_ratio, axis=1)
    else:
        colors = np.expand_dims(colors, axis=0)
        colors = np.repeat(colors, N_color_data // h_w_ratio, axis=0)

    # plot
    ax.imshow(colors, origin='lower')

    # adjust tick
    tick_id = np.arange(0, N_color_data + 1, step=N_color_data // ytick_num)
    tick_id[-1] = N_color_data - 1

    if orientation == "vertical":
        ax.yaxis.set_ticks(tick_id)
        ax.set_yticklabels(np.round(color_data[tick_id], 1))
        ax.set_xticklabels([])
    else:
        ax.xaxis.set_ticks(tick_id)
        ax.set_xticklabels(np.round(color_data[tick_id], 1))
        ax.set_yticklabels([])

    return ax


"""Default color norm"""

SIGNIFICANT_NORM = make_color_norm(
    [np.linspace(0, 0.05, 40),
     np.linspace(0.05, 0.95, 20),
     np.linspace(0.95, 1, 40)],
    method="percentile")

UNC_COLOR_PALETTE = {
    "para": "#ED553B",
    "str_system": "#20639B",
    "str_random": "#173F5F",
    "alea": "grey"
}
