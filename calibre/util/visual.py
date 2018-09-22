"""Utility functions for visualization"""
import numpy as np

import matplotlib.pyplot as plt


def gpr_1d_visual(pred_mean_test, pred_cov_test,
                  X_train=None, y_train=None, X_test=None, y_test=None,
                  X_induce=None,
                  title="", save_addr=""):
    """Plots the GP posterior predictive mean and uncertainty.

    :param pred_mean_test: (np.ndarray) posterior predictive mean at X_test
    :param pred_cov_test: (np.ndarray) posterior predictive variance at X_test
    :param X_train: (np.ndarray) X values in training dataset.
    :param y_train: (np.ndarray) y values in training dataset.
    :param X_test: (np.ndarray) X values in test dataset.
    :param y_test: (np.ndarray) y values in test dataset.
    :param title: (str) Title of the image.
    :param save_addr: (str) Address to save image to.
    :return:
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

    if save_addr:
        plt.savefig(save_addr)
        plt.close()
        plt.ion()
