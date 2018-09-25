"""Utility functions to generate toy datasets"""
import numpy as np


def sin_curve_1d(x):
    return x + np.sin(4 * x) + np.sin(13 * x)


def generate_1d_data(N, f, noise_sd=0.03, seed=None, uniform_x=False):
    """Generate 1D regression data in Louizos and Welling (2016)"""
    np.random.seed(seed)

    if uniform_x:
        x = np.random.uniform(low=0, high=1, size=N)
    else:
        x = np.concatenate([
            np.random.uniform(low=0, high=0.6, size=int(N * 0.8)),
            np.random.uniform(low=0.8, high=1, size=N - int(N * 0.8))])
    eps = np.random.normal(loc=0, scale=noise_sd, size=N)
    y = f(x + eps)
    return x, y
