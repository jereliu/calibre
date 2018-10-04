"""Utility functions to generate toy datasets"""
import numpy as np


def sin_curve_1d(x):
    return x + np.sin(4 * x) + np.sin(13 * x)


def cos_curve_1d(x):
    return x + np.cos(4 * x) + np.cos(13 * x)


def generate_1d_data(N, f, noise_sd=0.03, seed=None,
                     uniform_x=False, uniform_x_range=[0., 1.]):
    """Generate 1D regression data in Louizos and Welling (2016)"""
    np.random.seed(seed)

    if uniform_x:
        x = np.random.uniform(low=uniform_x_range[0],
                              high=uniform_x_range[1], size=N)
    else:
        x = np.concatenate([
            np.random.uniform(low=0, high=0.6, size=int(N * 0.8)),
            np.random.uniform(low=0.8, high=1, size=N - int(N * 0.8))])
    eps = np.random.normal(loc=0, scale=noise_sd, size=N)

    y = f(x + eps)

    return x, y


def generate_1d_data_multimodal(N, f_list=[sin_curve_1d, cos_curve_1d],
                                noise_sd=0.03, seed=None, uniform_x=False):
    """Generate 1D regression data in Louizos and Welling (2016)"""
    np.random.seed(seed)

    if uniform_x:
        x = np.random.uniform(low=0, high=1, size=N)
    else:
        x = np.concatenate([
            np.random.uniform(low=0, high=0.6, size=int(N * 0.8)),
            np.random.uniform(low=0.8, high=1, size=N - int(N * 0.8))])

    y = []
    for partition_id, x_partition in enumerate(np.split(x, len(f_list))):
        eps = np.random.normal(loc=0, scale=noise_sd, size=len(x_partition))
        y.append(f_list[partition_id](x_partition + eps))
    y = np.concatenate(y)

    return x, y
