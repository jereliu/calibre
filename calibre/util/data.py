"""Utility functions to generate toy datasets"""
import numpy as np
import random
from mayavi import mlab

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def sin_curve_1d(x, freq=(4, 13), x_rate=1.):
    return x_rate * x + np.sin(freq[0] * x) + np.sin(freq[1] * x)


def sin_curve_1d_fast_local(x, bound=[0.1, 0.4], freq=50., scale=0.2):
    return np.sin(freq * x) * (bound[0] < x) * (x < bound[1]) * scale


def cos_curve_1d(x):
    return x + np.cos(4 * x) + np.cos(13 * x)


def simple_sin_curve_1d(x):
    return np.sin(x)


def simple_cos_curve_1d(x):
    return np.cos(x)


def eggholder(x, y):
    x = 400 * x
    y = 400 * y
    return (-(y + 47) * np.sin(np.sqrt(np.abs(y + x / 2 + 47))) -
            x * np.sin(np.sqrt(np.abs(x - (y + 47))))) / 1000.


def townsend(x, y):
    x = 2 * x
    y = 2 * y
    return (-(np.cos((x - 0.1) * y)) ** 2 - x * np.sin(3 * x + y)) / 2.


def goldstein(x, y):
    x = 2 * x
    y = 2 * y

    x = 4 * x - 2
    y = 4 * y - 2
    core = ((1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) *
            (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2)))
    return (np.log(core) - 8.693) / (2.427 * 4)


def bird(x, y):
    x = 5 * x
    y = 5 * y
    return (np.sin(x) * np.exp((1 - np.cos(y)) ** 2) +
            np.cos(y) * np.exp((1 - np.sin(x)) ** 2) + (x - y) ** 2) / 100.


FUNC_LIST_1D = [sin_curve_1d, cos_curve_1d]
FUNC_LIST_2D = [townsend, goldstein, bird, eggholder]


def train_test_split_id(n_data, train_perc=0.9):
    n_train = int(n_data * train_perc)
    data_id = np.random.choice(n_data, n_data, replace=False)
    train_id, test_id = data_id[:n_train], data_id[n_train:]

    return train_id, test_id


##

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


def generate_1d_data_multimodal(N, f_list=[simple_sin_curve_1d, simple_cos_curve_1d],
                                noise_sd=0.03, seed=None,
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

    y = []
    for partition_id, x_partition in enumerate(np.split(x, len(f_list))):
        eps = np.random.normal(loc=0, scale=noise_sd, size=len(x_partition))
        y.append(f_list[partition_id](x_partition + eps))
    y = np.concatenate(y)

    return x, y


def generate_1d_data_multiscale(N=100,
                                f_list=(sin_curve_1d,
                                        sin_curve_1d_fast_local),
                                noise_sd=0.03, x=None, seed=None,
                                uniform_x=False, uniform_x_range=(0., 1.)):
    """Generate 1D regression data in Louizos and Welling (2016)"""
    np.random.seed(seed)

    if x is None:
        if uniform_x:
            x = np.random.uniform(low=uniform_x_range[0],
                                  high=uniform_x_range[1], size=N)
        else:
            x = np.concatenate([
                np.random.uniform(low=0, high=0.6, size=int(N * 0.8)),
                np.random.uniform(low=0.8, high=1, size=N - int(N * 0.8))])

    x = np.sort(x.squeeze())
    y = []
    for func in f_list:
        eps = np.random.normal(loc=0, scale=noise_sd, size=len(x))
        y.append(func(x + eps))
    y = np.sum(np.asarray(y), 0)

    return x, y


def fractal_mountain(levels=11, visual=False, seed=None):
    """Generates fractal mountain.

    Adopted from
        https://github.com/dafarry/python-fractal-landscape/blob/master/fractal-mountain.py
    """
    random.seed(seed)

    size = 2 ** (levels - 1)
    height = np.zeros((size + 1, size + 1))

    for lev in range(levels):
        step = size // 2 ** lev
        for y in range(0, size + 1, step):
            jumpover = 1 - (y // step) % 2 if lev > 0 else 0
            for x in range(step * jumpover, size + 1, step * (1 + jumpover)):
                pointer = 1 - (x // step) % 2 + 2 * jumpover if lev > 0 else 3
                yref, xref = step * (1 - pointer // 2), step * (1 - pointer % 2)
                corner1 = height[y - yref, x - xref]
                corner2 = height[y + yref, x + xref]
                average = (corner1 + corner2) / 2.0
                variation = step * (random.random() - 0.5)
                height[y, x] = average + variation if lev > 0 else 0

    xg, yg = np.mgrid[-1:1:1j * size, -1:1:1j * size]

    if visual:
        surf = mlab.surf(xg, yg, height,
                         colormap='gist_earth', warp_scale='auto')
        mlab.show()

    data = np.stack([xg, yg, height[1:, 1:]], axis=-1)

    return data.reshape(-1, data.shape[-1])


def generate_2d_data(func, size=100, data_range=(-2., 2.),
                     validation=False, visualize=False, seed=100):
    """Generates 2d data according to function.

    Args:
        func: (function) function that takes (x, y) and return a scalar
        size: (int) size of training sample to generate
        data_range: (tuple of float32) range of x/y data.
        validation: (bool) whether to generate validation data instead
            of training data.
        visualize: (bool) whether to also visualize 2d surface, used
            only when validation=True
        seed: (int) random seed for generating training data.

    Returns:
        data: (np.ndarray) data containing values of x, y and f(x, y)
            dimension (size, 3).
    """

    lower_bound = data_range[0]
    upper_bound = data_range[1]
    total_range = data_range[1] - data_range[0]

    if validation:
        size = np.sqrt(size).astype(np.int)
        x_range = np.linspace(lower_bound, upper_bound, size)
        y_range = np.linspace(lower_bound, upper_bound, size)
        xg, yg = np.meshgrid(x_range, y_range)

        output = np.zeros((size, size))
        for x_id in enumerate(range(len(x_range))):
            for y_id in enumerate(range(len(y_range))):
                output[x_id, y_id] = func(xg[x_id, y_id], yg[x_id, y_id])

        if visualize:
            ax = plt.axes(projection='3d')
            ax.plot_surface(X=xg, Y=yg, Z=output, cmap='inferno')
    else:
        np.random.seed(seed)

        xg = np.random.sample(size) * total_range + lower_bound
        yg = np.random.sample(size) * total_range + lower_bound

        output = np.zeros(size)

        for data_id in range(size):
            output[data_id] = func(xg[data_id], yg[data_id])

    data = np.stack([xg, yg, output], axis=-1)

    return data.reshape(-1, data.shape[-1]).astype(np.float32)
