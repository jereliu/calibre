"""Utility functions for data generation and result visualization in experiments."""

import numpy as np

from functools import partial

import calibre.util.data as data_util

""" 1. Data generation """


def generate_data_1d(N_train=20, N_test=20, N_calib=500, N_valid=500,
                     noise_sd=0.03,
                     data_gen_func=data_util.sin_curve_1d,
                     data_gen_func_x=None, data_gen_func_x_test=None,
                     data_range=(0., 1.), valid_range=(-0.5, 1.5),
                     seed_train=1000, seed_test=1500, seed_calib=100,
                     valid_sample_size=1000):
    """
    Generates data for 1d experiment.

    Args:
        N_train: (int) size of training data
        N_test: (int) size of testing data (to be used by ensemble)
        N_valid: (int) size of validation data
        noise_sd: (int) noise level for train/test.
        data_gen_func: (function) data-generation function.
        data_gen_func_x: (function) data-generation function for feature.
        data_gen_func_x_test: (function) data-generation function for feature
            used for testing.
        data_range: (tuple of float32) range of x to sample from for train/test.
        valid_range: (tuple of float32) range of x to sample from for validation.
        seed_train: (int) random seed for generating training data.
        seed_test: (int) random seed for generating testing data.
        seed_calib: (int) random seed for generating calibration index.
        valid_sample_size: (int) sample to draw to calculate mean(y_valid).

    Returns:
        X_train, y_train, X_test, y_test, X_valid, y_valid. (np.ndarray of float32)
            training/testing/validation data of shape (N_**, 1)
        calib_sample_id (np.ndarray of int)
            index of X_valid used to compute calibration metrics.

    """
    # generate train and test data, then adjust shape
    X_train, y_train = data_util.generate_1d_data(N=N_train,
                                                  f=data_gen_func,
                                                  f_x=data_gen_func_x,
                                                  noise_sd=noise_sd,
                                                  seed=seed_train,
                                                  uniform_x_range=data_range)
    X_test, y_test = data_util.generate_1d_data(N=N_test,
                                                f=data_gen_func,
                                                f_x=data_gen_func_x_test,
                                                noise_sd=noise_sd,
                                                seed=seed_test,
                                                uniform_x_range=data_range)

    X_calib, y_calib = data_util.generate_1d_data(N=N_calib,
                                                  f=data_gen_func,
                                                  f_x=data_gen_func_x_test,
                                                  noise_sd=noise_sd,
                                                  seed=seed_calib,
                                                  uniform_x_range=data_range)

    X_train = np.expand_dims(X_train, 1).astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_test = np.expand_dims(X_test, 1).astype(np.float32)
    y_test = y_test.astype(np.float32)

    # generate validation data
    X_valid = np.linspace(valid_range[0], valid_range[1], N_valid)
    X_valid = np.expand_dims(X_valid, 1).astype(np.float32)

    y_valid_sample = data_gen_func(np.repeat(X_valid, valid_sample_size, axis=-1))

    # generate id for calibration data
    np.random.seed(seed_calib)
    calib_sample_id = np.argmin(np.abs(X_valid - X_calib[None, :]),
                                axis=0)

    # standardize data
    std_centr_X = np.mean(X_test)
    std_scale_X = np.std(X_test)
    std_centr_y = np.mean(y_test)
    std_scale_y = np.std(y_test) * 2

    X_train = (X_train - std_centr_X) / std_scale_X
    X_test = (X_test - std_centr_X) / std_scale_X
    X_valid = (X_valid - std_centr_X) / std_scale_X

    y_train = (y_train - std_centr_y) / std_scale_y
    y_test = (y_test - std_centr_y) / std_scale_y
    y_valid_sample = (y_valid_sample - std_centr_y) / std_scale_y

    return (X_train, y_train, X_test, y_test,
            X_valid, y_valid_sample, calib_sample_id)


def generate_data_1d_multiscale(
        N_train=20, N_test=20, N_valid=500, noise_sd=0.01,
        data_gen_func_list=None,
        data_range=(0., 1.), valid_range=(-0.5, 1.5),
        seed_train=1500, seed_test=2500, seed_calib=100):
    """
    Generates data for multiscale 1d experiment.

    Args:
        N_train: (int) size of training data
        N_test: (int) size of testing data (to be used by ensemble)
        N_valid: (int) size of validation data
        noise_sd: (int) noise level for train/test.
        data_gen_func_list: (list of function) data-generation functions.
        data_range: (tuple of float32) range of x to sample from for train/test.
        valid_range: (tuple of float32) range of x to sample from for validation.
        seed_train: (int) random seed for generating training data.
        seed_test: (int) random seed for generating testing data.
        seed_calib: (int) random seed for generating calibration index.

    Returns:
        X_train, y_train, X_test, y_test, X_valid, y_valid. (np.ndarray of float32)
            training/testing/validation data of shape (N_**, 1)
        calib_sample_id (np.ndarray of int)
            index of X_valid used to compute calibration metrics.

    """
    if data_gen_func_list is None:
        data_gen_func_list = [
            partial(data_util.sin_curve_1d,
                    freq=(3, 6), x_rate=0.1),
            partial(data_util.sin_curve_1d_fast_local,
                    bound=[0.1, 0.6], freq=50., scale=0.5)
        ]

    # generate train and test data, then adjust shape
    X_train, y_train = data_util.generate_1d_data_multiscale(
        N=N_train, f_list=data_gen_func_list,
        noise_sd=noise_sd, seed=seed_train,
        uniform_x=True,
        uniform_x_range=data_range)
    X_test, y_test = data_util.generate_1d_data_multiscale(
        N=N_test, f_list=data_gen_func_list,
        noise_sd=noise_sd, seed=seed_test,
        uniform_x=True,
        uniform_x_range=data_range)

    X_train = np.expand_dims(X_train, 1).astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_test = np.expand_dims(X_test, 1).astype(np.float32)
    y_test = y_test.astype(np.float32)

    # generate validation data
    X_valid = np.expand_dims(
        np.linspace(valid_range[0], valid_range[1], N_valid), 1).astype(np.float32)
    _, y_valid = data_util.generate_1d_data_multiscale(
        f_list=data_gen_func_list,
        noise_sd=0.00001, x=X_valid, )

    # generate id for calibration data
    np.random.seed(seed_calib)
    calib_sample_id = np.where((X_valid > data_range[0]) &
                               (X_valid <= data_range[1]))[0]
    calib_sample_id = np.random.choice(calib_sample_id,
                                       size=len(calib_sample_id),
                                       replace=False)

    return X_train, y_train, X_test, y_test, X_valid, y_valid, calib_sample_id


def generate_data_1d_multimodal(
        N_train=20, N_test=20, N_valid=500, noise_sd=0.01,
        data_gen_func_list=None,
        data_range=(0., 1.), valid_range=(-0.5, 1.5),
        seed_train=1500, seed_test=2500, seed_calib=100):
    """
    Generates data for multimodal 1d experiment.

    Args:
        N_train: (int) size of training data
        N_test: (int) size of testing data (to be used by ensemble)
        N_valid: (int) size of validation data
        noise_sd: (int) noise level for train/test.
        data_gen_func_list: (list of function) data-generation functions.
        data_range: (tuple of float32) range of x to sample from for train/test.
        valid_range: (tuple of float32) range of x to sample from for validation.
        seed_train: (int) random seed for generating training data.
        seed_test: (int) random seed for generating testing data.
        seed_calib: (int) random seed for generating calibration index.

    Returns:
        X_train, y_train, X_test, y_test, X_valid, y_valid. (np.ndarray of float32)
            training/testing/validation data of shape (N_**, 1)
        calib_sample_id (np.ndarray of int)
            index of X_valid used to compute calibration metrics.

    """
    if data_gen_func_list is None:
        data_gen_func_list = [data_util.sin_curve_1d, data_util.cos_curve_1d]

    # generate train and test data, then adjust shape
    X_train, y_train = data_util.generate_1d_data_multimodal(
        N=N_train, f_list=data_gen_func_list,
        noise_sd=noise_sd, seed=seed_train,
        uniform_x=True,
        uniform_x_range=data_range)
    X_test, y_test = data_util.generate_1d_data_multimodal(
        N=N_test, f_list=data_gen_func_list,
        noise_sd=noise_sd, seed=seed_test,
        uniform_x=True,
        uniform_x_range=data_range)

    X_train = np.expand_dims(X_train, 1).astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_test = np.expand_dims(X_test, 1).astype(np.float32)
    y_test = y_test.astype(np.float32)

    # generate validation data
    X_valid = np.concatenate((np.linspace(valid_range[0], valid_range[1], N_valid),
                              X_test.squeeze()))
    X_valid = np.expand_dims(np.sort(X_valid), 1).astype(np.float32)
    y_valid = np.concatenate([func(X_valid) for func in data_gen_func_list])
    X_valid = np.concatenate([X_valid] * len(data_gen_func_list))

    # generate id for calibration data
    np.random.seed(seed_calib)
    calib_sample_id = np.where((X_valid > data_range[0]) &
                               (X_valid <= data_range[1]))[0]
    calib_sample_id = np.random.choice(calib_sample_id,
                                       size=len(calib_sample_id),
                                       replace=False)

    return X_train, y_train, X_test, y_test, X_valid, y_valid, calib_sample_id


""" 2. Visualization """
# TODO(jereliu): collect all the visualization functions in one place.
