"""Utility functions for Calibration.

For a predictive CDF F_pred(y|X) to be calibrated,
build a nonparametric calibration function C: F_pred -> F_calib such that

P(Y_obs < t) = C[ F_pred( Y < t | X_obs ) ], where the P is empirical CDF.

Model C using a flexible model C_\theta such that

I(Y_obs < t) = C_\theta(t, F_pred(Y<t), X_obs )

"""
import tqdm

import numpy as np
import tensorflow as tf

from sklearn.model_selection import ShuffleSplit


def build_training_dataset(y_pred_sample, y_obs, X_obs, num_cdf_eval=100):
    """Building training dataset for nonparametric calibration.

    Specifically, assume N observations and K CDF evaluation point, learn C
        by building a classification dataset with size N*K as below:

            for k in 1:K, for i in 1:N:
                label:      I(Y_i < t_k)
                feature_1:  t_k
                feature_2:  F_pred( t_k | X_i)
                feature_3:  X_i

    Both features and labels are are organized in batches of shape
        [n_cdf_eval, n_obs]

    Args:
        y_pred_sample: (np.ndarray) Samples from posterior predictive for each
            observed y, with dimension (n_obs, n_posterior_sample).
        y_obs: (np.ndarray) Observed y, with dimension (n_obs, 1).
        X_obs: (np.ndarray) Observed X corresponding to y_obs,
            with dimension (n_obs, n_feature)
        num_cdf_eval: (int) Number of CDF evaluation for each y_obs.

    Returns:
        (dict of np.ndarray): Dictionary of np.ndarrays of labels and
            features. It contains below key-value pair:
                - "label": shape (n_cdf_eval, n_obs, 1)
                - "feature_t": shape (n_cdf_eval, n_obs, 1)
                - "feature_cdf": shape (n_cdf_eval, n_obs, 1)
                - "feature_x":  shape (n_cdf_eval, n_obs, n_feature)

    Raises:
        (ValueError): If dimension of y_pred_sample different from len(y_obs)
        (ValueError): Shape of np.ndarray in dataset does not conform with
            expected batch shape.
        (ValueError): ndim of np.ndarray in dataset is not 2 or 3.
    """
    n_obs, n_sample = y_pred_sample.shape

    if n_obs != len(y_obs):
        raise ValueError(
            "First dimension of y_pred_sample must be same as len(y_obs). "
            "Expected: {}, Observed: {}".format(len(y_obs), n_obs, ))

    t_vals = np.linspace(np.min(y_obs), np.max(y_obs), num_cdf_eval)

    # create and fill data dictionary
    data_dict = dict()

    data_dict["label"] = np.asarray(
        [y_obs < t_val for t_val in t_vals])  # (n_obs, n_cdf_eval)

    data_dict["feature_t"] = np.repeat(np.expand_dims(t_vals, -1),
                                       repeats=n_obs, axis=-1)
    data_dict["feature_cdf"] = np.asarray(
        [np.mean(y_pred_sample < t_val, -1) for t_val in t_vals]
    )
    data_dict["feature_x"] = np.repeat(np.expand_dims(X_obs, 0),
                                       repeats=num_cdf_eval, axis=0)

    # check dimensions
    for key, value in data_dict.items():
        if value.shape[:2] != (num_cdf_eval, n_obs):
            raise ValueError(
                "Shape of '{}' does not conform with expected batch shape.\n"
                "Observed: ({}, {}), Expected: ({}, {})".format(
                    key, value.shape[0], value.shape[1],
                    num_cdf_eval, n_obs,
                )
            )

        if value.ndim != 3:
            if value.ndim == 2:
                data_dict[key] = np.expand_dims(value, axis=-1)
            else:
                raise ValueError(
                    "ndim of '{}' is expected to be either 2 or 3, "
                    "observed {}".format(key, value.ndim))

    return data_dict


def build_input_pipeline(train_data_dict, test_data_dict,
                         train_batch_size=1000,
                         test_batch_size=100, seed=100):
    """Build an Iterator switching between train and heldout data."""

    # extract label and feature, organize into np.ndarrays
    label_train = train_data_dict["label"]
    label_test = test_data_dict["label"]

    feature_train = np.concatenate(
        [train_data_dict[key] for key in train_data_dict.keys()
         if "feature" in key], axis=-1)
    feature_test = np.concatenate(
        [test_data_dict[key] for key in test_data_dict.keys()
         if "feature" in key], axis=-1)

    # organize into np.ndarrays
    n_train_data = label_train.size
    n_test_data = label_test.size

    label_train = label_train.reshape(n_train_data, 1).astype(np.int32)
    label_test = label_test.reshape(n_test_data, 1).astype(np.int32)
    feature_train = feature_train.reshape(n_train_data,
                                          feature_train.shape[-1]).astype(np.float32)
    feature_test = feature_test.reshape(n_test_data,
                                        feature_test.shape[-1]).astype(np.float32)

    # Build an iterator over training batches.
    training_dataset = tf.data.Dataset.from_tensor_slices(
        (feature_train, label_train))
    training_batches = training_dataset.shuffle(
        50000, reshuffle_each_iteration=True).repeat().batch(
        train_batch_size)
    training_iterator = training_batches.make_one_shot_iterator()

    # Build a iterator over the heldout set with batch_size=heldout_size,
    # i.e., return the entire heldout set as a constant.
    heldout_dataset = tf.data.Dataset.from_tensor_slices(
        (feature_test, label_test))
    heldout_frozen = (heldout_dataset.take(n_test_data).repeat().batch(
        test_batch_size))
    heldout_iterator = heldout_frozen.make_one_shot_iterator()

    # Combine these into a feedable iterator that can switch between
    # training and validation inputs.
    handle = tf.placeholder(tf.string, shape=[])
    feedable_iterator = tf.data.Iterator.from_string_handle(
        handle, training_batches.output_types, training_batches.output_shapes)
    features, labels = feedable_iterator.get_next()

    return (features, labels, handle, training_iterator, heldout_iterator,
            n_train_data, n_test_data)


# def build_local_calibration_dataset(X_obs, Y_obs, Y_sample, n_eval = 5):
#     """Building training dataset for localized calibration.
#
#     Specifically, assume N observations, learn a calibration function
#         C( F(y_obs|x), x ): F x X -> [0, 1]
#
#         by running monotonic regression on below dataset:
#
#             feature 1:      F_ij = F_i(y_j)
#             feature 2:      x_i
#             label:          P_{ij} = I(y_i < y_j)
#
#         where y_i,x_i are elements in Y_obs and X_obs,
#         and F_i(y_j) = F(y<y_j|x_i) is the model cdf evaluated at x_i.
#
#     Args:
#         X_obs: (tf.Tensor) Observed x, with dimension (n_obs, p).
#         Y_obs: (tf.Tensor) Observed y, with dimension (n_obs, ).
#         Y_sample: (tf.Tensor) Samples from posterior predictive for each
#             observed y, with dimension (n_obs, n_posterior_sample).
#         n_eval: (int) Number of y_j's to evaluate F_i's at.
#
#     Returns:
#         (ValueError): If sample size indicated in Y_sample different
#             from that of Y_obs.
#     """
#     X_obs = tf.convert_to_tensor(X_obs, dtype=tf.float32)
#     Y_obs = tf.convert_to_tensor(Y_obs.squeeze(), dtype=tf.float32)
#     Y_sample = tf.convert_to_tensor(Y_sample, dtype=tf.float32)
#
#     # check model dimension
#     n_obs, = Y_obs.shape.as_list()
#     n_obs_1, n_sample = Y_sample.shape.as_list()
#
#     if n_obs != n_obs_1:
#         raise ValueError(
#             "First dimension of y_pred_sample must be same as len(y_obs). "
#             "Expected: {}, Observed: {}".format(n_obs, n_obs_1, ))
#
#     # selects evaluation points
#     y_eval =
#
#     # prepare features
#     # prepare feature 1: model cdf
#
#
#     # prepare feature 2
#     # compute empirical cdf evaluations
#     F_obs = tf.reduce_mean(
#         tf.cast(Y_sample < tf.expand_dims(Y_obs, -1),
#                 dtype=tf.float32), axis=-1)
#
#     P_obs = tf.reduce_mean(
#         tf.cast(tf.expand_dims(F_obs, -1) <
#                 tf.expand_dims(F_obs, 0), dtype=tf.float32), axis=0)
#
#     return {"feature": F_obs, "label": P_obs}


def build_calibration_dataset(Y_obs, Y_sample):
    """Building training dataset for nonparametric calibration.

    Specifically, assume N observations, learn a calibration function
        P(Y<F(y_obs|x)): F -> [0, 1]

        by running monotonic regression on below dataset:

            feature:      F_obs = F(y < y_obs | x_obs)
            label:        P(F < F_obs)

    where P(F < F_obs) is the empirical cdf built from all F_obs'

    Args:
        Y_obs: (tf.Tensor) Observed y, with dimension (n_obs, ).
        Y_sample: (tf.Tensor) Samples from posterior predictive for each
            observed y, with dimension (n_obs, n_posterior_sample).

    Returns:
        (dict of tf.Tensor): Dictionary of tf.Tensor of labels and
            features. It contains below key-value pair:
                - "label": shape (n_obs, )
                - "feature": shape (n_obs, )

    Raises:
        (ValueError): If sample size indicated in Y_sample different
            from that of Y_obs.
    """
    Y_obs = tf.convert_to_tensor(Y_obs.squeeze(), dtype=tf.float32)
    Y_sample = tf.convert_to_tensor(Y_sample, dtype=tf.float32)

    # check model dimension
    n_obs, = Y_obs.shape.as_list()
    n_obs_1, n_sample = Y_sample.shape.as_list()

    if n_obs != n_obs_1:
        raise ValueError(
            "First dimension of y_pred_sample must be same as len(y_obs). "
            "Expected: {}, Observed: {}".format(n_obs, n_obs_1, ))

    # compute empirical cdf evaluations
    F_obs = tf.reduce_mean(
        tf.cast(Y_sample < tf.expand_dims(Y_obs, -1),
                dtype=tf.float32), axis=-1)

    P_obs = tf.reduce_mean(
        tf.cast(tf.expand_dims(F_obs, -1) <
                tf.expand_dims(F_obs, 0), dtype=tf.float32), axis=0)

    return {"feature": F_obs, "label": P_obs}


def sample_ecdf(n_sample, base_sample, quantile, y_range=None, seed=None):
    """Sample observations form 1D empirical cdf using inverse CDF method.

    Here empirical cdf is defined by base_sample and the
        corresponding quantiles.

    Args:
        n_sample: (int) Number of samples.
        base_sample: (np.ndarray of float32) Base samples to sample
            from, shape (n_sample0, )
        quantile: (np.ndarray of float32) Quantiles corresponding to
            the base samples.
        y_range: (tuple) (upper, lower) limit of the data.

    Returns:
        (np.ndarray of float32) Sample of shape (n_sample,) corresponding
            to the empirical cdf.
    """
    quantile = quantile.squeeze()
    # for i in range(1, len(quantile)):
    #     quantile[i] = np.max([quantile[i], quantile[i-1]])

    base_sample = np.sort(base_sample.squeeze())

    # adjust sample if quantile doens't cover full range
    min_quantile, max_quantile = quantile[0], quantile[-1]

    if y_range:
        if max_quantile < 1.:
            additional_sample_size = int(
                ((1 - max_quantile) / (max_quantile - min_quantile)) * len(base_sample))
            sample_limit_lower = np.max(base_sample)
            sample_limit_higher = y_range[1]

            additional_sample = np.random.uniform(low=sample_limit_lower,
                                                  high=sample_limit_higher,
                                                  size=additional_sample_size)
            base_sample = np.concatenate([base_sample, additional_sample])

        if min_quantile > 0.:
            additional_sample_size = int(
                (min_quantile / (1 - min_quantile)) * len(base_sample))
            sample_limit_lower = y_range[0]
            sample_limit_higher = np.min(base_sample)
            additional_sample = np.random.uniform(low=sample_limit_lower,
                                                  high=sample_limit_higher,
                                                  size=additional_sample_size)
            base_sample = np.concatenate([base_sample, additional_sample])

    if len(base_sample) > len(quantile):
        base_sample = base_sample[np.random.choice(len(base_sample),
                                                   len(quantile),
                                                   replace=False)]
    elif len(base_sample) < len(quantile):
        quantile = quantile[np.random.choice(len(quantile),
                                             len(base_sample),
                                             replace=False)]
        quantile = np.sort(quantile.squeeze())

    quantile = np.sort(quantile.squeeze())
    base_sample = np.sort(base_sample.squeeze())

    # identify sample id using inverse CDF lookup
    np.random.seed(seed)
    sample_prob = np.random.sample(size=n_sample)
    sample_id = np.sum(np.expand_dims(sample_prob, 0) >
                       np.expand_dims(quantile, 1), axis=0) - 1
    return base_sample[sample_id]


def resample_ecdf_batch(n_sample, base_sample_batch, quantile_batch,
                        y_range=None, seed=None, verbose=False):
    """Sample observations form 1D empirical cdf using inverse CDF method.

    Args:
        n_sample: (int) Number of samples.
        base_sample_batch: (np.ndarray of float32) Base samples to sample
            from, shape (n_batch, n_original_sample, )
        quantile_batch: (np.ndarray of float32) Quantiles corresponding to
            the base samples, shape (n_batch, n_quantiles, )
        y_range: (tuple) (upper, lower) limit of the data
        verbose: (bool) If True then print progress.

    Returns:
        (np.ndarray of float32) Sample of shape (n_batch, n_sample,)
            corresponding to the empirical cdf.

    Raises:
        (ValueError) Batch size between base_sample_batch and
            quantile_batch disagree.
    """
    n_batch0, _ = base_sample_batch.shape
    n_batch, _ = quantile_batch.shape

    if n_batch != n_batch0:
        raise ValueError(
            "Batch sizes for base samples ({}) and "
            "for quantiles ({}) disagree".format(n_batch0, n_batch))

    # constrain quantile values to be within [0., 1.]
    quantile_batch[quantile_batch > 1.] = 1.
    quantile_batch[quantile_batch < 0.] = 0.

    # process by batch
    calibrated_sample_batch = []
    batch_range = tqdm.tqdm(range(n_batch)) if verbose else range(n_batch)

    for batch_id in batch_range:
        base_sample = base_sample_batch[batch_id]
        quantile = quantile_batch[batch_id]

        calibrated_sample = sample_ecdf(n_sample=n_sample,
                                        base_sample=base_sample,
                                        quantile=quantile,
                                        y_range=y_range,
                                        seed=seed)

        calibrated_sample_batch.append(calibrated_sample)

    return np.asarray(calibrated_sample_batch)
