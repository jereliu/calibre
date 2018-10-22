"""Utility functions for Matrix Operations in Tensorflow/Numpy"""
import numpy as np
import tensorflow as tf


def pinv(A, reltol=1e-15):
    # Compute the SVD of the input matrix A
    s, u, v = tf.svd(A)

    # Invert s, clear entries lower than reltol*s[0].
    atol = tf.reduce_max(s) * reltol
    num_eig = tf.reduce_sum(tf.cast(s > atol, tf.int32))
    s_inv = tf.diag(1. / s[:num_eig])
    K_inv = tf.matmul(v[:, :num_eig],
                      tf.matmul(s_inv, u[:, :num_eig], transpose_b=True))
    return K_inv


def corr_mat(X, axis=0, N_max=5000):
    # for a 3D array, compute covariance along given axis
    if X.ndim != 3:
        raise ValueError("Input must be a 3D array.")

    X = np.swapaxes(X, axis, 0)
    axis_K, axis_N, axis_P = 0, 1, 2
    N = X.shape[axis_N]

    if N > N_max:
        # specify slice index
        slice_idx = [slice(None)] * len(X.shape)
        sample_idx = np.random.choice(N, N_max, replace=False)
        slice_idx[axis_N] = sample_idx

        # subsample
        X = X[slice_idx]
        N = N_max

    m1 = X - X.sum(axis_N, keepdims=True) / N
    m1_sd = np.std(X, axis=axis_N, keepdims=True)

    cov_mat = np.einsum('ijk,ijl->ikl', m1, m1) / (N - 1)
    sd_out = np.einsum('ijk,ijl->ikl', m1_sd, m1_sd)

    return cov_mat / sd_out


def replicate_along_zero_axis(A, n_replicate):
    return tf.ones([n_replicate] + [1] * (len(A.shape))) * A


def make_block_matrix(M_00, M_01, M_11=None, ridge_factor=0.):
    """Creates block matrix."""
    if isinstance(M_00, np.ndarray):
        if M_11 is not None:
            block_mat = np.block([[M_00, M_01],
                                  [M_01.T, M_11]])
            ridge_mat = ridge_factor * np.eye(block_mat.shape[0])
            return block_mat + ridge_mat
        else:
            return np.block([M_00, M_01])

    elif isinstance(M_00, tf.Tensor):
        if M_11 is not None:
            row_1 = tf.concat([M_00, M_01], axis=1)
            row_2 = tf.concat([tf.transpose(M_01), M_11], axis=1)

            concat_mat = tf.concat([row_1, row_2], axis=0)
            ridge_mat = ridge_factor * tf.eye(concat_mat.shape.as_list()[0])
            return concat_mat + ridge_mat
        else:
            return tf.concat([M_00, M_01], axis=1)