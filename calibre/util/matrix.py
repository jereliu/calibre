"""Utility functions for Matrix Operations in Tensorflow"""
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