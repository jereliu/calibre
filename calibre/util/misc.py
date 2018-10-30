"""Misc helper functions."""
import numpy as np


def find_nearest(array, value):
    """Return array index for elements closest to those in value.

    Args:
        array: (np.ndarray) numpy array of shape (N_arr, D)
        value: (np.ndarray) list numpy array of shape (N_val, D)

    Returns:
        (np.ndarray of int) N_val indexes corresponding to elements in array
    """
    array = np.asarray(array)
    if len(array.shape) == 1:
        array = array[:, np.newaxis]

    idx_list = []
    for val_idx in range(value.shape[0]):
        idx_list.append(
            np.sum(np.abs(array - value[val_idx, :]), axis=1).argmin())
    return np.asarray(idx_list)
