"""Functions to compute the coverage index.

Coverage index refers to the coverage probability of the posterior predictive
credible interval. i.e. a well-calibrated posterior predictive distribution
should output 95% credible intervals that marginally covers 95% of the
observation in training/testing data set.

"""

import numpy as np


def credible_interval_coverage(Y_obs, Y_sample, n_perc_eval=100):
    """Computes the coverage of posterior predictive credible intervals.

    Args:
        Y_sample: (np.ndarray) Samples from posterior distribution,
            shape (N_obs, N_sample)
        Y_obs: (np.ndarray) Observations corresponds to Y_sample predictions,
            shape (N_obs, 1)
        n_perc_eval: (int) Number of credible interval coverage evaluations.
            with credible percentiles being np.linspace(0, 1, num=n_perc_eval)

    Returns:
        (np.ndarray) An ndarray of nominal coverage and observed coverage.
            Shapes are both (n_perc_eval, 1)
    """
    if Y_obs.ndim == 1:
        Y_obs = Y_obs.reshape([Y_obs.size, 1])

    nom_coverage = np.linspace(0, 1, n_perc_eval)
    perc_lower = ((1 - nom_coverage) / 2) * 100
    perc_upper = 100 - perc_lower

    # compute percentiles
    interval_upper = np.percentile(Y_sample, q=perc_upper, axis=1).T
    interval_lower = np.percentile(Y_sample, q=perc_lower, axis=1).T

    # evaluate whether lower < Y_obs < upper
    is_leq_upper = (interval_upper - Y_obs > 0)
    is_geq_lower = (Y_obs - interval_lower > 0)
    is_in_interval = is_leq_upper * is_geq_lower

    #
    observed_coverage = np.mean(is_in_interval, axis=0)

    return nom_coverage, observed_coverage
