"""Functions Performing Post-hoc Nonparametric Calibration.

For a predictive CDF F_pred(y|X) to be calibrated,
build a nonparametric calibration function C: F_pred -> F_calib such that

P(Y_obs < t) = C[ F_pred( Y<t |X_obs) ], where the P is empirical CDF.

Specifically, assume N observations and K CDF evaluation point, learn C
    by building a classification dataset with size N*K as below:

        for k in 1:K, for i in 1:N:
            label:      I(Y_i < t_k)
            feature_1:  t_k
            feature_2:  F_pred( t_k | X_i)
            feature_3:  X_i

Ideally, C should be
    (1) monotonic wrt t_k
    (2) A flexible, but usually simple transformation of F_pred
    (3) Impose smoothness in prediction between neighboring X_i's.

#### References

[1]:    Han Song, Meelis Kull, Peter Flach. Non-parametric Calibration
        of Probablistic Regression. _arXiv:1806.07690_. 2018
        https://arxiv.org/abs/1806.07690
"""
# TODO(jereliu): Extend to proper posterio.
import tensorflow as tf


#def kernel_score(g=tf.abs):
