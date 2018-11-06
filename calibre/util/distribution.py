"""Multivariate Normal distribution class for decoupled representation.

#### References

[1]:    Ching-An Cheng and Byron Boots. Variational Inference for Gaussian
        Process Models with Linear Complexity. _Advances in NIPS 30_, 2017.
        http://papers.nips.cc/paper/7103-variational-inference-for-gaussian-process-models-with-linear-complexity
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_probability.python.distributions import mvn_linear_operator
from tensorflow_probability.python.distributions import mvn_full_covariance


class VariationalGaussianProcessDecoupled(mvn_full_covariance.MultivariateNormalFullCovariance):
    """Variational Multivariate Normal under Decoupled Representation in [1]"""

    def __init__(self,
                 loc=None,
                 covariance_matrix=None,
                 func_norm_mm=None,
                 log_det_ss=None,
                 cond_norm_ss=None,
                 validate_args=False, allow_nan_stats=True,
                 name="VariationalGaussianProcessDecoupled"):
        """Construct Multivariate Normal distribution on `R^k`.

        The `batch_shape` is the broadcast shape between `loc` and
        `covariance_matrix` arguments.

        The `event_shape` is given by last dimension of the matrix implied by
        `covariance_matrix`. The last dimension of `loc` (if provided) must
        broadcast with this.

        A non-batch `covariance_matrix` matrix is a `k x k` symmetric positive
        definite matrix.  In other words it is (real) symmetric with all eigenvalues
        strictly positive.

        Additional leading dimensions (if any) will index batches.

        Args:
          loc: Floating-point `Tensor`. If this is set to `None`, `loc` is
            implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
            `b >= 0` and `k` is the event size.
          covariance_matrix: Floating-point, symmetric positive definite `Tensor` of
            same `dtype` as `loc`.  The strict upper triangle of `covariance_matrix`
            is ignored, so if `covariance_matrix` is not symmetric no error will be
            raised (unless `validate_args is True`).  `covariance_matrix` has shape
            `[B1, ..., Bb, k, k]` where `b >= 0` and `k` is the event size.
          func_norm_mm: L2 norm for variational mean function.
          log_det_ss: log determinant for covariance.
          cond_cov_inv: Inverse of conditional covariance.
          validate_args: Python `bool`, default `False`. When `True` distribution
            parameters are checked for validity despite possibly degrading runtime
            performance. When `False` invalid inputs may silently render incorrect
            outputs.
          allow_nan_stats: Python `bool`, default `True`. When `True`,
            statistics (e.g., mean, mode, variance) use the value "`NaN`" to
            indicate the result is undefined. When `False`, an exception is raised
            if one or more of the statistic's batch members are undefined.
          name: Python `str` name prefixed to Ops created by this class.

        Raises:
          ValueError: if neither `loc` nor `covariance_matrix` are specified.
        """
        parameters = dict(locals())

        super(VariationalGaussianProcessDecoupled, self).__init__(
            loc=loc,
            covariance_matrix=covariance_matrix,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name)

        self._parameters = parameters

        self.func_norm_mm = func_norm_mm
        self.log_det_ss = log_det_ss
        self.cond_norm_ss = cond_norm_ss


@tf.distributions.RegisterKL(mvn_linear_operator.MultivariateNormalLinearOperator,
                             VariationalGaussianProcessDecoupled)
def _kl_brute_force(a, b, name=None):
    """Batched KL divergence `KL(a || b)` for decoupled GP in [1].

    Args:
      a: Instance of `MultivariateNormalLinearOperator`.
      b: Instance of `VariationalGaussianProcessDecoupled`.
      name: (optional) name to use for created ops. Default "kl_mvn".

    Returns:
      scalar kl divergence
    """

    with tf.name_scope(
            name,
            "kl_dgp_decouple",
            values=[b.func_norm_mm, b.log_det_ss, b.cond_norm_ss]):
        kl_div = 0.5 * (b.func_norm_mm + b.log_det_ss - b.cond_norm_ss)
        return kl_div
