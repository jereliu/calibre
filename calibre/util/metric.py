"""Utility functions for calibration and evaluation metrics."""
import tensorflow as tf
import numpy as np


def ecdf_eval(Y_obs, Y_sample, axis=-1):
    """Computes empirical cdf calibration (i.e. P(Y<y) ) for M samples.

    Args:
        Y_obs: (np.ndarray or tf.Tensor) N observations of dim (N, 1), dtype float32
        Y_sample: (np.ndarray or tf.Tensor) Samples of size M corresponding
        to the N observations. dim (N, M), dtype float32
        axis: (int) axis average over.

    Returns:
        (np.ndarray of float32) empirical cdf evaluations for Y_obs
            based on Y_sample. dim (N,)
    """
    if isinstance(Y_obs, np.ndarray) and isinstance(Y_sample, np.ndarray):
        if Y_obs.ndim == 1:
            Y_obs = np.expand_dims(Y_obs, axis)

        return np.mean(Y_sample < Y_obs, axis)

    elif isinstance(Y_obs, tf.Tensor) and isinstance(Y_sample, tf.Tensor):
        return tf.reduce_mean(tf.cast(Y_sample < Y_obs, tf.float32), axis)
    else:
        raise ValueError("'Y_obs' and 'Y_sample' must both be np.ndarray or tf.Tensor")


def make_empirical_cdf_1d(sample, reduce_mean=True):
    """Creates a 1D empirical cdf.

    Args:
        sample: (np.ndarray or tf.Tensor) Observed samples of dimension (M, )
        reduce_mean: (bool) Whether to reduce empirical cdf evaluation to scalar
            by mean averaging.

    Returns:
        (function) The empirical cdf function based on Y_sample
    """
    if isinstance(sample, np.ndarray):
        def ecdf_func(val):
            if val == 0.:
                return 0.
            else:
                ecdf_val = (sample <= val)
                if reduce_mean:
                    ecdf_val = np.mean(ecdf_val)
                return ecdf_val

        return np.vectorize(ecdf_func)

    elif isinstance(sample, tf.Tensor):
        def ecdf_func(val):
            if val == 0.:
                return 0.
            else:
                ecdf_val = tf.cast(sample <= val, tf.float32)
                if reduce_mean:
                    ecdf_val = tf.reduce_mean(ecdf_val)
                return ecdf_val

        return ecdf_func
    else:
        raise ValueError("'sample' must both be either np.ndarray or tf.Tensor")


def monte_carlo_dual_expectation(f, samples_1, samples_2,
                                 log_prob=None, axis=None, name=None):
    """Computes the Monte-Carlo approximation of `E_p[f(X, X')]` for i.i.d. X, X'.

    This function computes the Monte-Carlo approximation of an expectation, i.e.,

    ```none
    E_p[f(X)] approx= m**-1 sum_i^m f(x_j),  x_j ~iid p(X)
    ```

    where:

    - `x_j = samples[j, ...]`,
    - `log(p(samples)) = log_prob(samples)` and
    - `m = prod(shape(samples)[axis])`.

    Tricks: Reparameterization and Score-Gradient

    When p is "reparameterized", i.e., a diffeomorphic transformation of a
    parameterless distribution (e.g.,
    `Normal(Y; m, s) <=> Y = sX + m, X ~ Normal(0,1)`), we can swap gradient and
    expectation, i.e.,
    `grad[ Avg{ s_i : i=1...n } ] = Avg{ grad[s_i] : i=1...n }` where
    `S_n = Avg{s_i}` and `s_i = f(x_i), x_i ~ p`.

    However, if p is not reparameterized, TensorFlow's gradient will be incorrect
    since the chain-rule stops at samples of non-reparameterized distributions.
    (The non-differentiated result, `approx_expectation`, is the same regardless
    of `use_reparametrization`.) In this circumstance using the Score-Gradient
    trick results in an unbiased gradient, i.e.,

    ```none
    grad[ E_p[f(X)] ]
    = grad[ int dx p(x) f(x) ]
    = int dx grad[ p(x) f(x) ]
    = int dx [ p'(x) f(x) + p(x) f'(x) ]
    = int dx p(x) [p'(x) / p(x) f(x) + f'(x) ]
    = int dx p(x) grad[ f(x) p(x) / stop_grad[p(x)] ]
    = E_p[ grad[ f(x) p(x) / stop_grad[p(x)] ] ]
    ```

    Unless p is not reparametrized, it is usually preferable to
    `use_reparametrization = True`.

    Warning: users are responsible for verifying `p` is a "reparameterized"
    distribution.

    Example Use:

    ```python
    # Monte-Carlo approximation of a reparameterized distribution, e.g., Normal.

    num_draws = int(1e5)
    p = tfp.distributions.Normal(loc=0., scale=1.)
    q = tfp.distributions.Normal(loc=1., scale=2.)
    exact_kl_normal_normal = tfp.distributions.kl_divergence(p, q)
    # ==> 0.44314718
    approx_kl_normal_normal = tfp.monte_carlo.expectation(
        f=lambda x: p.log_prob(x) - q.log_prob(x),
        samples=p.sample(num_draws, seed=42),
        log_prob=p.log_prob,
        use_reparametrization=(p.reparameterization_type
                               == tfp.distributions.FULLY_REPARAMETERIZED))
    # ==> 0.44632751
    # Relative Error: <1%

    # Monte-Carlo approximation of non-reparameterized distribution,
    # e.g., Bernoulli.

    num_draws = int(1e5)
    p = tfp.distributions.Bernoulli(probs=0.4)
    q = tfp.distributions.Bernoulli(probs=0.8)
    exact_kl_bernoulli_bernoulli = tfp.distributions.kl_divergence(p, q)
    # ==> 0.38190854
    approx_kl_bernoulli_bernoulli = tfp.monte_carlo.expectation(
        f=lambda x: p.log_prob(x) - q.log_prob(x),
        samples=p.sample(num_draws, seed=42),
        log_prob=p.log_prob,
        use_reparametrization=(p.reparameterization_type
                               == tfp.distributions.FULLY_REPARAMETERIZED))
    # ==> 0.38336259
    # Relative Error: <1%

    # For comparing the gradients, see `monte_carlo_test.py`.
    ```

    Note: The above example is for illustration only. To compute approximate
    KL-divergence, the following is preferred:

    ```python
    approx_kl_p_q = bf.monte_carlo_csiszar_f_divergence(
        f=bf.kl_reverse,
        p_log_prob=q.log_prob,
        q=p,
        num_draws=num_draws)
    ```

    Args:
      f: Python callable which can return `f(samples_1, samples_2)`.
      samples_1: `Tensor` of samples used to form the Monte-Carlo approximation of
        `E_p[f(X, X')]`.  A batch of samples should be indexed by `axis` dimensions.
      samples_2: `Tensor` of samples used to form the Monte-Carlo approximation of
        `E_p[f(X, X')]`.  A batch of samples should be indexed by `axis` dimensions.
      log_prob: Python callable which can return `log_prob(samples)`. Must
        correspond to the natural-logarithm of the pdf/pmf of each sample. Only
        required/used if `use_reparametrization=False`.
        Default value: `None`.
      axis: The dimensions to average. If `None`, averages all
        dimensions.
        Default value: `0` (the left-most dimension).
      keep_dims: If True, retains averaged dimensions using size `1`.
        Default value: `False`.
      name: A `name_scope` for operations created by this function.
        Default value: `None` (which implies "expectation").

    Returns:
      approx_expectation: `Tensor` corresponding to the Monte-Carlo approximation
        of `E_p[f(X)]`.

    Raises:
      ValueError: if `f` is not a Python `callable`.
      ValueError: if `use_reparametrization=False` and `log_prob` is not a Python
        `callable`.
    """

    with tf.name_scope(name, 'expectation', [samples_1, samples_2]):
        if not callable(f):
            raise ValueError('`f` must be a callable function.')
        else:
            if not callable(log_prob):
                raise ValueError('`log_prob` must be a callable function.')
            stop = tf.stop_gradient  # For readability.
            x_1 = stop(samples_1)
            x_2 = stop(samples_2)

            logpx_1 = log_prob(x_1)
            logpx_2 = log_prob(x_2)

            # Call `f` once in case it has side-effects.
            fx = f(x_1, x_2)  # shape (n_sample_1, n_sample_2)

            # To achieve this, we use the fact that:
            #   `h(x) - stop(h(x)) == zeros_like(h(x))`
            # but its gradient is grad[h(x)].
            #
            # This technique was published as:
            # Jakob Foerster, Greg Farquhar, Maruan Al-Shedivat, Tim Rocktaeschel,
            # Eric P. Xing, Shimon Whiteson (ICML 2018)
            # "DiCE: The Infinitely Differentiable Monte-Carlo Estimator"
            # https://arxiv.org/abs/1802.05098
            #
            # Unlike using:
            #   fx = fx + stop(fx) * (logpx - stop(logpx)),
            # DiCE ensures that any order gradients of the objective
            # are unbiased gradient estimators.
            #
            # Note that IEEE754 specifies that `x - x == 0.` and `x + 0. == x`, hence
            # this trick loses no precision. For more discussion regarding the
            # relevant portions of the IEEE754 standard, see the StackOverflow
            # question,
            # "Is there a floating point value of x, for which x-x == 0 is false?"
            # http://stackoverflow.com/q/2686644
            dice_exp_1 = tf.expand_dims(tf.exp(logpx_1 - stop(logpx_1)), 1)
            dice_exp_2 = tf.expand_dims(tf.exp(logpx_2 - stop(logpx_2)), 0)

            dice = fx * tf.multiply(dice_exp_1, dice_exp_2)

            return tf.reduce_mean(dice, axis=axis)


"""Standard evaluation metrics"""


def rmse(y_obs, y_pred):
    """Computes root mean square error."""
    return np.sqrt(np.mean((y_obs.squeeze() - y_pred.squeeze()) ** 2))


def rsqure(y_obs, y_pred):
    """Computes Standardized R-square."""
    nom = np.mean((y_obs.squeeze() - y_pred.squeeze()) ** 2)
    denom = np.mean((y_obs.squeeze() - np.mean(y_obs)) ** 2)
    return 1 - (nom / denom)


def boot_sample(y_obs, y_pred, n_boot=1000, metric_func=rmse, seed=100):
    """Computes bootstrap sample for given metric function.

    Args:
        y_obs: (np.ndarray) observation, shape (N_obs, )
        y_pred: (np.ndarray) prediction, shape (N_obs, )
        n_boot: (int) sample size
        metric_func: (function) function that takes
            y_obs, y_pred and return a scalar value.
        seed: (int) random seed for bootstrap sampling.

    Returns:
        boot_mean, boot_sd (np.ndarray)
            mean and standard dev of the boot sample.
        boot_sample (np.ndarray) bootstrap samples, with size (n_boot, )
    """
    # sample
    np.random.seed(seed)
    N_obs = y_obs.size

    boot_sample = []
    for _ in range(n_boot):
        boot_id = np.random.choice(range(N_obs), N_obs, replace=True)
        boot_sample.append(metric_func(y_obs[boot_id], y_pred[boot_id]))

    boot_sample = np.asarray(boot_sample)

    return np.mean(boot_sample), np.std(boot_sample), boot_sample
