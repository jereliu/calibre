import time
import numpy as np

from importlib import reload

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability.python.edward2 as ed

from sklearn.model_selection import train_test_split
import seaborn as sns

import calibre.util.data as data_util
from calibre.util.data import simple_sin_curve_1d, simple_cos_curve_1d, generate_1d_data

import matplotlib.pyplot as plt

tfd = tfp.distributions

_DATASET = "sigmoid"

if _DATASET == "sigmoid":
    def build_toy_dataset(N):
        y_data = np.random.uniform(-10.5, 10.5, N).astype(np.float32)
        r_data = np.random.normal(size=N)  # random noise
        x_data = np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + r_data * 1.0
        x_data = x_data.reshape((N, 1)).astype(np.float32)
        return train_test_split(x_data, y_data, random_state=42)


    N = 5000  # number of data points
    D = 1  # number of features
    K = 20

    X_train, X_test, y_train, y_test = build_toy_dataset(N)
    print("Size of features in training data: {}".format(X_train.shape))
    print("Size of output in training data: {}".format(y_train.shape))
    print("Size of features in test data: {}".format(X_test.shape))
    print("Size of output in test data: {}".format(y_test.shape))

    sns.regplot(X_train, y_train, fit_reg=False)
else:
    N_train = 1000
    N_test = 1000
    N_valid = 500

    X_train, y_train = data_util.generate_1d_data_multimodal(
        N=N_train, f_list=[simple_sin_curve_1d, simple_cos_curve_1d],
        noise_sd=0.03, seed=1000,
        uniform_x=True, uniform_x_range=[-0.5, 8])
    X_test, y_test = data_util.generate_1d_data_multimodal(
        N=N_test, f_list=[simple_sin_curve_1d, simple_cos_curve_1d],
        noise_sd=0.03, seed=2000,
        uniform_x=True, uniform_x_range=[-0.5, 8])

    X_train = np.expand_dims(X_train, 1).astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_test = np.expand_dims(X_test, 1).astype(np.float32)
    y_test = y_test.astype(np.float32)

    std_y_train = np.std(y_train)

    X_valid = np.concatenate((np.linspace(-2.5, 10, N_valid),
                              X_test.squeeze()))
    X_valid = np.expand_dims(np.sort(X_valid), 1).astype(np.float32)
    y_valid = np.concatenate([simple_sin_curve_1d(X_valid),
                              simple_cos_curve_1d(X_valid)])
    X_valid = np.concatenate([X_valid, X_valid])

    N, D = X_train.shape

    #
    plt.plot(np.linspace(-0.5, 6.5, 100),
             simple_sin_curve_1d(np.linspace(-0.5, 6.5, 100)), c='black')
    plt.plot(np.linspace(-0.5, 6.5, 100),
             simple_cos_curve_1d(np.linspace(-0.5, 6.5, 100)), c='black')
    plt.plot(X_train.squeeze(), y_train.squeeze(),
             'o', c='red', markeredgecolor='black')
    plt.plot(X_test.squeeze(), y_test.squeeze(),
             'o', c='blue', markeredgecolor='black')
    plt.close()


#


def mixture_density_net(X, K=20, units=20):
    """Implements Mixture density network."""
    latent_feature = tf.keras.Sequential([
        tfp.layers.DenseFlipout(units=units, activation=tf.nn.tanh),
        tfp.layers.DenseFlipout(units=units, activation=tf.nn.tanh),
    ])
    latent_to_loc = tf.keras.Sequential([
        tfp.layers.DenseFlipout(units=K, activation=None)])
    latent_to_scale = tf.keras.Sequential([
        tfp.layers.DenseFlipout(units=K, activation=tf.exp)])
    latent_to_logit = tf.keras.Sequential([
        tfp.layers.DenseFlipout(units=K, activation=None)])

    # build network
    latent = latent_feature(X)
    locs = latent_to_loc(latent)
    scales = latent_to_scale(latent)
    logits = latent_to_logit(latent)
    probs = tf.nn.softmax(logits, axis=1)

    network_loss = sum(latent_feature.losses + latent_to_loc.losses +
                       latent_to_scale.losses + latent_to_logit.losses)

    # specify output
    mix_prob = tfd.Categorical(logits=logits)
    component = [tfd.Normal(loc=loc, scale=scale) for loc, scale in
                 zip(tf.unstack(tf.transpose(locs)),
                     tf.unstack(tf.transpose(scales)))]
    y = tfd.Mixture(cat=mix_prob, components=component)

    return y, locs, scales, probs, network_loss


def mixture_density_net_sample(x, pred_weights, pred_means, pred_std, amount):
    """Draws samples from mixture model.

    Returns 2 d array with input X and sample from prediction of mixture model.
    """
    samples = np.zeros((amount, 2))
    n_mix = len(pred_weights[0])
    to_choose_from = np.arange(n_mix)
    for j, (weights, means, std_devs) in enumerate(
            zip(pred_weights, pred_means, pred_std)):
        index = np.random.choice(to_choose_from, p=weights)
        samples[j, 1] = np.random.normal(means[index], std_devs[index], size=1)
        samples[j, 0] = x[j]
        if j == amount - 1:
            break
    return samples


#

learning_rate = 0.01
max_steps = 20000  # number of training iterations
n_sample = 100

vi_graph = tf.Graph()
with vi_graph.as_default():
    X = tf.placeholder(tf.float32, [None, D])

    y, locs, scales, probs, network_loss = mixture_density_net(
        X, K=20, units=15)

    neg_log_likelihood = -tf.reduce_mean(y.log_prob(y_train))
    kl = network_loss / len(y_train)
    elbo_loss = neg_log_likelihood + kl

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(elbo_loss)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    vi_graph.finalize()

with tf.Session(graph=vi_graph) as sess:
    start_time = time.time()

    sess.run(init_op)
    for step in range(max_steps):
        start_time = time.time()
        _, elbo_value = sess.run([train_op, elbo_loss], feed_dict={X: X_train})
        if step % 1000 == 0:
            duration = time.time() - start_time
            print("Step: {:>3d} Loss: {:.3f} ({:.3f} sec)".format(
                step, elbo_value, duration))

    sample_list = []
    for _ in range(n_sample):
        sample_list.append(
            sess.run([locs, scales, probs], feed_dict={X: X_valid}))

    sess.close()

mixture_sample_list = []
for sample in sample_list[:100]:
    locs_pred, scales_pred, probs_pred = sample
    mixture_sample = mixture_density_net_sample(
        X_valid, probs_pred, locs_pred, scales_pred, amount=len(X_valid))
    mixture_sample_list.append(mixture_sample)

mixture_sample_arr = np.asarray(mixture_sample_list).squeeze()
mixture_sample_arr = mixture_sample_arr.reshape(-1, mixture_sample_arr.shape[-1])

sns.regplot(mixture_sample[:, 0], mixture_sample[:, 1],
            fit_reg=False, scatter_kws={'alpha': 0.1})
plt.plot(np.linspace(-0.5, 8, 100),
         simple_sin_curve_1d(np.linspace(-0.5, 8, 100)), c='black')
plt.plot(np.linspace(-0.5, 8, 100),
         simple_cos_curve_1d(np.linspace(-0.5, 8, 100)), c='black')
plt.plot(X_train.squeeze(), y_train.squeeze(),
         'o', c='red', markeredgecolor='black')
