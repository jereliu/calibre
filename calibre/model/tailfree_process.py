"""Model definition and variational family for Dependent Tail-free Process Prior.

#### References

[1]: Alejandro Jara and Timothy Hanson. A class of mixtures of dependent tail-free
        processes. _Biometrika. 2011;98(3):553-566._. 2011

[2]: Eswar G. Phadia. Tailfree Processes. In: Prior Processes and Their Applications.
        _Springer Series in Statistics. Springer, Cham_. 2016.

[3]: Subhashis Ghoshal. A Invitation to Bayesian Nonparametrics.
        _Presentation Slides_, 2011.
        https://www.eurandom.tue.nl/EURANDOM_chair/minicourseghoshal.pdf
"""
import functools

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed

from calibre.util import inference as inference_util

from calibre.model import gaussian_process as gp

from calibre.util.model import sparse_softmax

tfd = tfp.distributions

_TEMP_PRIOR_MEAN = -4.
_TEMP_PRIOR_SDEV = 1.

ROOT_NODE_DEFAULT_NAME = "root"
BASE_WEIGHT_NAME_PREFIX = "base_weight"
COND_WEIGHT_NAME_PREFIX = "conditional_weight"
TEMP_NAME_PREFIX = "temp"

# TODO(jereliu): add option to force binary tree.

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Utility functions """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def get_nonroot_node_names(family_tree):
    """Get names of non-root nodes of input family tree.

    Args:
        family_tree: (dict of list or None) A dictionary of list of strings to
            specify the family tree between models, if None then assume there's
            no structure (i.e. flat structure).

    Raises:
        (ValueError) If name of any leaf node did not appear in base_pred.
    """
    return np.concatenate(list(family_tree.values()))


def get_parent_node_names(family_tree):
    """Get names of non-leaf nodes of input family tree.

    Args:
        family_tree: (dict of list or None) A dictionary of list of strings to
            specify the family tree between models, if None then assume there's
            no structure (i.e. flat structure).

    Raises:
        (ValueError) If name of any leaf node did not appear in base_pred.
    """
    return np.asarray(list(family_tree.keys()))


def get_leaf_model_names(family_tree):
    """Get names of leaf nodes of input family tree.

    Args:
        family_tree: (dict of list or None) A dictionary of list of strings to
            specify the family tree between models, if None then assume there's
            no structure (i.e. flat structure).

    Raises:
        (ValueError) If name of any leaf node did not appear in base_pred.
    """
    all_node_names = np.concatenate(list(family_tree.values()))
    all_parent_names = get_parent_node_names(family_tree)

    all_leaf_names = [name for name in all_node_names
                      if name not in all_parent_names]

    return all_leaf_names


def get_leaf_ancestry(family_tree):
    """Get ancestry of every leaf nodes of input family tree.

    Args:
        family_tree: (dict of list or None) A dictionary of list of strings to
            specify the family tree between models, if None then assume there's
            no structure (i.e. flat structure).

    Returns:
        (dict of list of str) Dictionary of list of strings containing
            ancestry name of each parent.
    """
    # get mapping from child to parent
    parent_name_dict = dict()
    for parent_name, child_names in family_tree.items():
        for child_name in child_names:
            parent_name_dict.update({child_name: parent_name})

    # get list of child names
    leaf_model_names = get_leaf_model_names(family_tree)

    # recusively build list of ancestor names for each child.
    # TODO(jereliu): Ugly code. Improve with recursion.
    leaf_ancestry_dict = dict()

    for leaf_model in leaf_model_names:
        ancestry_list = [leaf_model]
        # append immediate parent
        if parent_name_dict[leaf_model] == ROOT_NODE_DEFAULT_NAME:
            leaf_ancestry_dict[leaf_model] = ancestry_list
            continue
        else:
            ancestry_list.append(parent_name_dict[leaf_model])
        # append rest of ancestors
        while True:
            child_name = ancestry_list[-1]
            parent_name = parent_name_dict[child_name]
            if parent_name == ROOT_NODE_DEFAULT_NAME:
                break
            ancestry_list.append(parent_name)

        leaf_ancestry_dict[leaf_model] = ancestry_list

    return leaf_ancestry_dict


def compute_cond_weights(X, family_tree,
                         raw_weights_dict=None,
                         parent_temp_dict=None,
                         **kwargs):
    """Computes conditional weights P(child|parent) for each child nodes.

    Args:
        X: (np.ndarray) Input features of dimension (N, D).
        family_tree: (dict of list) A dictionary of list of strings to
            specify the family tree between models.
        raw_weights_dict: (dict of tf.Tensor or None) A dictionary of tf.Tensor
            for raw weights for each child node, dimension (batch_size, n_obs,)
            To be passed to sparse_conditional_weight().
        parent_temp_dict: (dict of tf.Tensor or None) A dictionary of tf.Tensor
            for temp parameter for each parent node, dimension (batch_size,)
            To be passed to sparse_conditional_weight().
        kwargs: Additional parameters to pass to sparse_conditional_weight.

    Returns:
        (dict of tf.Tensor) A dictionary of tf.Tensor for normalized conditional
            weights for each child node.
    """
    # TODO(jereliu): consistency check for name-value correspondence in tensors
    node_weight_dict = {}

    # compute conditional weight for each child node in the tree
    # then aggregate into a dictionary
    for parent_name, child_names in family_tree.items():
        # extract tensor input for sparse_conditional_weight
        weight_raw, temp = None, None
        if raw_weights_dict:
            weight_raw = tf.stack([
                raw_weights_dict[model_name] for
                model_name in child_names], axis=-1)
        if parent_temp_dict:
            temp = tf.convert_to_tensor(parent_temp_dict[parent_name])

        # compute conditional weight
        child_weights = sparse_conditional_weight(X,
                                                  parent_name=parent_name,
                                                  child_names=child_names,
                                                  base_weights=weight_raw,
                                                  temp=temp,
                                                  **kwargs)

        node_weight_dict.update(dict(zip(child_names, child_weights)))

    return node_weight_dict


def compute_leaf_weights(node_weights, family_tree, name=""):
    """Computes the ensemble weight for leaf nodes using Tail-free Process.

    Args:
        node_weights: (dict of tf.Tensor) A dictionary containing the ensemble
            weight (tf.Tensor of float32, dimension (batch_size, n_obs, ) )
        family_tree: (dict of list) A dictionary of list of strings to
            specify the family tree between models.
        name: (str) Name of the output tensor.

    Returns:
        model_weight_tensor (tf.Tensor) A tf.Tensor of float32 specifying the ensemble weight for
            each leaf node. Dimension (batch_size, n_obs, n_leaf_model).
        model_names_list (list of str) A list of string listing name of leaf-node
            models.
    """
    model_ancestry_dict = get_leaf_ancestry(family_tree)

    # for each child, multiply weights of its ancestry
    # TODO(jereliu): Ugly code.
    model_names_list = []
    model_weight_list = []
    for model_name, ancestor_names in model_ancestry_dict.items():
        model_names_list.append(model_name)
        ancestor_weight_list = [
            node_weights[ancestor_name] for ancestor_name in ancestor_names]
        model_weight_tensor = tf.reduce_prod(
            tf.stack(ancestor_weight_list, axis=-1), axis=-1)
        model_weight_list.append(model_weight_tensor)

    model_weight_tensor = tf.stack(model_weight_list, axis=-1, name=name)

    return model_weight_tensor, model_names_list


def check_leaf_models(family_tree, base_pred):
    """Check validity of input family tree, and return names of child models.

    Args:
        family_tree: (dict of list or None) A dictionary of list of strings to
            specify the family tree between models, if None then assume there's
            no structure (i.e. flat structure).
        base_pred: (dict of np.ndarray) A dictionary of out-of-sample prediction
            from base models. For detail, see calibre.adaptive_ensemble.model.

    Raises:
        (ValueError) If root name (_ROOT_NAME) is not found in family_tree.
        (ValueError) If any parent node has less than two child.
        (ValueError) If name of any leaf node did not appear in base_pred.
    """
    # TODO(jereliu): check if there's conflict/collision in model/family names
    # TODO(jereliu): check if there's missing link between nodes

    # check root name
    try:
        family_tree[ROOT_NODE_DEFAULT_NAME]
    except KeyError:
        raise ValueError(
            "Root node name must be '{}'. "
            "However it is not found in family_tree".format(ROOT_NODE_DEFAULT_NAME))

    # check number of child for each parent node
    for parent_name, child_name in family_tree.items():
        if len(child_name) < 2:
            raise ValueError(
                "Number of child node of each parent must be greater than 2."
                "However observed {} child for parent node '{}'".format(
                    len(child_name), parent_name)
            )

    # check all leaf nodes in family_tree exists in base_pred
    leaf_ancestry_dict = get_leaf_ancestry(family_tree)
    for leaf_node_name in list(leaf_ancestry_dict.keys()):
        try:
            base_pred[leaf_node_name]
        except KeyError:
            raise ValueError(
                "model name '{}' in family_tree not found in base_pred.\n"
                "Models available in base_pred are: \n {}".format(
                    leaf_node_name, list(base_pred.keys())))

    return leaf_ancestry_dict


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Model definition for Dependent Tail-free Process Prior """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def prior(X, base_pred, family_tree=None,
          kernel_func=gp.rbf,
          link_func=sparse_softmax,
          ridge_factor=1e-3,
          name="ensemble_weight",
          **kwargs):
    """Defines the nonparametric (tail-free process) prior for p(model, feature).

    Args:
        X: (np.ndarray) Input features of dimension (N, D)
        base_pred: (dict of np.ndarray) A dictionary of out-of-sample prediction
            from base models. For detail, see calibre.adaptive_ensemble.model.
        family_tree: (dict of list or None) A dictionary of list of strings to
            specify the family tree between models, if None then assume there's
            no structure (i.e. flat structure).
        kernel_func: (function) kernel function for base ensemble,
            with args (X, **kwargs). Default to rbf.
        link_func: (function) a link function that transforms the unnormalized
            base ensemble weights to a K-dimension simplex. Default to sparse_softmax.
            This function has args (logits, temp)
        ridge_factor: (float32) ridge factor to stabilize Cholesky decomposition.
        name: (str) name of the ensemble weight node on the computation graph.
        **kwargs: Additional parameters to pass to sparse_conditional_weight.

    Returns:
        model_weights: (tf.Tensor of float32)  Tensor of ensemble model weights
            with dimension (num_batch, num_obs, num_model).
        model_names_list: (list of str) List of model names corresponding to
            the order of model_weights
    """
    if not family_tree:
        family_tree = {ROOT_NODE_DEFAULT_NAME: list(base_pred.keys())}

    check_leaf_models(family_tree, base_pred)

    # build a dictionary of conditional weights for each node in family_tree.
    node_weight_dict = compute_cond_weights(X, family_tree,
                                            kernel_func=kernel_func,
                                            link_func=link_func,
                                            ridge_factor=ridge_factor,
                                            **kwargs)

    # compute model-specific weights for each leaf node.
    return compute_leaf_weights(node_weights=node_weight_dict,
                                family_tree=family_tree,
                                name=name)


def sparse_conditional_weight(X, parent_name, child_names,
                              base_weights=None, temp=None,
                              kernel_func=gp.rbf,
                              link_func=sparse_softmax,
                              ridge_factor=1e-3,
                              **kernel_kwargs):
    """Defines the conditional distribution of model given parent in the tail-free tree.

    Defines the feature-dependent conditional distribution of model as:

        w(model | x ) = link_func( w_model(x) )
        w_model(x) ~ gaussian_process[0, k_w(x)]


    Args:
        X: (np.ndarray) Input features of dimension (N, D)
        parent_name: (str) The name of the mother node.
        child_names: (list of str) A list of model names for each child in the family.
        base_weights: (tf.Tensor of float32 or None) base logits to be passed to
            link_func corresponding to each child. It has dimension
            (batch_size, num_obs, num_model).
        temp: (tf.Tensor of float32 or None) temperature parameter corresponding
            to the parent node to be passed to link_func, it has dimension
            (batch_size, ).
        kernel_func: (function) kernel function for base ensemble,
            with args (X, **kwargs).
        link_func: (function) a link function that transforms the unnormalized
            base ensemble weights to a K-dimension simplex.
            This function has args (logits, temp)
        ridge_factor: (float32) ridge factor to stabilize Cholesky decomposition.
        **kernel_kwargs: Additional parameters to pass to kernel_func through gp.prior.

    Returns:
        (list of tf.Tensor) List normalized ensemble weights, dimension (N, M) with
            dtype float32.
    """
    num_model = len(child_names)

    # define random variables: temperature and raw GP weights
    if not isinstance(temp, tf.Tensor):
        temp = ed.Normal(loc=_TEMP_PRIOR_MEAN,
                         scale=_TEMP_PRIOR_SDEV,
                         name='{}_{}'.format(TEMP_NAME_PREFIX, parent_name))

    if not isinstance(base_weights, tf.Tensor):
        base_weights = tf.stack([
            gp.prior(X, kernel_func=kernel_func,
                     ridge_factor=ridge_factor,
                     name='{}_{}'.format(BASE_WEIGHT_NAME_PREFIX, model_name),
                     **kernel_kwargs)
            for model_name in child_names], axis=-1)

    # define transformed random variables
    weight_transformed = link_func(base_weights, tf.exp(temp),
                                   name='{}_{}'.format(COND_WEIGHT_NAME_PREFIX, parent_name))

    # split into list then return
    # TODO(jereliu): Ugly code.
    weight_transformed = tf.split(weight_transformed, num_model, axis=-1)
    weight_transformed = [tf.squeeze(weight, axis=-1) for weight in weight_transformed]
    return weight_transformed


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Variational Family """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def variational_family(X, base_pred, family_tree=None,
                       gp_vi_family=gp.variational_mfvi,
                       **kwargs):
    """Defines the variational family for tail-free process prior.

    Args:
        X: (np.ndarray) Input features of dimension (N, D)
        base_pred: (dict of np.ndarray) A dictionary of out-of-sample prediction
            from base models. For detail, see calibre.adaptive_ensemble.model.
        family_tree: (dict of list or None) A dictionary of list of strings to
            specify the family tree between models, if None then assume there's
            no structure (i.e. flat structure).
        gp_vi_family: (function) A variational family for node weight
            GPs in the family tree.
        kwargs: Additional arguments to pass to gp_vi_family.

    Returns:
        weight_gp_dict: (dict of ed.RandomVariable) Dictionary of GP random variables
            for each non-root model/model family.
        temp_dict: (dict of ed.RandomVariable) Dictionary of temperature random variables
            for each parent model.
        weight_gp_mean_dict: (dict of tf.Variable) Dictionary of variational parameters for
            the mean of GP.
        weight_gp_vcov_dict: (dict of tf.Variable) Dictionary of variational parameters for
            the stddev or covariance matrix of GP.
        temp_mean_dict: (dict of tf.Variable) Dictionary of variational parameters for
            the mean of temperatures.
        temp_sdev_dict: (dict of tf.Variable) Dictionary of variational parameters for
            the stddev or temperatures.
    """
    if not family_tree:
        family_tree = {ROOT_NODE_DEFAULT_NAME: list(base_pred.keys())}

    check_leaf_models(family_tree, base_pred)

    # define variational family for temperature.
    # TODO(jereliu): Ugly code.
    temp_names = ['temp_{}'.format(name) for name in get_parent_node_names(family_tree)]
    temp_list = [
        list(inference_util.scalar_gaussian_variational(name='vi_{}'.format(temp_name)))
        for temp_name in temp_names]
    temp_arr = np.asarray(temp_list).T

    temp_dict = dict(zip(temp_names, temp_arr[0]))
    temp_mean_dict = dict(zip(temp_names, temp_arr[1]))
    temp_sdev_dict = dict(zip(temp_names, temp_arr[2]))

    # define variational family for GP.
    # TODO(jereliu): Ugly code.
    base_weight_names = ['base_weight_{}'.format(name)
                         for name in get_nonroot_node_names(family_tree)]
    base_weight_list = [list(gp_vi_family(X, name='vi_{}'.format(weight_name), **kwargs))
                        for weight_name in base_weight_names]
    base_weight_arr = np.asarray([weight_par[:-1] for weight_par in base_weight_list]).T
    mixture_par_arr = [weight_par[-1] for weight_par in base_weight_list]

    # prepare outcome containers
    weight_gp_dict = dict(zip(base_weight_names, base_weight_arr[0]))
    weight_gp_mean_dict = dict(zip(base_weight_names, base_weight_arr[1]))
    weight_gp_vcov_dict = dict(zip(base_weight_names, base_weight_arr[2]))
    mixture_par_dict = dict(zip(base_weight_names, mixture_par_arr))

    return (weight_gp_dict, temp_dict,
            weight_gp_mean_dict, weight_gp_vcov_dict,
            temp_mean_dict, temp_sdev_dict,
            mixture_par_dict,)


def variational_family_sample(n_sample, mfvi_mixture,
                              weight_gp_mean_dict, weight_gp_vcov_dict,
                              temp_mean_dict, temp_sdev_dict, mixture_par_dict,
                              gp_sample_func=gp.variational_mfvi_sample):
    """Samples from the variational family for tail-free process prior.

    Args:
        n_sample: (int) Number of samples to draw from variational family.
        mfvi_mixture: (bool) Whether the family is a GP-MF mixture.
        weight_gp_mean_dict: (dict of tf.Variable) Dictionary of variational parameters
            for the mean of GP.
        weight_gp_vcov_dict: (dict of tf.Variable) Dictionary of variational parameters
            for the stddev or covariance matrix of GP.
        temp_mean_dict: (dict of tf.Variable) Dictionary of variational parameters for
            the mean of temperatures.
        temp_sdev_dict: (dict of tf.Variable) Dictionary of variational parameters for
            the stddev or temperatures.
        gp_sample_func: (function) Sampling function for GP weights from
            calibre.model.gaussian_process.

    Returns:
        weight_gp_sample_dict: (dict of tf.Tensor) Dictionary of temperature random
            variables for each parent model.
        temp_sample_dict: (dict of tf.Tensor) Dictionary of GP random variables
            for each non-root model/model family.

    Raises:
        (ValueError) If key names are different for weight_gp_mean_dict and weight_gp_vcov_dict.
        (ValueError) If key names are different for temp_mean_dict and temp_sdev_dict.
    """
    if not set(weight_gp_mean_dict.keys()) == set(weight_gp_vcov_dict.keys()):
        raise ValueError("Key values for 'weight_gp_mean_dict' and 'weight_gp_vcov_dict' should be identical.")

    if not set(temp_mean_dict.keys()) == set(temp_sdev_dict.keys()):
        raise ValueError("Key values for 'temp_mean_dict' and 'temp_sdev_dict' should be identical.")

    # sample raw weight gp
    weight_gp_sample_dict = dict()
    for model_names in weight_gp_mean_dict.keys():
        weight_gp_sample_dict[model_names] = gp_sample_func(
            n_sample,
            weight_gp_mean_dict[model_names],
            weight_gp_vcov_dict[model_names],
            mfvi_mixture=mfvi_mixture,
            mixture_par_list=mixture_par_dict[model_names]
        )

    # sample temperature.
    temp_sample_dict = dict()
    for model_names in temp_mean_dict.keys():
        temp_sample_dict[model_names] = (
            inference_util.sample_scalar_gaussian_variational(
                n_sample,
                temp_mean_dict[model_names],
                temp_sdev_dict[model_names]))

    return weight_gp_sample_dict, temp_sample_dict


variational_mfvi = functools.partial(variational_family,
                                     gp_vi_family=gp.variational_mfvi)
variational_sgpr = functools.partial(variational_family,
                                     gp_vi_family=gp.variational_sgpr)
variational_dgpr = functools.partial(variational_family,
                                     gp_vi_family=gp.variational_dgpr)

variational_mfvi_sample = functools.partial(variational_family_sample,
                                            gp_sample_func=gp.variational_mfvi_sample)
variational_sgpr_sample = functools.partial(variational_family_sample,
                                            gp_sample_func=gp.variational_sgpr_sample)
variational_dgpr_sample = functools.partial(variational_family_sample,
                                            gp_sample_func=gp.variational_dgpr_sample)
