"""Model definitions and sampling functions for Dependent Tail-free Process.

#### References

[1]: Alejandro Jara and Timothy Hanson. A class of mixtures of dependent tail-free
        processes. _Biometrika. 2011;98(3):553-566._. 2011

[2]: Eswar G. Phadia. Tailfree Processes. In: Prior Processes and Their Applications.
        _Springer Series in Statistics. Springer, Cham_. 2016.

[3]: Subhashis Ghoshal. A Invitation to Bayesian Nonparametrics.
        _Presentation Slides_, 2011.
        https://www.eurandom.tue.nl/EURANDOM_chair/minicourseghoshal.pdf
"""
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed

from calibre.model import gaussian_process as gp

from calibre.util.model import sparse_softmax

_TEMP_PRIOR_MEAN = -5.
_TEMP_PRIOR_SDEV = 1.
_ROOT_NODE_NAME = "root"

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


def get_nonleaf_node_names(family_tree):
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
    all_parent_names = get_nonleaf_node_names(family_tree)

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
        ancestry_list = [leaf_model, parent_name_dict[leaf_model]]
        while True:
            child_name = ancestry_list[-1]
            parent_name = parent_name_dict[child_name]
            if parent_name == _ROOT_NODE_NAME:
                break
            ancestry_list.append(parent_name)

        leaf_ancestry_dict[leaf_model] = ancestry_list

    return leaf_ancestry_dict


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
        (ValueError) If name of any leaf node did not appear in base_pred.
    """
    # TODO(jereliu): check if there's missing link between nodes

    # check root name
    try:
        family_tree[_ROOT_NODE_NAME]
    except KeyError:
        raise ValueError(
            "Root node name must be '{}'. "
            "However it is not found in family_tree".format(_ROOT_NODE_NAME))

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
""" Model definition for Dependent Tailfree Process Prior """
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
        family_tree = {_ROOT_NODE_NAME: list(base_pred.keys())}

    model_ancestry_dict = check_leaf_models(family_tree, base_pred)

    # build a dictionary of conditional weights for each node in family_tree.
    node_weight_dict = {}
    for parent_name, child_names in family_tree.items():
        child_weights = sparse_conditional_weight(X,
                                                  parent_name=parent_name,
                                                  child_names=child_names,
                                                  kernel_func=kernel_func,
                                                  link_func=link_func,
                                                  ridge_factor=ridge_factor,
                                                  **kwargs)
        node_weight_dict.update(dict(zip(child_names, child_weights)))

    # compute model-specific weights for each leaf node.
    # TODO(jereliu): Ugly code.
    model_names_list = []
    model_weight_list = []
    for model_name, ancestor_names in model_ancestry_dict.items():
        model_names_list.append(model_name)
        ancestor_weight_list = [
            node_weight_dict[ancestor_name] for ancestor_name in ancestor_names]
        model_weight_tensor = tf.reduce_prod(
            tf.stack(ancestor_weight_list, axis=-1), axis=-1)
        model_weight_list.append(model_weight_tensor)

    model_weights = tf.stack(model_weight_list, axis=-1, name=name)

    return model_weights, model_names_list


def sparse_conditional_weight(X, parent_name, child_names,
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
    temp = ed.Normal(loc=_TEMP_PRIOR_MEAN,
                     scale=_TEMP_PRIOR_SDEV,
                     name='temp_{}'.format(parent_name))

    weight_raw = tf.stack([
        gp.prior(X, kernel_func=kernel_func,
                 ridge_factor=ridge_factor,
                 name='base_weight_{}'.format(model_name),
                 **kernel_kwargs)
        for model_name in child_names], axis=1)

    # define transformed random variables
    weight_transformed = link_func(weight_raw, tf.exp(temp),
                                   name='conditional_weight_{}'.format(parent_name))

    # split into list then return
    # TODO(jereliu): Ugly code.
    weight_transformed = tf.split(weight_transformed, num_model, -1)
    weight_transformed = [tf.squeeze(weight, axis=-1) for weight in weight_transformed]
    return weight_transformed
