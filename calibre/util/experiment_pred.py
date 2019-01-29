"""Utility functions for model prediction in experiments."""

import numpy as np

# sys.path.extend([os.getcwd()])

from calibre.model import gaussian_process as gp
from calibre.model import tailfree_process as tail_free
from calibre.model import adaptive_ensemble


def prediction_tailfree(X_pred, X_train,
                        base_pred_dict, family_tree,
                        weight_sample_list, resid_sample, temp_sample,
                        default_log_ls_weight=None,
                        default_log_ls_resid=None,
                        ):
    """
    Generates predictive samples for adaptive ensemble

    Args:
        X_pred: (np.ndarray of float32) testing locations, N_new x D
        X_train: (np.ndarray of float32) training locations, N_train x D
        base_pred_dict: (dict of np.ndarray) A dictionary of out-of-sample prediction
            from base models.
        family_tree: (dict of list or None) A dictionary of list of strings to
            specify the family tree between models, if None then assume there's
            no structure (i.e. flat structure).
        weight_sample_list: (list of np.ndarray of float32) List of untransformed
            ensemble weight for each base model, shape (M, N_train).
        resid_sample: (np.ndarray of float32) GP samples for residual process
            corresponding to X_pred, shape (M, N_train).
        temp_sample: (np.ndarray of float32) Temperature random variables
            for each parent model.
        default_log_ls_weight: (float32) default value for length-scale parameter for
            weight GP.
        default_log_ls_resid: (float32) default value for length-scale parameter for
            residual GP.

    Returns:
        ensemble_sample: (np.ndarray) Samples from full posterior predictive.
        ensemble_mean: (np.ndarray) Samples from posterior mean.
    """

    if not default_log_ls_weight:
        default_log_ls_weight = np.log(0.35)
    if not default_log_ls_resid:
        default_log_ls_resid = np.log(0.1)

    default_log_ls_weight = default_log_ls_weight.astype(np.float32)
    default_log_ls_resid = default_log_ls_resid.astype(np.float32)

    # compute GP prediction for weight GP and residual GP
    model_weight_valid_sample = []
    for model_weight_sample in weight_sample_list:
        model_weight_valid_sample.append(
            gp.sample_posterior_full(X_new=X_pred, X=X_train,
                                     f_sample=model_weight_sample.T,
                                     ls=np.exp(default_log_ls_weight),
                                     kernel_func=gp.rbf).T.astype(np.float32)
        )

    ensemble_resid_valid_sample = (
        gp.sample_posterior_full(X_new=X_pred, X=X_train,
                                 f_sample=resid_sample.T,
                                 ls=np.exp(default_log_ls_resid),
                                 kernel_func=gp.rbf).T
    )

    # compute sample for posterior mean
    raw_weights_dict = dict(zip(tail_free.get_nonroot_node_names(family_tree),
                                model_weight_valid_sample))
    parent_temp_dict = dict(zip(tail_free.get_parent_node_names(family_tree),
                                temp_sample))

    (ensemble_sample_val, ensemble_mean_val,
     ensemble_weights_val, cond_weights_dict_val,
     ensemble_model_names) = (
        adaptive_ensemble.sample_posterior_tailfree(X=X_pred,
                                                    base_pred_dict=base_pred_dict,
                                                    family_tree=family_tree,
                                                    weight_gp_dict=raw_weights_dict,
                                                    temp_dict=parent_temp_dict,
                                                    resid_gp_sample=ensemble_resid_valid_sample,
                                                    log_ls_weight=default_log_ls_weight))

    return (ensemble_sample_val, ensemble_mean_val,
            ensemble_weights_val, cond_weights_dict_val,
            ensemble_model_names)
