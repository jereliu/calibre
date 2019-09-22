"""Functions to define tf graph and sessions for MCMC inference."""

import os
import time

from importlib import reload

import pickle as pk
import pandas as pd

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed

# sys.path.extend([os.getcwd()])

from calibre.model import tailfree_process as tail_free
from calibre.model import adaptive_ensemble


def make_inference_graph_tailfree(X_train, y_train, base_pred, family_tree,
                                  default_log_ls_weight=None,
                                  default_log_ls_resid=None,
                                  num_mcmc_samples=1000, 
                                  num_burnin_steps=5000):
    """Defines computation graph for MCMC sampling with tailfree model.

    Args:
        X_train: (np.ndarray) Input features of dimension (N, D)
        y_train: (np.ndarray) Training labels of dimension (N, )
        base_pred: (dict of np.ndarray) A dictionary of out-of-sample prediction
            from base models. For each item in the dictionary,
            key is the model name, and value is the model prediction with
            dimension (N, ).
        family_tree: (dict of list or None) A dictionary of list of strings to
            specify the family tree between models, if None then assume there's
            no structure (i.e. flat).
        default_log_ls_weight: (float32) default value for length-scale parameter for
            weight GP.
        default_log_ls_resid: (float32) default value for length-scale parameter for
            residual GP.
        num_mcmc_samples: (int) Integer number of Markov chain draws.
        num_burnin_steps: (int) Number of chain steps to take before starting to
            collect results.

    Returns:
        mcmc_graph (Graph) A computation graph for MCMC that contains
            init ops, parameter samples, and sampling states.
        init_op (tf.Operation) Initialization op
        parameter_samples (dict of tf.Tensors) Dictionary of parameters and their
            MCMC samples of shape (param_dim, num_mcmc_samples)
        is_accepted (tf.Tensor) A tensor indicating whether each mcmc samples is accepted.
    """

    INFER_LS_PARAM = False
    N = X_train.shape[0]
    
    if not default_log_ls_weight or not default_log_ls_resid:
        INFER_LS_PARAM = True

    mcmc_graph = tf.Graph()
    with mcmc_graph.as_default():
        # build likelihood explicitly
        log_joint = ed.make_log_joint_fn(adaptive_ensemble.model_tailfree)

        # aggregate node-specific variable names
        cond_weight_temp_names = ['temp_{}'.format(model_name) for
                                  model_name in
                                  tail_free.get_parent_node_names(family_tree)]
        node_weight_names = ['base_weight_{}'.format(model_name) for
                             model_name in
                             tail_free.get_nonroot_node_names(family_tree)]
        node_specific_varnames = cond_weight_temp_names + node_weight_names
        
        if INFER_LS_PARAM:
            # treat ls_weight and ls_resid as part of model parameter
            # and pass them to log likelihood function.
            def target_log_prob_fn(ls_weight, ls_resid,
                                   sigma, ensemble_resid,
                                   *node_specific_positional_args):
                """Unnormalized target density as a function of states."""
                # build kwargs for base model weight using positional args
                node_specific_kwargs = dict(zip(node_specific_varnames,
                                                node_specific_positional_args))

                return log_joint(X=X_train,
                                 base_pred=base_pred,
                                 family_tree=family_tree,
                                 y=y_train.squeeze(),
                                 ls_weight=ls_weight,
                                 ls_resid=ls_resid,
                                 sigma=sigma,
                                 ensemble_resid=ensemble_resid,
                                 **node_specific_kwargs)
        else:
            def target_log_prob_fn(sigma, ensemble_resid,
                                   *node_specific_positional_args):
                """Unnormalized target density as a function of states."""
                # build kwargs for base model weight using positional args
                node_specific_kwargs = dict(zip(node_specific_varnames,
                                                node_specific_positional_args))
    
                return log_joint(X=X_train,
                                 base_pred=base_pred,
                                 family_tree=family_tree,
                                 y=y_train.squeeze(),
                                 log_ls_weight=default_log_ls_weight,
                                 log_ls_resid=default_log_ls_resid,
                                 sigma=sigma,
                                 ensemble_resid=ensemble_resid,
                                 **node_specific_kwargs)

        # set up state container
        initial_state = [
                            tf.constant(0.1, name='init_sigma'),
                            tf.random_normal([N], stddev=0.01,
                                             name='init_ensemble_resid'),
                        ] + [
                            tf.random_normal([], stddev=0.01,
                                             name='init_{}'.format(var_name)) for
                            var_name in cond_weight_temp_names
                        ] + [
                            tf.random_normal([N], stddev=0.01,
                                             name='init_{}'.format(var_name)) for
                            var_name in node_weight_names
                        ]
        
        if INFER_LS_PARAM:
            initial_state = [tf.constant(-1., name='init_ls_weight'),
                             tf.constant(-1., name='init_ls_resid'),
                             ] + initial_state

        # set up HMC transition kernel
        step_size = tf.get_variable(
            name='step_size',
            initializer=1.,
            use_resource=True,  # For TFE compatibility.
            trainable=False)

        hmc = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            num_leapfrog_steps=3,
            step_size=step_size,
            step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(num_burnin_steps))

        # set up main sampler
        state, kernel_results = tfp.mcmc.sample_chain(
            num_results=num_mcmc_samples,
            num_burnin_steps=num_burnin_steps,
            current_state=initial_state,
            kernel=hmc,
            parallel_iterations=1
        )

        # setup output tensors
        parameter_samples = dict()
        param_init_idx = 0
        if INFER_LS_PARAM:
            param_init_idx = 2
            parameter_samples["ls_weight_sample"] = state[0]
            parameter_samples["ls_resid_sample"] = state[1]
        
        # collect other parameters
        parameter_samples[
            "sigma_sample"] = state[param_init_idx + 0]
        parameter_samples[
            "ensemble_resid_sample"] = state[param_init_idx + 1]
        parameter_samples["temp_sample"] = (
            state[param_init_idx + 2:
                  param_init_idx + 2 + len(cond_weight_temp_names)])
        parameter_samples["weight_sample"] = (
            state[param_init_idx + 2 + len(cond_weight_temp_names):]
        )
        
        # set up init op
        with tf.name_scope("init_op") as scope:
            init_op = tf.global_variables_initializer()

        # set up mcmc sampler information
        with tf.name_scope("mcmc_info"):
            is_accepted = tf.identity(kernel_results.is_accepted,
                                      name="acceptance")

        mcmc_graph.finalize()

    return mcmc_graph, init_op, parameter_samples, is_accepted


def run_sampling(mcmc_graph, init_op, parameter_samples, is_accepted):
    """

    Args:
        mcmc_graph: (tf.Graph) A computation graph for MCMC that contains
            init ops, parameter samples, and sampling states.
        init_op: (tf.Operation) Initialization op
        parameter_samples: (dict of tf.Tensors) Dictionary of parameters and their
            MCMC samples of shape (param_dim, num_mcmc_samples)
        is_accepted: (tf.Tensor) A tensor indicating whether each mcmc samples is accepted.

    Returns:
        parameter_samples_val: (dict of np.ndarray) Dictionary of
            parameters and the MCMC samples evaluated by tf session,
            shape (param_dim, num_mcmc_samples)
    """

    time_start = time.time()
    with tf.Session(graph=mcmc_graph) as sess:
        init_op.run()
        [
            parameter_samples_val,
            is_accepted_,
        ] = sess.run(
            [
                parameter_samples,
                is_accepted,
            ])

        total_min = (time.time() - time_start) / 60.
        print('Acceptance Rate: {}'.format(np.mean(is_accepted_)))
        print('Total time: {:.2f} min'.format(total_min))
        sess.close()

    return parameter_samples_val
