"""Utility functions implementing other ensemble methods.

#### References

[1]: Lianfa Li et al. An Ensemble Spatiotemporal Model for Predicting PM2.5 Concentrations.
     _Int J Environ Res Public Health. 14(5): 549._, 2017
     https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5451999/

[2]: Philippe Rigollet and Alexandre B. Tsybakov. Sparse Estimation by Exponential
     Weighting. _Statistical Science_, Vol. 27, No. 4, 558–575, 2012
"""
from abc import ABC, abstractmethod

import numpy as np
import scipy.optimize as opt

from pygam import LinearGAM, s, te, l
from pygam.terms import TermList


# TODO(jereliu): change to uncertainty estimate to variance rather than quantile

class EnsembleModel(ABC):
    def __init__(self, ensemble_model_name):
        self.name = ensemble_model_name

    @abstractmethod
    def train(self, X, y, base_pred):
        """Trains ensemble model based on data and base predictions.

        Args:
            X: (np.ndarray) Training features, shape (N, D)
            y: (np.ndarray)  Training labels, shape (N, 1)
            base_pred: (dict of np.ndarray) Dictionary of base model predictions
                With keys (str) being model name, and values (np.ndarray) being
                predictions corresponds to X and y.

        """
        pass

    @abstractmethod
    def predict(self, X, base_pred):
        """Predicts label based on feature and base model.

        Args:
            X: (np.ndarray) Training features, shape (N, D)
            base_pred: (dict of np.ndarray) Dictionary of base model predictions
                With keys (str) being model name, and values (np.ndarray) being
                predictions corresponds to X and y.
        """
        pass


class AveragingEnsemble(EnsembleModel):
    """Implements naive averaging method."""

    def __init__(self):
        super().__init__("Averaging Ensemble")
        self.model_weight = None

    def train(self, X, y, base_pred):
        """Trains ensemble model based on data and base predictions.

        Adds value to class attribute "model_weight"

        Args:
            X: (np.ndarray) Training features, shape (N, D)
            y: (np.ndarray)  Training labels, shape (N, 1)
            base_pred: (dict of np.ndarray) Dictionary of base model predictions
                With keys (str) being model name, and values (np.ndarray) being
                predictions corresponds to X and y.
        """
        self.model_weight = {model_name: 1 / len(base_pred)
                             for model_name in base_pred.keys()}

    def predict(self, X, base_pred):
        """Predicts label based on feature and base model.

        Args:
            X: (np.ndarray) Training features, shape (N, D)
            base_pred: (dict of np.ndarray) Dictionary of base model predictions
                With keys (str) being model name, and values (np.ndarray) being
                predictions corresponds to X and y.

        Returns:
            (np.ndarray) ensemble prediction

        Raises:
            (ValueError) If self.model_weight is empty.
        """
        if not self.model_weight:
            raise ValueError("Attribute model_weight empty."
                             "Model was not trained properly.")

        predict = [base_pred[model_name] * self.model_weight[model_name]
                   for model_name in self.model_weight.keys()]

        prediction_var = None

        return np.sum(np.asarray(predict), axis=0), prediction_var


class ExpWeighting(EnsembleModel):
    """Implements exponential weighting in [2]."""

    def __init__(self):
        super().__init__("Exponential Weighting")
        self.model_weight = None
        self.model_resid = dict()
        self.temp_param = None

    def train(self, X, y, base_pred):
        """Trains ensemble model based on data and base predictions.

        Adds value to class attribute "model_weight"

        Args:
            X: (np.ndarray) Training features, shape (N, D)
            y: (np.ndarray)  Training labels, shape (N, 1)
            base_pred: (dict of np.ndarray) Dictionary of base model predictions
                With keys (str) being model name, and values (np.ndarray) being
                predictions corresponds to X and y.
        """

        self.model_resid = {model_name: np.mean((y - base_pred[model_name]) ** 2)
                            for model_name in base_pred.keys()}

        # greedy tuning for minimize error.
        param_cand_list = self._make_param_candidates(num_candidates=50)

        # choose optimal temperature parameter
        self.temp_param = self._tune_temperature_param(
            y, base_pred, param_cand_list)

    def predict(self, X, base_pred):
        """Predicts label based on feature and base model.

        Args:
            X: (np.ndarray) Training features, shape (N, D)
            base_pred: (dict of np.ndarray) Dictionary of base model predictions
                With keys (str) being model name, and values (np.ndarray) being
                predictions corresponds to X and y.

        Returns:
            (np.ndarray) ensemble prediction and predictive intervals

        Raises:
            (ValueError) If self.model_weight is empty.
        """
        if not self.temp_param:
            raise ValueError("Attribute temp_param empty."
                             "Model was not trained properly.")

        prediction, model_weight = self._exponential_sum(self.temp_param, base_pred,
                                                         return_normal_weight=True)
        self.model_weight = model_weight

        prediction_var = None

        return prediction, prediction_var

    def _exponential_sum(self, temp_param, base_pred, return_normal_weight=False):
        """Sum over base predictions under given temperature parameter.

        Args:
            temp_param: (float) temperature parameter under exponential averaging
            base_pred: (dict of np.ndarray) Dictionary of base model predictions
                With keys (str) being model name, and values (np.ndarray) being
                predictions corresponds to X and y.
            return_normal_weight: (bool) Whether to return normalized weight.

        Returns:
            (np.ndarray) ensemble prediction under temp_param.
        """
        exp_weight = {model_name: np.exp(- self.model_resid[model_name] / temp_param)
                      for model_name in base_pred.keys()}
        weight_denom = np.sum(list(exp_weight.values()))

        predict = [base_pred[model_name] * exp_weight[model_name] / weight_denom
                   for model_name in base_pred.keys()]

        if return_normal_weight:
            exp_weight = {model_name: exp_weight[model_name] / weight_denom
                          for model_name in exp_weight.keys()}

        return np.sum(np.asarray(predict), axis=0), exp_weight

    def _make_param_candidates(self, num_candidates=50):
        """Makes a list of candidates for temperature parameter.

        Based on quantiles of error residual. 60% candidates for error quantiles,
        and 40% for upper/low extremes

        Returns:
            (np.ndarray of float32) a list of candidates for temperature parameter.
        """
        error_arr = np.asarray(list(self.model_resid.values()))
        num_quantile = int(num_candidates * 0.6)
        num_extrem_low = int(num_candidates * 0.2)
        num_extrem_high = num_candidates - num_quantile - num_extrem_low

        error_quantiles = np.percentile(error_arr,
                                        np.linspace(0, 1, num_quantile) * 100)
        extrem_quantiles = np.concatenate(
            [np.min(error_arr) * np.linspace(0, 1, num_extrem_low + 1)[1:],
             np.max(error_arr) * np.linspace(1, 2, num_extrem_high + 1)[1:]],
            axis=0)

        return np.concatenate([error_quantiles, extrem_quantiles])

    def _tune_temperature_param(self, y, base_pred, param_cand_list):
        """Selects best temperature parameter"""
        error_best = np.inf
        param_best = None

        for param_cand in param_cand_list:
            pred_cur, _ = self._exponential_sum(param_cand, base_pred)
            error_new = np.mean((y - pred_cur) ** 2)
            if error_best > error_new:
                param_best = param_cand
                error_best = error_new

        return param_best


class CVStacking(EnsembleModel):
    def __init__(self):
        super().__init__("Cross-validated Stacking")
        self.model_weight = None

    def train(self, X, y, base_pred):
        """Trains ensemble model based on data and base predictions.

        Adds value to class attribute "model_weight"

        Args:
            X: (np.ndarray) Training features, shape (N, D)
            y: (np.ndarray)  Training labels, shape (N, 1)
            base_pred: (dict of np.ndarray) Dictionary of base model predictions
                With keys (str) being model name, and values (np.ndarray) being
                predictions corresponds to X and y.
        """
        model_names = list(base_pred.keys())
        model_error_array = (np.expand_dims(y.squeeze(), -1) -
                             np.asarray(list(base_pred.values())).T)

        model_weight_est = self._estimate_simplex_weight(
            base_error=model_error_array)

        self.model_weight = dict(zip(model_names,
                                     model_weight_est))

    def predict(self, X, base_pred):
        """Predicts label based on feature and base model.

        Args:
            X: (np.ndarray) Training features, shape (N, D)
            base_pred: (dict of np.ndarray) Dictionary of base model predictions
                With keys (str) being model name, and values (np.ndarray) being
                predictions corresponds to X and y.

        Returns:
            (np.ndarray) ensemble prediction and predictive intervals

        Raises:
            (ValueError) If self.model_weight is empty.
        """
        if not self.model_weight:
            raise ValueError("Attribute model_weight empty."
                             "Model was not trained properly.")

        predict = [base_pred[model_name] * self.model_weight[model_name]
                   for model_name in self.model_weight.keys()]
        prediction_var = None

        return np.sum(np.asarray(predict), axis=0), prediction_var

    @staticmethod
    def _estimate_simplex_weight(base_error):
        """Solves a strictly convex quadratic program.

        Minimize     1/2 x^T G x - a^T x, where G = E^TE
        Subject to   C.T x >= b

        Args:
            base_error: (np.ndarray) The "E" matrix in above equation.
                with shape (n_obs, n_model)

        Returns:
            (np.ndarray) the 'x' matrix subject to simplex constraint.
        """
        # prepare constraint matrix
        dim_w = base_error.shape[1]
        G = base_error.T.dot(base_error)

        C_eq = np.ones(dim_w)[np.newaxis, :]

        def square_loss(w):
            return np.dot(w, G).dot(w)

        constraints_list = [{'type': 'eq', 'fun': lambda w: C_eq.dot(w) - 1}]
        bounds = [(0, None)] * dim_w

        opt_res = opt.minimize(square_loss,
                               x0=np.array([1 / dim_w] * dim_w),
                               method='SLSQP',
                               constraints=constraints_list,
                               bounds=bounds,
                               tol=1e-10)

        if not opt_res.success:
            raise ValueError("Optimization not successful.")

        return opt_res.x


class GAMEnsemble(EnsembleModel):
    """Implements GAM ensemble in [1]."""

    def __init__(self, nonlinear_ensemble=False, residual_process=True):
        """
        Initializer.

        Args:
            nonlinear_ensemble: (bool) Whether use nonlinear term to transform base model.
            residual_process: (bool) Whether model residual process.
        """
        model_name = (
            "Generalized Additive Ensemble" if residual_process
            else "{} Stacking".format("Nonlinear" if nonlinear_ensemble else "Linear"))

        super().__init__(model_name)
        self.gam_model = None
        self.nonlinear_ensemble = nonlinear_ensemble
        self.model_residual = residual_process

    def train(self, X, y, base_pred):
        """Trains ensemble model based on data and base predictions.

        Adds value to class attribute "model_weight"

        Args:
            X: (np.ndarray) Training features, shape (N, D)
            y: (np.ndarray)  Training labels, shape (N, 1)
            base_pred: (dict of np.ndarray) Dictionary of base model predictions
                With keys (str) being model name, and values (np.ndarray) being
                predictions corresponds to X and y.
        """
        # build feature and  gam terms
        ens_feature, feature_terms = self._build_ensemble_feature(X, base_pred)

        # define model
        self.gam_model = LinearGAM(feature_terms)

        # additional fine-tuning
        lam_grid = self._build_lambda_grid(n_grid=100)
        self.gam_model.gridsearch(X=ens_feature, y=y, lam=lam_grid,
                                  progress=False)

    def predict(self, X, base_pred):
        """Predicts label based on feature and base model.

        Args:
            X: (np.ndarray) Training features, shape (N, D)
            base_pred: (dict of np.ndarray) Dictionary of base model predictions
                With keys (str) being model name, and values (np.ndarray) being
                predictions corresponds to X and y.

        Returns:
            (np.ndarray) ensemble prediction and variance

        Raises:
            (ValueError) If self.model_weight is empty.
        """
        if not self.gam_model:
            raise ValueError("Attribute gam_model empty."
                             "Model was not trained properly.")

        # build feature and  gam terms
        ens_feature, _ = self._build_ensemble_feature(X, base_pred)

        # prediction
        prediction = self.gam_model.predict(ens_feature)
        prediction_var = ((self.gam_model.prediction_intervals(
            ens_feature, width=.95)[:, 1] - prediction) / 2) ** 2

        return prediction, prediction_var

    def _build_ensemble_feature(self, X, base_pred):
        """Builds featurre array and corresponding GAM TermList.

        Terms corresponding to X will be summation of
            dimension-wise splines, plus a tensor-product term across all dimension.

        """
        ensemble_term_func = s if self.nonlinear_ensemble else l

        ens_feature = np.asarray(list(base_pred.values())).T
        term_list = [ensemble_term_func(dim_index) for dim_index in range(ens_feature.shape[1])]

        # optionally, add residual process
        if self.model_residual:
            # build gam terms
            term_list += [s(dim_index) for dim_index in
                          range(ens_feature.shape[1],
                                ens_feature.shape[1] + X.shape[1])]
            if X.shape[1] > 1:
                term_list += [te(*list(ens_feature.shape[1] +
                                       np.array(range(X.shape[1]))))]

            # update features
            ens_feature = np.concatenate([ens_feature, X], axis=1)

        gam_feature_terms = TermList(*term_list)

        return ens_feature, gam_feature_terms

    def _build_lambda_grid(self, n_grid=100):
        # count actual number of terms in each nonlinear term
        # (e.g. te(0, 1) will actually have two terms)
        n_terms = np.sum([len(model_term._terms) if model_term.istensor else 1
                          for model_term in self.gam_model.terms])
        lam = np.random.rand(n_grid, n_terms)
        # rescale to between (0, 1)
        lam_norm = (lam - np.min(lam)) / (np.max(lam) - np.min(lam))

        return np.exp((lam_norm - 0.5) * 6)
