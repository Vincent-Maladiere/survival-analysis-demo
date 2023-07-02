from urllib.request import urlretrieve
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from lifelines.metrics import concordance_index
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline, _final_estimator_has
from sklearn.utils.metaestimators import available_if

from hazardous.metrics.brier_score import (
    brier_score, integrated_brier_score
)
from hazardous.utils import check_y_survival


def show_progress(block_num, block_size, total_size):
    current_size = round(block_num * block_size / total_size * 100, 2)
    print(f"{current_size:.2f} %", end="\r")


def download_theoretical_hazards():
    data_folder = Path("data")
    hazards_file = data_folder / "truck_failure_10k_hazards.npz"
    data_url = (
        "https://github.com/soda-inria/survival-analysis-benchmark"
        "/releases/download/jupytercon-2023-tutorial-data/truck_failure.zip"
    )

    if not hazards_file.exists():
        print(f"Downloading {data_url}...")

        try:
            tmp_file, _ = urlretrieve(data_url, reporthook=show_progress)

            with ZipFile(tmp_file, "r") as zip_file:
                zip_file.extractall(
                    data_folder,
                    members=["truck_failure_10k_hazards.npz"]
                )
        except Exception:
            if Path(tmp_file).exists():
                Path(tmp_file).unlink()
            raise
    else:
        print(f"Reusing downloaded {hazards_file}")


def survival_to_risk_estimate(survival_probs_matrix):
    return -np.log(survival_probs_matrix + 1e-8).sum(axis=1)


class SurvivalAnalysisEvaluator:
    
    def __init__(self, y_train, y_test, time_grid):
        self.model_data = {}
        self.y_train = y_train
        self.y_test = y_test
        self.time_grid = time_grid
        
    def add_model(self, model_name, survival_curves):
        survival_curves = np.asarray(survival_curves)
        _, brier_scores = brier_score(
            y_train=self.y_train,
            y_test=self.y_test,
            y_pred=survival_curves,
            times=self.time_grid,
        )
        ibs = integrated_brier_score(
            y_train=self.y_train,
            y_test=self.y_test,
            y_pred=survival_curves,
            times=self.time_grid,
        )
        c_index = concordance_index(
            event_times=self.y_test["duration"],
            event_observed=self.y_test["event"],
            predicted_scores=-survival_to_risk_estimate(survival_curves),
        )
        self.model_data[model_name] = {
            "brier_scores": brier_scores,
            "ibs": ibs,
            "c_index": c_index,
            "survival_curves": survival_curves,
        }

    def metrics_table(self):
        return pd.DataFrame([
            {
                "Model": model_name,
                "IBS": info["ibs"],
                "C-index": info["c_index"],
            }
            for model_name, info in self.model_data.items()
        ]).round(decimals=4)
        
    def plot(self, model_names=None):
        if model_names is None:
            model_names = list(self.model_data.keys())
        fig, ax = plt.subplots(figsize=(12, 5))
        self._plot_brier_scores(model_names, ax=ax)

    def _plot_brier_scores(self, model_names, ax):
        for model_name in model_names:
            info = self.model_data[model_name]
            ax.plot(
                self.time_grid,
                info["brier_scores"],
                label=f"{model_name}, IBS:{info['ibs']:.3f}");
        ax.set(
            title="Time-varying Brier score (lower is better)",
            xlabel="time (days)",
        )
        ax.legend()
        
    def __call__(self, model_name, survival_curves, model_names=None):
        self.add_model(model_name, survival_curves)
        self.plot(model_names=model_names)
        return self.metrics_table()
    

class LifelinesWrapper(BaseEstimator):

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **kwargs):
        event, duration = check_y_survival(y)

        X_copy = X.copy()
        if not isinstance(X, pd.DataFrame):
            X_copy = pd.DataFrame(X_copy)
        
        X_copy["event"] = event
        X_copy["duration"] = duration

        self.estimator.fit(
            X_copy,
            duration_col="duration",
            event_col="event",
            **kwargs,
        )

        feature_weights = self.estimator.params_
        self.feature_names_in_ = feature_weights.index
        self.coef_ = feature_weights.values

        return self
    
    def predict_survival_function(self, X, times=None):
        check_is_fitted(self, "coef_")
        survival_probas = self.estimator.predict_survival_function(X, times=times)
        return survival_probas.T.values
    
    def predict_cumulative_hazard(self, X, times=None):
        check_is_fitted(self, "coef_")
        cumulative_hazards = self.estimator.predict_cumulative_hazard(X, times=times)
        return cumulative_hazards.T.values
    
    def __repr__(self):
        return repr(self.estimator)


@available_if(_final_estimator_has("predict_cumulative_hazard_function"))
def predict_cumulative_hazard_function(self, X, **kwargs):
    """Predict cumulative hazard function.

    The cumulative hazard function for an individual
    with feature vector :math:`x` is defined as

    .. math::

        H(t \\mid x) = \\exp(x^\\top \\beta) H_0(t) ,

    where :math:`H_0(t)` is the baseline hazard function,
    estimated by Breslow's estimator.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        Data matrix.

    Returns
    -------
    cum_hazard : ndarray, shape = (n_samples,)
        Predicted cumulative hazard functions.
    """
    Xt = X
    for _, _, transform in self._iter(with_final=False):
        Xt = transform.transform(Xt)
    return self.steps[-1][-1].predict_cumulative_hazard_function(Xt, **kwargs)


@available_if(_final_estimator_has("predict_survival_function"))
def predict_survival_function(self, X, **kwargs):
    """Predict survival function.

    The survival function for an individual
    with feature vector :math:`x` is defined as

    .. math::

        S(t \\mid x) = S_0(t)^{\\exp(x^\\top \\beta)} ,

    where :math:`S_0(t)` is the baseline survival function,
    estimated by Breslow's estimator.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        Data matrix.

    Returns
    -------
    survival : ndarray, shape = (n_samples,)
        Predicted survival functions.
    """
    Xt = X
    for _, _, transform in self._iter(with_final=False):
        Xt = transform.transform(Xt)
    return self.steps[-1][-1].predict_survival_function(Xt, **kwargs)


def patch_pipeline():
    Pipeline.predict_survival_function = predict_survival_function
    Pipeline.predict_cumulative_hazard_function = predict_cumulative_hazard_function

    
patch_pipeline()