from urllib.request import urlretrieve
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from lifelines.metrics import concordance_index

from hazardous.metrics.brier_score import (
    brier_score, integrated_brier_score
)


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
            predicted_scores=survival_to_risk_estimate(survival_curves),
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