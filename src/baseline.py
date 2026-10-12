"""Level 0 naive baseline model predicting overall mean rental demand."""

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class Level0Baseline:
    """Predicts overall training set mean demand as an unconditioned benchmark."""

    def __init__(self, dataset_path: str = "dataset/hour.csv"):
        self.dataset_path = dataset_path
        self.mean_demand: float = 0.0
        self.rounded_demand: int = 0
        self.mae: float = 0.0
        self.sample_size: int = 0
        self._fit()

    def _fit(self) -> None:
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Historical dataset not found at: {self.dataset_path}"
            )
        df = pd.read_csv(self.dataset_path)
        if "cnt" not in df.columns:
            raise KeyError("Target column 'cnt' not found in dataset.")

        self.sample_size = len(df)
        self.mean_demand = float(df["cnt"].mean())
        self.rounded_demand = int(round(self.mean_demand))
        self.mae = float(np.mean(np.abs(df["cnt"] - self.mean_demand)))

    def predict(self, input_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return the Level 0 baseline prediction, ignoring input features."""
        return {
            "predicted_cnt": self.rounded_demand,
            "predicted_cnt_raw": self.mean_demand,
            "mae": round(self.mae, 2),
            "baseline_type": "Level 0 Baseline (Global Training Set Mean)",
            "is_fallback": True,
            "fallback_warning": "WARN_SYS_01: Model artifact unavailable; showing Level 0 fallback.",
            "note": (
                "This is a naive baseline estimate ignoring operational conditions. "
                "Trained ML regression models (L1-L5) will be evaluated in subsequent sprints."
            ),
        }


_baseline_instance: Optional[Level0Baseline] = None


def get_baseline_instance(dataset_path: str = "dataset/hour.csv") -> Level0Baseline:
    """Get or create singleton Level0Baseline instance."""
    global _baseline_instance
    if _baseline_instance is None:
        _baseline_instance = Level0Baseline(dataset_path)
    return _baseline_instance


def predict_demand_fallback(inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience helper to obtain a Level 0 fallback prediction."""
    baseline = get_baseline_instance()
    return baseline.predict(inputs)


if __name__ == "__main__":
    baseline = get_baseline_instance()
    print("=== Level 0 Baseline ===")
    print(f"Records: {baseline.sample_size}")
    print(f"Mean demand: {baseline.mean_demand:.2f} bikes")
    print(f"Baseline MAE: {baseline.mae:.2f} bikes")
    sample_res = baseline.predict({"hour": 17, "season": "Summer", "weather": "Clear", "working_day": "Yes"})
    print("Sample prediction:", sample_res["predicted_cnt"], "bikes")
