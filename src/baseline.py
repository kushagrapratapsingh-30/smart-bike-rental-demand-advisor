"""
Baseline Modeling Module
Sprint I Deliverable - Smart Bike Rental Demand Advisor

Calculates and serves the Level 0 Baseline:
- Predicts the global training set mean demand regardless of input conditions.
- Computes baseline Mean Absolute Error (MAE) for benchmark comparisons in Sprint II.
- Serves as the fallback mechanism when pre-trained ML artifacts are unavailable (FR-2.4).
"""

import os
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class Level0Baseline:
    """
    Level 0 Naive Baseline:
    Always returns the overall historical mean demand (cnt).
    """

    def __init__(self, dataset_path: str = "dataset/hour.csv"):
        self.dataset_path = dataset_path
        self.mean_demand: float = 0.0
        self.rounded_demand: int = 0
        self.mae: float = 0.0
        self.sample_size: int = 0
        self._fit()

    def _fit(self) -> None:
        """Load the historical dataset and compute baseline statistics."""
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
        # Compute baseline MAE: mean(|cnt - mean_demand|)
        self.mae = float(np.mean(np.abs(df["cnt"] - self.mean_demand)))

    def predict(self, input_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Produce Level 0 baseline prediction.
        Input features are deliberately ignored by design.
        """
        return {
            "predicted_cnt": self.rounded_demand,
            "predicted_cnt_raw": self.mean_demand,
            "mae": round(self.mae, 2),
            "baseline_type": "Level 0 Baseline (Global Training Set Mean)",
            "is_fallback": True,
            "fallback_warning": "WARN_SYS_01: Pre-trained model artifact missing. Loading Level 0 baseline.",
            "note": (
                "This is a naive baseline estimate ignoring operational conditions. "
                "Trained ML regression models (Level 1, Random Forest) will be integrated in Sprint II."
            ),
        }


# Singleton instance for quick access
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
    print("=" * 60)
    print("LEVEL 0 BASELINE CALCULATION")
    print("=" * 60)
    print(f"Dataset Records: {baseline.sample_size}")
    print(f"Global Mean Demand (cnt): {baseline.mean_demand:.2f} bikes")
    print(f"Rounded Baseline Demand: {baseline.rounded_demand} bikes")
    print(f"Baseline MAE: {baseline.mae:.2f} bikes")
    print("\nSample Prediction Output (inputs ignored):")
    sample_res = baseline.predict({"hour": 17, "season": "Summer", "weather": "Clear", "working_day": "Yes"})
    for k, v in sample_res.items():
        print(f"  {k}: {v}")
    print("=" * 60)
