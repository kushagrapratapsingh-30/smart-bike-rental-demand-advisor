"""Model management, training, baselines, and evaluation for L0-L5."""

import os
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from src.data_cleaning import load_dataset

FEATURE_COLUMNS = [
    "season",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]

DEFAULT_BUNDLE_PATH = "models/model_bundle.joblib"


class ModelManager:
    """Manages L0-L5 baselines and trained models evaluated on chronological 80/20 split."""

    def __init__(self, dataset_path: str = "dataset/hour.csv", bundle_path: str = DEFAULT_BUNDLE_PATH):
        self.dataset_path = dataset_path
        self.bundle_path = bundle_path
        self.df = load_dataset(dataset_path)
        self.n_total = len(self.df)
        self.n_train = int(np.floor(self.n_total * 0.80))
        self.train_df = self.df.iloc[: self.n_train].copy()
        self.test_df = self.df.iloc[self.n_train :].copy()

        self.y_train = self.train_df["cnt"].values
        self.y_test = self.test_df["cnt"].values

        self.X_train = self.train_df[FEATURE_COLUMNS]
        self.X_test = self.test_df[FEATURE_COLUMNS]

        self.is_artifact_loaded = False

        if os.path.exists(self.bundle_path):
            self._load_bundle()
        else:
            self._train_and_save()

        # Compute predictions & MAE scores if not loaded
        if not hasattr(self, "predictions"):
            self._evaluate()

    def _load_bundle(self) -> None:
        """Load pre-trained artifacts from bundle."""
        try:
            bundle = joblib.load(self.bundle_path)
            self.l0_mean = float(bundle["l0_mean"])
            self.l1_map = bundle["l1_map"]
            self.l2_map = bundle["l2_map"]
            self.l3_map = bundle["l3_map"]
            self.lr_model = bundle["lr_model"]
            self.rf_model = bundle["rf_model"]
            self.mae_scores = bundle["mae_scores"]
            self.improvements = bundle["improvements"]
            self.is_artifact_loaded = True
        except Exception:
            self._train_and_save()

    def _train_and_save(self) -> None:
        """Fit baselines and models, then persist bundle."""
        # L0: Overall train mean
        self.l0_mean = float(self.y_train.mean())

        # L1: Group by hr
        self.l1_map = self.train_df.groupby("hr")["cnt"].mean().to_dict()

        # L2: Group by (hr, workingday)
        self.l2_map = self.train_df.groupby(["hr", "workingday"])["cnt"].mean().to_dict()

        # L3: Group by (hr, workingday, season)
        self.l3_map = self.train_df.groupby(["hr", "workingday", "season"])["cnt"].mean().to_dict()

        self.lr_model = LinearRegression()
        self.lr_model.fit(self.X_train, self.y_train)

        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.rf_model.fit(self.X_train, self.y_train)

        self._evaluate()

        try:
            os.makedirs(os.path.dirname(self.bundle_path), exist_ok=True)
            bundle = {
                "artifact_version": "1.5.0",
                "srs_version": "v1.5",
                "feature_names": FEATURE_COLUMNS,
                "n_train": self.n_train,
                "n_test": len(self.test_df),
                "l0_mean": self.l0_mean,
                "l1_map": self.l1_map,
                "l2_map": self.l2_map,
                "l3_map": self.l3_map,
                "lr_model": self.lr_model,
                "rf_model": self.rf_model,
                "mae_scores": self.mae_scores,
                "improvements": self.improvements,
            }
            joblib.dump(bundle, self.bundle_path, compress=3)
            self.is_artifact_loaded = True
        except Exception:
            pass

    def _evaluate(self) -> None:
        """Evaluate L0-L5 on identical held-out chronological test rows."""
        # L0 test preds
        p_l0 = np.full(len(self.y_test), self.l0_mean)
        mae_l0 = float(mean_absolute_error(self.y_test, p_l0))

        # L1 test preds
        p_l1 = np.array([self.l1_map.get(h, self.l0_mean) for h in self.test_df["hr"]])
        mae_l1 = float(mean_absolute_error(self.y_test, p_l1))

        # L2 test preds
        p_l2 = []
        for _, r in self.test_df.iterrows():
            k2 = (int(r["hr"]), int(r["workingday"]))
            if k2 in self.l2_map:
                p_l2.append(self.l2_map[k2])
            else:
                p_l2.append(self.l1_map.get(int(r["hr"]), self.l0_mean))
        p_l2 = np.array(p_l2)
        mae_l2 = float(mean_absolute_error(self.y_test, p_l2))

        # L3 test preds
        p_l3 = []
        for _, r in self.test_df.iterrows():
            k3 = (int(r["hr"]), int(r["workingday"]), int(r["season"]))
            k2 = (int(r["hr"]), int(r["workingday"]))
            if k3 in self.l3_map:
                p_l3.append(self.l3_map[k3])
            elif k2 in self.l2_map:
                p_l3.append(self.l2_map[k2])
            else:
                p_l3.append(self.l1_map.get(int(r["hr"]), self.l0_mean))
        p_l3 = np.array(p_l3)
        mae_l3 = float(mean_absolute_error(self.y_test, p_l3))

        # L4 test preds (clipped >= 0)
        p_l4 = np.clip(self.lr_model.predict(self.X_test), 0, None)
        mae_l4 = float(mean_absolute_error(self.y_test, p_l4))

        # L5 test preds (clipped >= 0)
        p_l5 = np.clip(self.rf_model.predict(self.X_test), 0, None)
        mae_l5 = float(mean_absolute_error(self.y_test, p_l5))

        self.predictions = {
            "L0": p_l0,
            "L1": p_l1,
            "L2": p_l2,
            "L3": p_l3,
            "L4": p_l4,
            "L5": p_l5,
        }

        self.mae_scores = {
            "L0": round(mae_l0, 2),
            "L1": round(mae_l1, 2),
            "L2": round(mae_l2, 2),
            "L3": round(mae_l3, 2),
            "L4": round(mae_l4, 2),
            "L5": round(mae_l5, 2),
        }

        self.improvements = {}
        for lvl, score in self.mae_scores.items():
            imp = ((self.mae_scores["L1"] - score) / self.mae_scores["L1"]) * 100
            self.improvements[lvl] = round(imp, 2)

    def predict(self, level: str, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prediction for a single validated input vector."""
        hr = int(input_dict["hr"])
        workingday = int(input_dict["workingday"])
        season = int(input_dict["season"])

        method_level = level.upper()
        used_level = method_level
        fallback_note = ""

        if method_level == "L0":
            pred_val = self.l0_mean
        elif method_level == "L1":
            pred_val = self.l1_map.get(hr, self.l0_mean)
            if hr not in self.l1_map:
                used_level = "L0"
                fallback_note = "Unseen hour; fell back to L0 global mean."
        elif method_level == "L2":
            k2 = (hr, workingday)
            if k2 in self.l2_map:
                pred_val = self.l2_map[k2]
            elif hr in self.l1_map:
                pred_val = self.l1_map[hr]
                used_level = "L1"
                fallback_note = "Unseen (hour, workingday); fell back to L1 hour mean."
            else:
                pred_val = self.l0_mean
                used_level = "L0"
                fallback_note = "Unseen hour; fell back to L0 global mean."
        elif method_level == "L3":
            k3 = (hr, workingday, season)
            k2 = (hr, workingday)
            if k3 in self.l3_map:
                pred_val = self.l3_map[k3]
            elif k2 in self.l2_map:
                pred_val = self.l2_map[k2]
                used_level = "L2"
                fallback_note = "Unseen (hour, workingday, season); fell back to L2."
            elif hr in self.l1_map:
                pred_val = self.l1_map[hr]
                used_level = "L1"
                fallback_note = "Unseen group; fell back to L1."
            else:
                pred_val = self.l0_mean
                used_level = "L0"
                fallback_note = "Unseen group; fell back to L0."
        elif method_level == "L4":
            row_df = pd.DataFrame([[input_dict[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
            raw_pred = float(self.lr_model.predict(row_df)[0])
            pred_val = max(0.0, raw_pred)
        elif method_level == "L5":
            row_df = pd.DataFrame([[input_dict[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
            raw_pred = float(self.rf_model.predict(row_df)[0])
            pred_val = max(0.0, raw_pred)
        else:
            pred_val = self.l0_mean
            used_level = "L0"
            fallback_note = f"Unknown model level '{level}'; used L0 fallback."

        return {
            "predicted_cnt": int(round(pred_val)),
            "predicted_cnt_raw": round(pred_val, 2),
            "requested_level": method_level,
            "used_level": used_level,
            "mae": self.mae_scores.get(used_level, self.mae_scores["L0"]),
            "fallback_note": fallback_note,
        }

    def get_summary_table(self) -> List[Dict[str, Any]]:
        """Return table data for Model Performance page comparison."""
        definitions = {
            "L0": ("Overall Mean Baseline", "Historical training mean", False),
            "L1": ("Hour Mean Baseline", "Mean grouped by hour of day", False),
            "L2": ("Hour + Working Day Baseline", "Mean grouped by hour and working day", False),
            "L3": ("Hour + Working Day + Season", "Mean grouped by hour, working day, season", False),
            "L4": ("Multiple Linear Regression", "Trained parametric regression on candidate features", True),
            "L5": ("Random Forest Regressor", "Non-linear ensemble (100 trees, random_state=42)", True),
        }
        rows = []
        for lvl in ["L0", "L1", "L2", "L3", "L4", "L5"]:
            name, desc, is_ml = definitions[lvl]
            rows.append({
                "Level": lvl,
                "Model / Approach": name,
                "Description": desc,
                "Is_ML": is_ml,
                "Test MAE": self.mae_scores[lvl],
                "Improvement vs L1 (%)": self.improvements[lvl],
            })
        return rows


_model_manager_instance: Optional[ModelManager] = None


def get_model_manager(dataset_path: str = "dataset/hour.csv") -> ModelManager:
    """Get or create singleton ModelManager instance."""
    global _model_manager_instance
    if _model_manager_instance is None:
        _model_manager_instance = ModelManager(dataset_path)
    return _model_manager_instance
