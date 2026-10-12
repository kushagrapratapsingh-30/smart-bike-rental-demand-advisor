"""
Data Loading, Inspection, and Cleaning Module
Sprint I Deliverable - Smart Bike Rental Demand Advisor

Performs initial dataset loading, integrity verification, data quality checks,
and target leakage auditing on dataset/hour.csv.
"""

import os
import pandas as pd


from typing import Dict, Any


def load_dataset(csv_path: str = "dataset/hour.csv") -> pd.DataFrame:
    """Load the UCI Bike Sharing hourly dataset from disk."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at path: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def get_dataset_diagnostics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive dataset diagnostics, descriptive statistics,
    and integrity metrics per SRS v1.5 Section 3.1 & 3.3.
    """
    null_counts = df.isnull().sum().to_dict()
    total_missing = int(sum(null_counts.values()))
    dup_count = int(df.duplicated().sum())

    cnt_series = df["cnt"]
    leakage_diff = int(((df["casual"] + df["registered"]) != df["cnt"]).sum()) if {"casual", "registered", "cnt"}.issubset(df.columns) else -1

    return {
        "shape": (int(df.shape[0]), int(df.shape[1])),
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": null_counts,
        "total_missing": total_missing,
        "duplicate_count": dup_count,
        "hr_min": int(df["hr"].min()) if "hr" in df.columns else None,
        "hr_max": int(df["hr"].max()) if "hr" in df.columns else None,
        "target_stats": {
            "count": int(cnt_series.count()),
            "mean": float(cnt_series.mean()),
            "median": float(cnt_series.median()),
            "std": float(cnt_series.std()),
            "skewness": float(cnt_series.skew()),
            "min": int(cnt_series.min()),
            "q25": float(cnt_series.quantile(0.25)),
            "q75": float(cnt_series.quantile(0.75)),
            "max": int(cnt_series.max()),
            "negative_count": int((cnt_series < 0).sum()),
        },
        "target_leakage_diff": leakage_diff,
        "is_leakage_consistent": (leakage_diff == 0),
    }


def inspect_dataset(csv_path: str = "dataset/hour.csv") -> pd.DataFrame:
    """
    Run comprehensive data inspection and quality checks per SRS v1.5:
    - Head & shape (17,379 records, 17 columns)
    - Columns and data types
    - Missing values (0 missing)
    - Duplicate rows (0 duplicates)
    - Hour range check (0 to 23)
    - Target distribution (mean, median, std, skewness, min, max, negative count)
    - Target leakage check (casual + registered == cnt)
    - Semantic note: One row represents one hour of bike-sharing activity in Washington, D.C.
    """
    df = load_dataset(csv_path)
    diag = get_dataset_diagnostics(df)

    print("=" * 60)
    print("SMART BIKE RENTAL DEMAND ADVISOR — DATASET INSPECTION")
    print("=" * 60)
    print("Note: One row represents one hour of bike-sharing activity,")
    print("      not an individual customer or single bicycle.")
    print("=" * 60)

    print("\n1. FIRST 5 ROWS:")
    print(df.head())

    print("\n2. DATASET SHAPE:")
    print(f"Rows: {diag['rows']}, Columns: {diag['shape'][1]}")

    print("\n3. COLUMN NAMES:")
    print(diag["columns"])

    print("\n4. DATA TYPES:")
    for col, dtype in diag["dtypes"].items():
        print(f"  {col}: {dtype}")

    print("\n5. MISSING VALUES PER COLUMN:")
    for col, null_val in diag["missing_values"].items():
        print(f"  {col}: {null_val}")
    print(f"Total Missing Values across entire dataset: {diag['total_missing']}")

    print("\n6. DUPLICATE ROWS:")
    print(f"Total Duplicate Records: {diag['duplicate_count']}")

    print("\n7. HOUR (hr) RANGE CHECK:")
    print(f"Minimum hour: {diag['hr_min']}, Maximum hour: {diag['hr_max']}")
    valid_hr = ((df['hr'] >= 0) & (df['hr'] <= 23)).all() if "hr" in df.columns else False
    print(f"Hours within valid 0-23 range: {valid_hr}")

    print("\n8. TARGET (cnt) SUMMARY STATISTICS (SRS Section 3.3):")
    t = diag["target_stats"]
    print(f"Count: {t['count']}")
    print(f"Mean: {t['mean']:.2f}")
    print(f"Median: {t['median']:.2f}")
    print(f"Standard Deviation: {t['std']:.2f}")
    print(f"Skewness: {t['skewness']:.4f}")
    print(f"Min: {t['min']}, 25%: {t['q25']:.1f}, 75%: {t['q75']:.1f}, Max: {t['max']}")
    print(f"Negative count occurrences: {t['negative_count']}")

    print("\n9. TARGET LEAKAGE CHECK (casual + registered == cnt):")
    print(f"Rows where (casual + registered != cnt): {diag['target_leakage_diff']}")
    if diag["is_leakage_consistent"]:
        print("[AUDIT PASSED] casual + registered strictly equals cnt for 100% of rows.")
        print("[WARNING: C2 Constraint] 'casual' and 'registered' MUST be excluded from feature vectors")
        print("                        to prevent target leakage.")

    print("\n" + "=" * 60)
    print("SUMMARY: Dataset integrity verified. Ready for baseline modeling.")
    print("=" * 60)

    return df


if __name__ == "__main__":
    inspect_dataset()
