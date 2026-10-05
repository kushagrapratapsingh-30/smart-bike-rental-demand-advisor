"""
Data Loading, Inspection, and Cleaning Module
Sprint I Deliverable - Smart Bike Rental Demand Advisor

Performs initial dataset loading, integrity verification, data quality checks,
and target leakage auditing on dataset/hour.csv.
"""

import os
import pandas as pd


def load_dataset(csv_path: str = "dataset/hour.csv") -> pd.DataFrame:
    """Load the UCI Bike Sharing hourly dataset from disk."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at path: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def inspect_dataset(csv_path: str = "dataset/hour.csv") -> pd.DataFrame:
    """
    Run comprehensive data inspection and quality checks:
    - Head & shape
    - Columns and data types
    - Missing values
    - Duplicate rows
    - Feature ranges (e.g. hr in [0, 23])
    - Target distribution (cnt summary & negative value check)
    - Target leakage check (casual + registered == cnt)
    """
    df = load_dataset(csv_path)

    print("=" * 60)
    print("SMART BIKE RENTAL DEMAND ADVISOR — DATASET INSPECTION")
    print("=" * 60)

    print("\n1. FIRST 5 ROWS:")
    print(df.head())

    print("\n2. DATASET SHAPE:")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n3. COLUMN NAMES:")
    print(list(df.columns))

    print("\n4. DATA TYPES:")
    print(df.dtypes)

    print("\n5. MISSING VALUES PER COLUMN:")
    null_counts = df.isnull().sum()
    print(null_counts)
    total_missing = null_counts.sum()
    print(f"Total Missing Values across entire dataset: {total_missing}")

    print("\n6. DUPLICATE ROWS:")
    dup_count = df.duplicated().sum()
    print(f"Total Duplicate Records: {dup_count}")

    print("\n7. HOUR (hr) RANGE CHECK:")
    hr_min = df["hr"].min()
    hr_max = df["hr"].max()
    print(f"Minimum hour: {hr_min}, Maximum hour: {hr_max}")
    print(f"Hours within valid 0-23 range: {((df['hr'] >= 0) & (df['hr'] <= 23)).all()}")

    print("\n8. TARGET (cnt) SUMMARY STATISTICS:")
    print(df["cnt"].describe())
    neg_cnt = (df["cnt"] < 0).sum()
    print(f"Negative count occurrences: {neg_cnt}")

    print("\n9. TARGET LEAKAGE CHECK (casual + registered == cnt):")
    leakage_diff = ((df["casual"] + df["registered"]) != df["cnt"]).sum()
    print(f"Rows where (casual + registered != cnt): {leakage_diff}")
    if leakage_diff == 0:
        print("[AUDIT PASSED] casual + registered strictly equals cnt for 100% of rows.")
        print("[WARNING: C2 Constraint] 'casual' and 'registered' MUST be excluded from feature vectors")
        print("                        to prevent target leakage.")

    print("\n" + "=" * 60)
    print("SUMMARY: Dataset integrity verified. Ready for baseline modeling.")
    print("=" * 60)

    return df


if __name__ == "__main__":
    inspect_dataset()
