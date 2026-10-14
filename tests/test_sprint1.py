"""Sprint 1 automated test suite for data cleaning, validation rules, baseline, and EDA."""

import os
import unittest

import numpy as np
import pandas as pd

from src.baseline import Level0Baseline, get_baseline_instance, predict_demand_fallback
from src.data_cleaning import get_dataset_diagnostics, inspect_dataset, load_dataset
from src.eda import (
    compute_target_distribution,
    compute_hourly_patterns,
    compute_seasonal_patterns,
    compute_weather_patterns,
    compute_feature_correlations,
    generate_eda_summary,
)
from src.validation import (
    VALID_SEASONS,
    VALID_WEATHER,
    VALID_WORKING_DAY,
    REQUIRED_DATASET_COLUMNS,
    validate_hour,
    validate_season,
    validate_weather,
    validate_working_day,
    validate_inputs,
    sanitize_inputs,
    validate_dataset_schema,
    validate_dataset_record,
)


class TestSprint1DataAndCleaning(unittest.TestCase):
    """Verify dataset loading, target statistics, and leakage constraints."""

    @classmethod
    def setUpClass(cls):
        cls.csv_path = "dataset/hour.csv"
        cls.df = load_dataset(cls.csv_path)

    def test_dataset_exists_and_loads(self):
        self.assertTrue(os.path.exists(self.csv_path))
        self.assertEqual(self.df.shape, (17379, 17))

    def test_no_missing_values(self):
        self.assertEqual(self.df.isnull().sum().sum(), 0)

    def test_no_duplicates(self):
        self.assertEqual(self.df.duplicated().sum(), 0)

    def test_hour_bounds_in_dataset(self):
        self.assertEqual(self.df["hr"].min(), 0)
        self.assertEqual(self.df["hr"].max(), 23)

    def test_target_non_negative(self):
        self.assertTrue((self.df["cnt"] >= 0).all())
        self.assertGreaterEqual(self.df["cnt"].min(), 1)

    def test_target_leakage_constraint_c2(self):
        """Verify casual + registered == cnt to prove target leakage if included."""
        mismatches = ((self.df["casual"] + self.df["registered"]) != self.df["cnt"]).sum()
        self.assertEqual(mismatches, 0, "casual + registered must strictly equal cnt for 100% of rows")

    def test_target_descriptive_statistics(self):
        """Verify descriptive statistics and distribution per SRS Section 3.3."""
        diag = get_dataset_diagnostics(self.df)
        stats = diag["target_stats"]
        self.assertEqual(stats["count"], 17379)
        self.assertAlmostEqual(stats["mean"], 189.46, delta=0.1)
        self.assertEqual(stats["median"], 142.0)
        self.assertAlmostEqual(stats["std"], 181.39, delta=0.1)
        self.assertAlmostEqual(stats["skewness"], 1.28, delta=0.05)
        self.assertEqual(stats["min"], 1)
        self.assertEqual(stats["max"], 977)
        self.assertEqual(stats["negative_count"], 0)

    def test_season_coding_and_boundaries(self):
        """Verify season coding in hour.csv adheres to SRS v1.5 (1=Winter, 2=Spring, 3=Summer, 4=Fall)."""
        # Season 1 is present in Jan, Feb, Dec (Winter)
        s1_months = set(self.df[self.df["season"] == 1]["mnth"].unique())
        self.assertTrue({1, 2, 12}.issubset(s1_months))

        # 2011 transitions around astronomical solstices and equinoxes (SRS v1.5 Sec 4.4)
        s2_first = self.df[self.df["season"] == 2].iloc[0]
        self.assertEqual(s2_first["dteday"], "2011-03-21")  # Spring equinox

        s3_first = self.df[self.df["season"] == 3].iloc[0]
        self.assertEqual(s3_first["dteday"], "2011-06-21")  # Summer solstice

        s4_first = self.df[self.df["season"] == 4].iloc[0]
        self.assertEqual(s4_first["dteday"], "2011-09-23")  # Autumnal equinox


class TestSprint1Baseline(unittest.TestCase):
    """Verify Level 0 baseline calculation, MAE, and fallback behavior."""

    @classmethod
    def setUpClass(cls):
        cls.baseline = get_baseline_instance()

    def test_level_0_mean_value(self):
        self.assertAlmostEqual(self.baseline.mean_demand, 189.46, delta=0.5)
        self.assertEqual(self.baseline.rounded_demand, 189)

    def test_baseline_mae(self):
        self.assertAlmostEqual(self.baseline.mae, 142.4, delta=1.0)

    def test_tc_01_core_prediction(self):
        """TC-01: Valid inputs -> Level 0 baseline prediction with WARN_SYS_01."""
        inputs = {"hour": 17, "season": "Summer", "weather": "Clear", "working_day": "Yes"}
        result = self.baseline.predict(inputs)

        self.assertGreaterEqual(result["predicted_cnt"], 0)
        self.assertEqual(result["predicted_cnt"], 189)
        self.assertTrue(result["is_fallback"])
        self.assertIn("WARN_SYS_01", result["fallback_warning"])

    def test_tc_03_invariance_check(self):
        """TC-03: Level 0 ignores inputs, returning identical mean for all queries."""
        inputs_1 = {"hour": 17, "season": "Summer", "weather": "Clear", "working_day": "Yes"}
        inputs_2 = {"hour": 8, "season": "Winter", "weather": "Mist/Cloudy", "working_day": "Yes"}
        inputs_3 = {"hour": 23, "season": "Fall", "weather": "Light Rain/Snow", "working_day": "No"}

        res1 = self.baseline.predict(inputs_1)
        res2 = self.baseline.predict(inputs_2)
        res3 = self.baseline.predict(inputs_3)

        self.assertEqual(res1["predicted_cnt"], res2["predicted_cnt"])
        self.assertEqual(res2["predicted_cnt"], res3["predicted_cnt"])
        self.assertEqual(res1["predicted_cnt"], 189)


class TestSprint1Validation(unittest.TestCase):
    """Sprint 1B: Verifies single input boundary checks and error messages (FR-1.2, NFR-2)."""

    def test_valid_hour(self):
        ok, msg = validate_hour(17)
        self.assertTrue(ok)
        self.assertEqual(msg, "Valid hour.")

        ok0, _ = validate_hour(0)
        ok23, _ = validate_hour(23)
        self.assertTrue(ok0)
        self.assertTrue(ok23)

    def test_tc_02_invalid_hour_out_of_bounds(self):
        """TC-02: Hour 25 (out of bounds) halts execution with ERR_VAL_01."""
        ok_high, msg_high = validate_hour(25)
        self.assertFalse(ok_high)
        self.assertIn("ERR_VAL_01", msg_high)

        ok_low, msg_low = validate_hour(-1)
        self.assertFalse(ok_low)
        self.assertIn("ERR_VAL_01", msg_low)

        ok_float, msg_float = validate_hour(12.5)
        self.assertFalse(ok_float)
        self.assertIn("ERR_VAL_01", msg_float)

        ok_bool, msg_bool = validate_hour(True)
        self.assertFalse(ok_bool)
        self.assertIn("ERR_VAL_01", msg_bool)

    def test_season_mapping_srs_v15(self):
        """Verify season mapping adheres to SRS v1.5: 1=Winter, 2=Spring, 3=Summer, 4=Fall."""
        self.assertEqual(VALID_SEASONS["Winter"], 1)
        self.assertEqual(VALID_SEASONS["Spring"], 2)
        self.assertEqual(VALID_SEASONS["Summer"], 3)
        self.assertEqual(VALID_SEASONS["Fall"], 4)

    def test_invalid_season(self):
        ok, msg = validate_season("InvalidSeason")
        self.assertFalse(ok)
        self.assertIn("ERR_VAL_SEASON", msg)

    def test_invalid_weather(self):
        ok, msg = validate_weather("Tornado")
        self.assertFalse(ok)
        self.assertIn("ERR_VAL_WEATHER", msg)

    def test_invalid_working_day(self):
        ok, msg = validate_working_day("Maybe")
        self.assertFalse(ok)
        self.assertIn("ERR_VAL_WORKINGDAY", msg)

    def test_aggregate_validation_and_sanitization(self):
        ok, msg = validate_inputs(17, "Summer", "Clear", "Yes")
        self.assertTrue(ok)
        sanitized = sanitize_inputs(17, "Summer", "Clear", "Yes")
        self.assertEqual(sanitized["hour"], 17)
        self.assertEqual(sanitized["season_code"], 3)  # Summer is 3 per SRS v1.5
        self.assertEqual(sanitized["weather_code"], 1)
        self.assertEqual(sanitized["working_day_flag"], 1)

        sanitized_winter = sanitize_inputs(8, "Winter", "Mist/Cloudy", "No")
        self.assertEqual(sanitized_winter["season_code"], 1)  # Winter is 1 per SRS v1.5

        ok_fail, msg_fail = validate_inputs(25, "Summer", "Clear", "Yes")
        self.assertFalse(ok_fail)
        self.assertIn("ERR_VAL_01", msg_fail)


class TestSprint1DatasetSchemaValidation(unittest.TestCase):
    """Verify DataFrame schema rules and record-level validation."""

    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset("dataset/hour.csv")

    def test_valid_dataset_passes_schema_validation(self):
        """Verify the actual hour.csv passes all schema checks without error."""
        is_valid, errors = validate_dataset_schema(self.df)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_missing_required_column_detected(self):
        """Missing required column is detected and reported."""
        corrupt_df = self.df.drop(columns=["cnt"])
        is_valid, errors = validate_dataset_schema(corrupt_df)
        self.assertFalse(is_valid)
        self.assertTrue(any("Missing required columns" in e for e in errors))

    def test_missing_values_detected(self):
        """Missing / null values in dataset are detected and reported."""
        corrupt_df = self.df.head(50).copy()
        corrupt_df.loc[5, "temp"] = np.nan
        is_valid, errors = validate_dataset_schema(corrupt_df)
        self.assertFalse(is_valid)
        self.assertTrue(any("Missing/null values detected" in e for e in errors))

    def test_negative_rental_counts_detected(self):
        """Negative rental counts in target or components are rejected."""
        corrupt_df = self.df.head(50).copy()
        corrupt_df.loc[10, "cnt"] = -5
        is_valid, errors = validate_dataset_schema(corrupt_df)
        self.assertFalse(is_valid)
        self.assertTrue(any("negative rental counts" in e for e in errors))

    def test_invalid_hour_values_detected(self):
        """Hour values outside 0-23 are detected."""
        corrupt_df = self.df.head(50).copy()
        corrupt_df.loc[2, "hr"] = 28
        is_valid, errors = validate_dataset_schema(corrupt_df)
        self.assertFalse(is_valid)
        self.assertTrue(any("outside valid range 0–23" in e for e in errors))

    def test_invalid_season_values_detected(self):
        """Season values outside 1-4 domain are detected."""
        corrupt_df = self.df.head(50).copy()
        corrupt_df.loc[3, "season"] = 9
        is_valid, errors = validate_dataset_schema(corrupt_df)
        self.assertFalse(is_valid)
        self.assertTrue(any("outside valid domain 1–4" in e for e in errors))

    def test_invalid_weather_values_detected(self):
        """Weather values outside 1-4 domain are detected."""
        corrupt_df = self.df.head(50).copy()
        corrupt_df.loc[4, "weathersit"] = 0
        is_valid, errors = validate_dataset_schema(corrupt_df)
        self.assertFalse(is_valid)
        self.assertTrue(any("outside valid domain 1–4" in e for e in errors))

    def test_target_consistency_violation_detected(self):
        """Target consistency failure (casual + registered != cnt) is detected."""
        corrupt_df = self.df.head(50).copy()
        corrupt_df.loc[0, "casual"] = 999  # breaks casual + registered == cnt
        is_valid, errors = validate_dataset_schema(corrupt_df)
        self.assertFalse(is_valid)
        self.assertTrue(any("Target consistency failure" in e for e in errors))

    def test_record_level_validation(self):
        """Verify record-level validation on valid and invalid single records."""
        valid_rec = {
            "hr": 17,
            "season": 3,
            "weathersit": 1,
            "casual": 20,
            "registered": 80,
            "cnt": 100
        }
        ok_v, errs_v = validate_dataset_record(valid_rec)
        self.assertTrue(ok_v)
        self.assertEqual(len(errs_v), 0)

        invalid_rec = {
            "hr": 26,
            "season": 5,
            "weathersit": 9,
            "casual": 10,
            "registered": -5,
            "cnt": 50
        }
        ok_i, errs_i = validate_dataset_record(invalid_rec)
        self.assertFalse(ok_i)
        self.assertGreaterEqual(len(errs_i), 3)


class TestSprint1EDA(unittest.TestCase):
    """Sprint 1C: Verify exploratory data analysis computations and chart outputs."""

    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset("dataset/hour.csv")

    def test_target_distribution_and_outliers(self):
        dist = compute_target_distribution(self.df)
        self.assertAlmostEqual(dist["mean"], 189.46, delta=0.1)
        self.assertEqual(dist["median"], 142.0)
        self.assertAlmostEqual(dist["skewness"], 1.28, delta=0.05)
        self.assertEqual(dist["q1"], 40.0)
        self.assertEqual(dist["q3"], 281.0)
        self.assertEqual(dist["iqr"], 241.0)
        self.assertEqual(dist["upper_fence"], 642.5)
        self.assertEqual(dist["outlier_count"], 505)
        self.assertAlmostEqual(dist["outlier_pct"], 2.91, delta=0.05)

    def test_hourly_patterns_and_peaks(self):
        hourly = compute_hourly_patterns(self.df)
        top_peaks = [h[0] for h in hourly["top_peak_hours"]]
        self.assertEqual(top_peaks, [17, 18, 8])

        lowest_troughs = [h[0] for h in hourly["lowest_trough_hours"]]
        self.assertEqual(lowest_troughs, [5, 3, 4])

        # Working days have sharp morning peak at 8 AM
        self.assertGreater(hourly["workingday_hourly_mean"][8], 400.0)
        self.assertLess(hourly["nonworkingday_hourly_mean"][8], 150.0)

    def test_seasonal_demand_ordering(self):
        seasonal = compute_seasonal_patterns(self.df)
        metrics = {row["season"]: row["mean"] for row in seasonal["season_metrics"]}
        # Summer (3) is highest demand; Winter (1) is lowest per SRS v1.5
        self.assertGreater(metrics[3], metrics[4])
        self.assertGreater(metrics[4], metrics[1])
        self.assertAlmostEqual(metrics[1], 111.11, delta=0.5)

    def test_weather_demand_reduction(self):
        weather = compute_weather_patterns(self.df)
        metrics = {row["weathersit"]: row["mean"] for row in weather["weather_metrics"]}
        self.assertGreater(metrics[1], metrics[2])
        self.assertGreater(metrics[2], metrics[3])
        self.assertGreater(metrics[3], metrics[4])

    def test_feature_correlations(self):
        corrs = compute_feature_correlations(self.df)
        self.assertGreater(corrs["temp"], 0.35)
        self.assertLess(corrs["hum"], -0.25)
        self.assertGreater(corrs["hr"], 0.35)


if __name__ == "__main__":
    unittest.main()
