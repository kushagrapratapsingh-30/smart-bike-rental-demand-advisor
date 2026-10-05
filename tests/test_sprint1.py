"""
Sprint I Automated Test Suite
Smart Bike Rental Demand Advisor

Executes test cases mapped to SRS v1.1:
- TC-01: Valid inputs -> Level 0 baseline non-negative output (189 bikes)
- TC-02: Out-of-bounds input (Hour: 25) -> Validation halts with ERR_VAL_01
- TC-03: Invariance check across multiple valid inputs
- Data integrity & target leakage verification (C2)
- Fallback system notification (FR-2.4, WARN_SYS_01)
"""

import unittest
import os
import pandas as pd
from src.data_cleaning import load_dataset, inspect_dataset
from src.baseline import Level0Baseline, get_baseline_instance, predict_demand_fallback
from src.validation import (
    validate_hour,
    validate_season,
    validate_weather,
    validate_working_day,
    validate_inputs,
    sanitize_inputs,
)


class TestSprint1DataAndCleaning(unittest.TestCase):
    """Verifies dataset loading, integrity, and target leakage constraints."""

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

    def test_target_leakage_constraint_c2(self):
        """Verify casual + registered == cnt to prove target leakage if included."""
        mismatches = ((self.df["casual"] + self.df["registered"]) != self.df["cnt"]).sum()
        self.assertEqual(mismatches, 0, "casual + registered must equal cnt")


class TestSprint1Baseline(unittest.TestCase):
    """Verifies Level 0 baseline calculation, MAE, and fallback behavior."""

    @classmethod
    def setUpClass(cls):
        cls.baseline = get_baseline_instance()

    def test_level_0_mean_value(self):
        # Expected global mean ~189.46, rounded to 189
        self.assertAlmostEqual(self.baseline.mean_demand, 189.46, delta=0.5)
        self.assertEqual(self.baseline.rounded_demand, 189)

    def test_baseline_mae(self):
        # Expected MAE ~142.4 bikes
        self.assertAlmostEqual(self.baseline.mae, 142.4, delta=1.0)

    def test_tc_01_core_prediction(self):
        """TC-01: Valid inputs (Hour: 17, Season: Summer, Weather: Clear, WorkingDay: Yes)"""
        inputs = {"hour": 17, "season": "Summer", "weather": "Clear", "working_day": "Yes"}
        result = self.baseline.predict(inputs)

        self.assertGreaterEqual(result["predicted_cnt"], 0)
        self.assertEqual(result["predicted_cnt"], 189)
        self.assertTrue(result["is_fallback"])
        self.assertIn("WARN_SYS_01", result["fallback_warning"])

    def test_tc_03_invariance_check(self):
        """TC-03 & TC-04: Level 0 ignores inputs, returning the same mean for all queries."""
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
    """Verifies input boundary checks and error messages (FR-1.2, NFR-2)."""

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

    def test_invalid_season(self):
        ok, msg = validate_season("InvalidSeason")
        self.assertFalse(ok)
        self.assertIn("ERR_VAL_02", msg)

    def test_invalid_weather(self):
        ok, msg = validate_weather("Tornado")
        self.assertFalse(ok)
        self.assertIn("ERR_VAL_03", msg)

    def test_invalid_working_day(self):
        ok, msg = validate_working_day("Maybe")
        self.assertFalse(ok)
        self.assertIn("ERR_VAL_04", msg)

    def test_aggregate_validation_and_sanitization(self):
        # Valid
        ok, msg = validate_inputs(17, "Summer", "Clear", "Yes")
        self.assertTrue(ok)
        sanitized = sanitize_inputs(17, "Summer", "Clear", "Yes")
        self.assertEqual(sanitized["hour"], 17)
        self.assertEqual(sanitized["season_code"], 2)
        self.assertEqual(sanitized["weather_code"], 1)
        self.assertEqual(sanitized["working_day_flag"], 1)

        # Invalid hour halts aggregate validation
        ok_fail, msg_fail = validate_inputs(25, "Summer", "Clear", "Yes")
        self.assertFalse(ok_fail)
        self.assertIn("ERR_VAL_01", msg_fail)


if __name__ == "__main__":
    unittest.main()
