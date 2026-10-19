"""Unit tests for Sprint 2 models, baselines, weather client, and calendar utilities."""

from datetime import date, datetime
import unittest

import numpy as np
import pandas as pd

from src.calendar_utils import (
    derive_calendar_features,
    get_astronomical_season,
    is_us_dc_holiday,
)
from src.models import (
    FEATURE_COLUMNS,
    ModelManager,
    get_model_manager,
)
from src.weather import (
    WMO_CODE_MAP,
    map_wmo_code,
    normalize_weather_features,
)


class TestSprint2CalendarAndSeason(unittest.TestCase):
    """Verify calendar feature builder and astronomical season transitions."""

    def test_astronomical_season_boundaries(self):
        # Winter: Dec 21 to Mar 20
        self.assertEqual(get_astronomical_season(12, 21), 1)
        self.assertEqual(get_astronomical_season(1, 15), 1)
        self.assertEqual(get_astronomical_season(3, 20), 1)

        # Spring: Mar 21 to Jun 20
        self.assertEqual(get_astronomical_season(3, 21), 2)
        self.assertEqual(get_astronomical_season(5, 10), 2)
        self.assertEqual(get_astronomical_season(6, 20), 2)

        # Summer: Jun 21 to Sep 22
        self.assertEqual(get_astronomical_season(6, 21), 3)
        self.assertEqual(get_astronomical_season(7, 4), 3)
        self.assertEqual(get_astronomical_season(9, 22), 3)

        # Fall: Sep 23 to Dec 20
        self.assertEqual(get_astronomical_season(9, 23), 4)
        self.assertEqual(get_astronomical_season(10, 15), 4)
        self.assertEqual(get_astronomical_season(12, 20), 4)

    def test_derive_calendar_features_weekend(self):
        # 2026-10-18 was a Sunday -> UCI weekday = 0, workingday = 0
        dt = datetime(2026, 10, 18, 14, 0)
        cal = derive_calendar_features(dt)
        self.assertEqual(cal["weekday"], 0)
        self.assertEqual(cal["workingday"], 0)
        self.assertEqual(cal["hr"], 14)
        self.assertEqual(cal["season"], 4)

    def test_derive_calendar_features_workingday(self):
        # 2026-10-14 was a Wednesday -> UCI weekday = 3, workingday = 1
        dt = datetime(2026, 10, 14, 17, 0)
        cal = derive_calendar_features(dt)
        self.assertEqual(cal["weekday"], 3)
        self.assertEqual(cal["workingday"], 1)
        self.assertEqual(cal["hr"], 17)
        self.assertEqual(cal["holiday"], 0)


class TestSprint2WeatherNormalization(unittest.TestCase):
    """Verify weather normalization formulas adhering to SRS v1.5 Appendix B."""

    def test_temp_normalization(self):
        # tmin = -8°C -> 0.0; tmax = 39°C -> 1.0
        norm_min = normalize_weather_features(-8.0, 0.0, 50.0, 0.0)
        norm_max = normalize_weather_features(39.0, 0.0, 50.0, 0.0)
        self.assertAlmostEqual(norm_min["temp"], 0.0, delta=0.01)
        self.assertAlmostEqual(norm_max["temp"], 1.0, delta=0.01)

    def test_atemp_normalization(self):
        # amin = -16°C -> 0.0; amax = 50°C -> 1.0
        norm_min = normalize_weather_features(0.0, -16.0, 50.0, 0.0)
        norm_max = normalize_weather_features(0.0, 50.0, 50.0, 0.0)
        self.assertAlmostEqual(norm_min["atemp"], 0.0, delta=0.01)
        self.assertAlmostEqual(norm_max["atemp"], 1.0, delta=0.01)

    def test_windspeed_and_hum_normalization(self):
        norm = normalize_weather_features(20.0, 20.0, 75.0, 33.5)
        self.assertAlmostEqual(norm["hum"], 0.75, delta=0.01)
        self.assertAlmostEqual(norm["windspeed"], 0.50, delta=0.01)

    def test_wmo_code_mapping(self):
        # Code 0 (Clear) -> class 1
        cls_clear, desc_clear = map_wmo_code(0)
        self.assertEqual(cls_clear, 1)

        # Code 3 (Overcast) -> class 2
        cls_overcast, _ = map_wmo_code(3)
        self.assertEqual(cls_overcast, 2)

        # Code 61 (Slight rain) -> class 3
        cls_rain, _ = map_wmo_code(61)
        self.assertEqual(cls_rain, 3)

        # Code 65 (Heavy rain) -> class 4
        cls_heavy, _ = map_wmo_code(65)
        self.assertEqual(cls_heavy, 4)


class TestSprint2ModelManager(unittest.TestCase):
    """Verify chronological split, L0-L5 training, evaluation, and fallback rules."""

    @classmethod
    def setUpClass(cls):
        cls.mm = get_model_manager("dataset/hour.csv")

    def test_chronological_split_sizes(self):
        self.assertEqual(len(self.mm.train_df), 13903)
        self.assertEqual(len(self.mm.test_df), 3476)
        self.assertEqual(len(self.mm.train_df) + len(self.mm.test_df), 17379)

    def test_feature_columns_exclude_target_leakage(self):
        self.assertNotIn("casual", FEATURE_COLUMNS)
        self.assertNotIn("registered", FEATURE_COLUMNS)
        self.assertNotIn("cnt", FEATURE_COLUMNS)
        self.assertNotIn("yr", FEATURE_COLUMNS)

    def test_mae_scores_calculated(self):
        scores = self.mm.mae_scores
        # L0 baseline MAE around 175
        self.assertAlmostEqual(scores["L0"], 174.98, delta=1.0)
        # L1 hour mean MAE around 114
        self.assertAlmostEqual(scores["L1"], 113.73, delta=1.0)
        # L2 hour+workingday beats L1
        self.assertLess(scores["L2"], scores["L1"])
        # L5 Random Forest has the lowest MAE
        self.assertLess(scores["L5"], 90.0)

    def test_deterministic_fallback_l3_to_l2(self):
        # Create an unseen group in L3 to verify deterministic fallback
        input_data = {
            "season": 3,
            "mnth": 7,
            "hr": 17,
            "holiday": 0,
            "weekday": 2,
            "workingday": 1,
            "weathersit": 1,
            "temp": 0.5,
            "atemp": 0.5,
            "hum": 0.5,
            "windspeed": 0.2,
        }
        res = self.mm.predict("L3", input_data)
        self.assertGreater(res["predicted_cnt"], 0)
        self.assertIn(res["used_level"], ["L3", "L2", "L1", "L0"])

    def test_l4_and_l5_non_negative_predictions(self):
        input_data = {
            "season": 1,
            "mnth": 1,
            "hr": 3,
            "holiday": 0,
            "weekday": 1,
            "workingday": 1,
            "weathersit": 4,
            "temp": 0.0,
            "atemp": 0.0,
            "hum": 1.0,
            "windspeed": 0.8,
        }
        pred_l4 = self.mm.predict("L4", input_data)
        pred_l5 = self.mm.predict("L5", input_data)
        self.assertGreaterEqual(pred_l4["predicted_cnt"], 0)
        self.assertGreaterEqual(pred_l5["predicted_cnt"], 0)


if __name__ == "__main__":
    unittest.main()
