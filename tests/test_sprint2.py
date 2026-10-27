"""Unit tests for Sprint 2 models, baselines, weather client, and calendar utilities."""

from datetime import date, datetime
import json
import os
import tempfile
import unittest
import urllib.error
from unittest.mock import patch, MagicMock

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.calendar_utils import (
    derive_calendar_features,
    get_astronomical_season,
    is_us_dc_holiday,
)
from src.models import (
    FEATURE_COLUMNS,
    ModelManager,
    get_model_manager,
    RF_AVAILABLE,
    RF_IMPORT_ERROR,
)
from src.weather import (
    WMO_CODE_MAP,
    fetch_open_meteo_forecast,
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


class _MockHTTPResponse:
    """Mock HTTP response for urllib.request.urlopen."""

    def __init__(self, data: bytes, status: int = 200):
        self.data = data
        self.status = status

    def read(self):
        return self.data

    def decode(self, *args, **kwargs):
        return self.data.decode(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class TestSprint2WeatherAPIFailure(unittest.TestCase):
    """Testing Plan Step 2: Weather API Failure, Validation, and Recovery Tests per SRS v1.5."""

    def setUp(self):
        self.target_dt = datetime(2026, 10, 24, 14, 0)
        self.target_iso = self.target_dt.strftime("%Y-%m-%dT%H:00")
        self.valid_payload = {
            "hourly": {
                "time": [self.target_iso],
                "temperature_2m": [22.4],
                "apparent_temperature": [21.8],
                "relative_humidity_2m": [58.0],
                "wind_speed_10m": [14.5],
                "weather_code": [0],  # Clear sky -> class 1
            }
        }

    @patch("urllib.request.urlopen")
    def test_tc_w01_successful_weather_response(self, mock_urlopen):
        """TC-W01: Valid Open-Meteo response parsed into normalized and raw schema (FR-2.1, FR-2.4)."""
        payload_bytes = json.dumps(self.valid_payload).encode("utf-8")
        mock_urlopen.return_value = _MockHTTPResponse(payload_bytes, status=200)

        res = fetch_open_meteo_forecast(self.target_dt)
        self.assertTrue(res["success"])
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["weather_source"], "Open-Meteo Forecast API")
        self.assertEqual(res["timestamp"], self.target_iso)
        self.assertEqual(res["raw_weather"]["temperature_c"], 22.4)
        self.assertEqual(res["raw_weather"]["weathersit"], 1)
        self.assertAlmostEqual(res["normalized_weather"]["temp"], (22.4 - (-8)) / (39 - (-8)), delta=0.01)
        self.assertAlmostEqual(res["normalized_weather"]["hum"], 0.58, delta=0.01)

    @patch("urllib.request.urlopen")
    def test_tc_w02_connection_failure(self, mock_urlopen):
        """TC-W02: Network connection failure triggers weather_api_error and WARN_API_01 (FR-2.2, Section 5.5)."""
        mock_urlopen.side_effect = urllib.error.URLError("Failed to resolve host 'api.open-meteo.com'")

        res = fetch_open_meteo_forecast(self.target_dt)
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "weather_api_error")
        self.assertEqual(res["warning_code"], "WARN_API_01")
        self.assertIn("Live weather could not be retrieved", res["message"])

    @patch("urllib.request.urlopen")
    def test_tc_w03_request_timeout(self, mock_urlopen):
        """TC-W03: Timeout exceeding 2.0s triggers weather_api_error and WARN_API_01 (NFR-1, Appendix B)."""
        mock_urlopen.side_effect = TimeoutError("Request timed out after 2.0s")

        res = fetch_open_meteo_forecast(self.target_dt)
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "weather_api_error")
        self.assertEqual(res["warning_code"], "WARN_API_01")
        self.assertIn("Live weather could not be retrieved", res["message"])

    @patch("urllib.request.urlopen")
    def test_tc_w04_http_error(self, mock_urlopen):
        """TC-W04: HTTP error (503 / 500) triggers weather_api_error and WARN_API_01 (Section 5.5)."""
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.open-meteo.com", code=503, msg="Service Unavailable", hdrs={}, fp=None
        )

        res = fetch_open_meteo_forecast(self.target_dt)
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "weather_api_error")
        self.assertEqual(res["warning_code"], "WARN_API_01")

        # Also verify HTTP status != 200 via response object
        mock_urlopen.side_effect = None
        mock_urlopen.return_value = _MockHTTPResponse(b"Server Error", status=500)
        res_500 = fetch_open_meteo_forecast(self.target_dt)
        self.assertFalse(res_500["success"])
        self.assertEqual(res_500["status"], "weather_api_error")
        self.assertEqual(res_500["warning_code"], "WARN_API_01")

    @patch("urllib.request.urlopen")
    def test_tc_w05_invalid_response_malformed_json(self, mock_urlopen):
        """TC-W05a: Malformed JSON triggers weather_api_error and WARN_API_01 (Section 5.5)."""
        mock_urlopen.return_value = _MockHTTPResponse(b"<html>502 Bad Gateway</html>", status=200)

        res = fetch_open_meteo_forecast(self.target_dt)
        self.assertFalse(res["success"])
        self.assertEqual(res["warning_code"], "WARN_API_01")

    @patch("urllib.request.urlopen")
    def test_tc_w05_missing_required_fields_and_nulls(self, mock_urlopen):
        """TC-W05b: Missing required fields or null values trigger weather_validation_error (Section 5.5)."""
        # Missing temperature_2m
        bad_payload = {
            "hourly": {
                "time": [self.target_iso],
                "apparent_temperature": [20.0],
                "relative_humidity_2m": [50.0],
                "wind_speed_10m": [10.0],
                "weather_code": [0],
            }
        }
        mock_urlopen.return_value = _MockHTTPResponse(json.dumps(bad_payload).encode("utf-8"), status=200)
        res = fetch_open_meteo_forecast(self.target_dt)
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "weather_validation_error")
        self.assertEqual(res["warning_code"], "WARN_API_01")

        # Null value in array
        null_payload = {
            "hourly": {
                "time": [self.target_iso],
                "temperature_2m": [None],
                "apparent_temperature": [20.0],
                "relative_humidity_2m": [50.0],
                "wind_speed_10m": [10.0],
                "weather_code": [0],
            }
        }
        mock_urlopen.return_value = _MockHTTPResponse(json.dumps(null_payload).encode("utf-8"), status=200)
        res_null = fetch_open_meteo_forecast(self.target_dt)
        self.assertFalse(res_null["success"])
        self.assertEqual(res_null["status"], "weather_validation_error")
        self.assertEqual(res_null["warning_code"], "WARN_API_01")

    @patch("urllib.request.urlopen")
    def test_tc_w05_unknown_wmo_weather_code_warn_map_01(self, mock_urlopen):
        """TC-W05c & TC-11: Unknown WMO code rejects silent coercion with WARN_MAP_01 (Section 4.3, Appendix B)."""
        unknown_wmo_payload = {
            "hourly": {
                "time": [self.target_iso],
                "temperature_2m": [20.0],
                "apparent_temperature": [20.0],
                "relative_humidity_2m": [50.0],
                "wind_speed_10m": [10.0],
                "weather_code": [999],
            }
        }
        mock_urlopen.return_value = _MockHTTPResponse(json.dumps(unknown_wmo_payload).encode("utf-8"), status=200)

        res = fetch_open_meteo_forecast(self.target_dt)
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "weather_validation_error")
        self.assertEqual(res["warning_code"], "WARN_MAP_01")
        self.assertIn("Unknown weather code (WMO 999)", res["message"])

    @patch("urllib.request.urlopen")
    def test_tc_w05_unsupported_time_outside_forecast_coverage(self, mock_urlopen):
        """TC-W05d & TC-04: Target timestamp not in hourly array triggers unsupported_time and WARN_API_02."""
        outside_payload = {
            "hourly": {
                "time": ["2026-10-20T10:00"],
                "temperature_2m": [20.0],
                "apparent_temperature": [20.0],
                "relative_humidity_2m": [50.0],
                "wind_speed_10m": [10.0],
                "weather_code": [0],
            }
        }
        mock_urlopen.return_value = _MockHTTPResponse(json.dumps(outside_payload).encode("utf-8"), status=200)

        res = fetch_open_meteo_forecast(self.target_dt)
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "unsupported_time")
        self.assertEqual(res["warning_code"], "WARN_API_02")

    @patch("urllib.request.urlopen")
    def test_tc_w06_fallback_behavior_and_prediction(self, mock_urlopen):
        """TC-W06: Weather API failure routes to manual override or L0 baseline with non-negative prediction (NFR-5, FR-2.5)."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection timed out")

        weather_res = fetch_open_meteo_forecast(self.target_dt)
        self.assertFalse(weather_res["success"])
        self.assertEqual(weather_res["warning_code"], "WARN_API_01")

        mm = get_model_manager()

        # Pathway A: Manual Weather Override with labelled source
        manual_norm = normalize_weather_features(temp_c=18.0, atemp_c=16.0, hum_pct=50.0, wind_kmh=12.0)
        manual_norm["weathersit"] = 1
        weather_source_label = "Manual Input (Labelled Fallback)"
        self.assertNotEqual(weather_source_label, "Open-Meteo Forecast API (Live)")

        feature_row = {
            "season": 4, "mnth": 10, "hr": 14, "holiday": 0, "weekday": 5, "workingday": 1,
            "weathersit": manual_norm["weathersit"], "temp": manual_norm["temp"],
            "atemp": manual_norm["atemp"], "hum": manual_norm["hum"], "windspeed": manual_norm["windspeed"],
        }
        pred_l4 = mm.predict("L4", feature_row)
        self.assertGreaterEqual(pred_l4["predicted_cnt"], 0)

        # Pathway B: Explicit L0 baseline fallback
        pred_l0 = mm.predict("L0", feature_row)
        self.assertEqual(pred_l0["used_level"], "L0")
        self.assertEqual(pred_l0["predicted_cnt"], int(round(mm.l0_mean)))

    @patch("urllib.request.urlopen")
    def test_tc_w07_recovery_after_failure(self, mock_urlopen):
        """TC-W07: Transient weather API failure recovers seamlessly on subsequent successful request."""
        # Step 1: Simulated failure
        mock_urlopen.side_effect = TimeoutError("Temporary network blip")
        res_fail = fetch_open_meteo_forecast(self.target_dt)
        self.assertFalse(res_fail["success"])
        self.assertEqual(res_fail["warning_code"], "WARN_API_01")

        # Step 2: Immediate subsequent recovery with valid response
        payload_bytes = json.dumps(self.valid_payload).encode("utf-8")
        mock_urlopen.side_effect = None
        mock_urlopen.return_value = _MockHTTPResponse(payload_bytes, status=200)

        res_ok = fetch_open_meteo_forecast(self.target_dt)
        self.assertTrue(res_ok["success"])
        self.assertEqual(res_ok["status"], "ok")
        self.assertEqual(res_ok["weather_source"], "Open-Meteo Forecast API")
        self.assertEqual(res_ok["raw_weather"]["temperature_c"], 22.4)

        # Normal prediction operates cleanly
        mm = get_model_manager()
        norm = res_ok["normalized_weather"]
        feature_row = {
            "season": 4, "mnth": 10, "hr": 14, "holiday": 0, "weekday": 5, "workingday": 1,
            "weathersit": norm["weathersit"], "temp": norm["temp"],
            "atemp": norm["atemp"], "hum": norm["hum"], "windspeed": norm["windspeed"],
        }
        pred = mm.predict("L4", feature_row)
        self.assertGreaterEqual(pred["predicted_cnt"], 0)
        self.assertEqual(pred["status"], "ok")


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
        # L3 beats L2
        self.assertLess(scores["L3"], scores["L2"])
        # L4 is calculated
        self.assertIsNotNone(scores["L4"])

    @unittest.skipIf(not RF_AVAILABLE, f"RandomForestRegressor unavailable on host: {RF_IMPORT_ERROR}")
    def test_l5_random_forest_mae_score(self):
        scores = self.mm.mae_scores
        self.assertIsNotNone(scores["L5"])
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

    def test_l4_non_negative_predictions(self):
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
        self.assertGreaterEqual(pred_l4["predicted_cnt"], 0)

    @unittest.skipIf(not RF_AVAILABLE, f"RandomForestRegressor unavailable on host: {RF_IMPORT_ERROR}")
    def test_l5_non_negative_predictions(self):
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
        pred_l5 = self.mm.predict("L5", input_data)
        self.assertGreaterEqual(pred_l5["predicted_cnt"], 0)

    def test_tc_06_model_missing_uses_saved_l0_with_warn_sys_01(self):
        """TC-06: Missing model uses saved L0 configuration with WARN_SYS_01."""
        input_data = {
            "season": 1, "mnth": 1, "hr": 12, "holiday": 0, "weekday": 1,
            "workingday": 1, "weathersit": 1, "temp": 0.5, "atemp": 0.5,
            "hum": 0.5, "windspeed": 0.2,
        }
        with patch.object(self.mm, "lr_model", None):
            res = self.mm.predict("L4", input_data)
            self.assertEqual(res["used_level"], "L0")
            self.assertEqual(res["predicted_cnt"], int(round(self.mm.l0_mean)))
            self.assertEqual(res["warning_code"], "WARN_SYS_01")
            self.assertEqual(res["status"], "model_fallback_l0")
            self.assertIn("Model artifact unavailable; showing Level 0 fallback.", res["fallback_note"])

    def test_tc_07_model_and_l0_missing_returns_err_sys_01(self):
        """TC-07: Model and L0 config missing returns ERR_SYS_01; no fabricated count."""
        input_data = {
            "season": 1, "mnth": 1, "hr": 12, "holiday": 0, "weekday": 1,
            "workingday": 1, "weathersit": 1, "temp": 0.5, "atemp": 0.5,
            "hum": 0.5, "windspeed": 0.2,
        }
        with patch.object(self.mm, "l0_mean", None), patch.object(self.mm, "lr_model", None):
            res = self.mm.predict("L4", input_data)
            self.assertIsNone(res["predicted_cnt"])
            self.assertIsNone(res["predicted_cnt_raw"])
            self.assertEqual(res["error_code"], "ERR_SYS_01")
            self.assertEqual(res["status"], "prediction_error")
            self.assertIn("Model and L0 config unavailable", res["fallback_note"])

    def test_tc_08_cascading_baseline_fallback_chain(self):
        """TC-08: Complete cascading fallback chain: L3 -> L2 -> L1 -> L0."""
        inp_l3 = {
            "season": 1, "mnth": 1, "hr": 12, "holiday": 0, "weekday": 1,
            "workingday": 1, "weathersit": 1, "temp": 0.5, "atemp": 0.5,
            "hum": 0.5, "windspeed": 0.2,
        }
        res_l3 = self.mm.predict("L3", inp_l3)
        self.assertEqual(res_l3["used_level"], "L3")
        self.assertEqual(res_l3["status"], "ok")

        with patch.object(self.mm, "l3_map", {}):
            res_l2 = self.mm.predict("L3", inp_l3)
            self.assertEqual(res_l2["used_level"], "L2")
            self.assertEqual(res_l2["status"], "baseline_fallback")

        with patch.object(self.mm, "l3_map", {}), patch.object(self.mm, "l2_map", {}):
            res_l1 = self.mm.predict("L3", inp_l3)
            self.assertEqual(res_l1["used_level"], "L1")
            self.assertEqual(res_l1["status"], "baseline_fallback")

        with patch.object(self.mm, "l3_map", {}), patch.object(self.mm, "l2_map", {}), patch.object(self.mm, "l1_map", {}):
            res_l0 = self.mm.predict("L3", inp_l3)
            self.assertEqual(res_l0["used_level"], "L0")
            self.assertEqual(res_l0["status"], "baseline_fallback")
            self.assertEqual(res_l0["warning_code"], "WARN_SYS_01")

    def test_tc_09_repeat_identical_input_five_times(self):
        """TC-09: Repeated identical inputs produce identical outputs within numeric tolerance (NFR-4)."""
        input_data = {
            "season": 2, "mnth": 4, "hr": 8, "holiday": 0, "weekday": 2,
            "workingday": 1, "weathersit": 1, "temp": 0.45, "atemp": 0.45,
            "hum": 0.50, "windspeed": 0.15,
        }
        l4_runs = [self.mm.predict("L4", input_data)["predicted_cnt_raw"] for _ in range(5)]
        self.assertEqual(len(set(l4_runs)), 1, f"L4 outputs differed across repeated runs: {l4_runs}")

        if RF_AVAILABLE and self.mm.rf_model is not None:
            l5_runs = [self.mm.predict("L5", input_data)["predicted_cnt_raw"] for _ in range(5)]
            self.assertEqual(len(set(l5_runs)), 1, f"L5 outputs differed across repeated runs: {l5_runs}")

    def test_tc_10_chronological_evaluation_integrity(self):
        """TC-10: All six prediction tiers evaluate on identical chronological held-out rows (3,476)."""
        expected_test_count = 3476
        self.assertEqual(len(self.mm.test_df), expected_test_count)
        self.assertEqual(len(self.mm.y_test), expected_test_count)

        for lvl in ["L0", "L1", "L2", "L3", "L4"]:
            self.assertIn(lvl, self.mm.predictions)
            self.assertEqual(len(self.mm.predictions[lvl]), expected_test_count)
            self.assertIsNotNone(self.mm.mae_scores[lvl])
            self.assertGreater(self.mm.mae_scores[lvl], 0)

        if RF_AVAILABLE and self.mm.rf_model is not None:
            self.assertEqual(len(self.mm.predictions["L5"]), expected_test_count)
            self.assertIsNotNone(self.mm.mae_scores["L5"])
            self.assertLess(self.mm.mae_scores["L5"], 90.0)

    def test_deployment_artifact_contract_metadata(self):
        """SRS Section 5.7: Verify bundle metadata contains verified ranges, split, and dependency versions."""
        meta = getattr(self.mm, "bundle_metadata", {})
        self.assertEqual(meta.get("artifact_version"), "1.5.0")
        self.assertEqual(meta.get("srs_version"), "v1.5")
        self.assertIsNotNone(meta.get("training_date_range"))
        self.assertIsNotNone(meta.get("test_date_range"))
        self.assertEqual(meta.get("split_definition", {}).get("split_index"), 13903)
        self.assertIn("scikit-learn", meta.get("dependency_versions", {}))
        self.assertIn("pandas", meta.get("dependency_versions", {}))
        self.assertIn("numpy", meta.get("dependency_versions", {}))
        self.assertIn("python", meta.get("dependency_versions", {}))


class _MockRF:
    """Mock Random Forest estimator for bundle loading tests."""
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), 85.0)


class TestSprint2BundleLoading(unittest.TestCase):
    """Verify robust model bundle loading, integrity validation, and fallback logic."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.bundle_path = os.path.join(self.temp_dir.name, "test_bundle.joblib")
        # Lightweight fitted linear regression model with feature names
        lr = LinearRegression()
        dummy_X = pd.DataFrame(np.zeros((5, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
        lr.fit(dummy_X, np.array([10, 20, 30, 40, 50]))
        self.valid_base_bundle = {
            "artifact_version": "1.5.0",
            "srs_version": "v1.5",
            "feature_names": FEATURE_COLUMNS,
            "n_train": 13903,
            "n_test": 3476,
            "l0_mean": 174.64,
            "l1_map": {h: 100.0 for h in range(24)},
            "l2_map": {(h, w): 100.0 for h in range(24) for w in (0, 1)},
            "l3_map": {(h, w, s): 100.0 for h in range(24) for w in (0, 1) for s in (1, 2, 3, 4)},
            "lr_model": lr,
            "rf_model": None,
            "mae_scores": {
                "L0": 174.98,
                "L1": 113.73,
                "L2": 101.55,
                "L3": 95.48,
                "L4": 134.70,
                "L5": None,
            },
            "improvements": {
                "L0": -53.86,
                "L1": 0.0,
                "L2": 10.71,
                "L3": 16.05,
                "L4": -18.44,
                "L5": None,
            },
        }
        self.sample_input = {
            "season": 1,
            "mnth": 1,
            "hr": 12,
            "holiday": 0,
            "weekday": 1,
            "workingday": 1,
            "weathersit": 1,
            "temp": 0.5,
            "atemp": 0.5,
            "hum": 0.5,
            "windspeed": 0.2,
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_bundle_load_rf_unavailable_enforces_none_l5(self):
        """When RF is unavailable, loaded bundle must not fabricate L5 metrics and must fall back to L4."""
        joblib.dump(self.valid_base_bundle, self.bundle_path)
        with patch("src.models.RF_AVAILABLE", False):
            mm = ModelManager(bundle_path=self.bundle_path)
            self.assertTrue(mm.is_artifact_loaded)
            self.assertIsNone(mm.rf_model)
            self.assertIsNone(mm.mae_scores["L5"])
            self.assertIsNone(mm.improvements["L5"])
            self.assertIsNone(mm.predictions["L5"])

            pred = mm.predict("L5", self.sample_input)
            self.assertEqual(pred["used_level"], "L4")
            self.assertIn("fell back to L4", pred["fallback_note"])

    def test_bundle_load_rf_unavailable_strips_fabricated_l5(self):
        """Even if a bundle contains fabricated L5 scores, RF_AVAILABLE=False zeroes them out."""
        bogus_bundle = dict(self.valid_base_bundle)
        bogus_bundle["mae_scores"] = dict(self.valid_base_bundle["mae_scores"], L5=50.0)
        bogus_bundle["improvements"] = dict(self.valid_base_bundle["improvements"], L5=56.0)
        joblib.dump(bogus_bundle, self.bundle_path)

        with patch("src.models.RF_AVAILABLE", False):
            mm = ModelManager(bundle_path=self.bundle_path)
            self.assertIsNone(mm.mae_scores["L5"])
            self.assertIsNone(mm.improvements["L5"])
            self.assertIsNone(mm.predictions["L5"])

    def test_bundle_load_rf_available_triggers_retrain_when_l5_missing(self):
        """When RF is available but saved bundle has rf_model=None, retrain to enable L5."""
        joblib.dump(self.valid_base_bundle, self.bundle_path)

        with patch("src.models.RF_AVAILABLE", True), \
             patch("src.models.RandomForestRegressor", _MockRF):
            mm = ModelManager(bundle_path=self.bundle_path)
            self.assertIsNotNone(mm.rf_model)
            self.assertIsNotNone(mm.mae_scores["L5"])
            self.assertIsNotNone(mm.predictions["L5"])
            pred = mm.predict("L5", self.sample_input)
            self.assertEqual(pred["used_level"], "L5")
            self.assertEqual(pred["fallback_note"], "")

    def test_bundle_load_rf_available_preserves_complete_bundle(self):
        """When RF is available and bundle is complete with valid L5, no retraining occurs."""
        complete_bundle = dict(self.valid_base_bundle)
        complete_bundle["rf_model"] = _MockRF()
        complete_bundle["mae_scores"] = dict(self.valid_base_bundle["mae_scores"], L5=85.19)
        complete_bundle["improvements"] = dict(self.valid_base_bundle["improvements"], L5=25.09)
        joblib.dump(complete_bundle, self.bundle_path)

        with patch("src.models.RF_AVAILABLE", True), \
             patch("src.models.RandomForestRegressor", _MockRF), \
             patch.object(ModelManager, "_train_and_save") as mock_train:
            mm = ModelManager(bundle_path=self.bundle_path)
            mock_train.assert_not_called()
            self.assertTrue(mm.is_artifact_loaded)
            self.assertIsNotNone(mm.rf_model)
            self.assertIsNotNone(mm.mae_scores["L5"])

    def test_bundle_load_corrupted_file_falls_back_to_l0_with_warn_sys_01(self):
        """SRS Section 5.4, 5.5 & FR-3.6: Corrupted bundle file must not retrain; falls back to saved L0 with WARN_SYS_01."""
        with open(self.bundle_path, "wb") as f:
            f.write(b"not-a-valid-joblib-stream")

        mm = ModelManager(bundle_path=self.bundle_path)
        self.assertFalse(mm.is_artifact_loaded)
        self.assertIsNone(mm.lr_model)
        self.assertIsNone(mm.rf_model)
        self.assertIsNotNone(mm.l0_mean)

        # File on disk was NOT overwritten by auto-retraining
        with open(self.bundle_path, "rb") as f:
            self.assertEqual(f.read(), b"not-a-valid-joblib-stream")

        # Prediction falls back to L0 with WARN_SYS_01
        res = mm.predict("L4", self.sample_input)
        self.assertEqual(res["used_level"], "L0")
        self.assertEqual(res["warning_code"], "WARN_SYS_01")
        self.assertEqual(res["status"], "model_fallback_l0")
        self.assertEqual(res["predicted_cnt"], int(round(mm.l0_mean)))
        self.assertIn("Model artifact unavailable; showing Level 0 fallback.", res["fallback_note"])

    def test_bundle_load_corrupted_file_and_missing_l0_returns_err_sys_01(self):
        """SRS Section 5.4, 5.5 & TC-07: When model and L0 config are both unavailable, return ERR_SYS_01."""
        with open(self.bundle_path, "wb") as f:
            f.write(b"not-a-valid-joblib-stream")

        mm = ModelManager(bundle_path=self.bundle_path)
        mm.l0_mean = None

        res = mm.predict("L4", self.sample_input)
        self.assertIsNone(res["predicted_cnt"])
        self.assertIsNone(res["predicted_cnt_raw"])
        self.assertEqual(res["error_code"], "ERR_SYS_01")
        self.assertEqual(res["status"], "prediction_error")
        self.assertIn("Model and L0 config unavailable", res["fallback_note"])


class TestSprint2ResponsiveLayout(unittest.TestCase):
    """Testing Plan Step 4: Responsive layout breakpoints, controls, and table overflow checks."""

    @classmethod
    def setUpClass(cls):
        with open("app.py", "r", encoding="utf-8") as f:
            cls.app_source = f.read()

    def test_responsive_media_queries_defined(self):
        """Verify media query breakpoints exist for tablet (768px) and mobile (480px / 390px)."""
        self.assertIn("@media (max-width: 768px)", self.app_source)
        self.assertIn("@media (max-width: 480px)", self.app_source)

    def test_responsive_weather_grid_breakpoint(self):
        """Verify weather grid transitions from 4 columns to 2 columns on mobile/tablet viewports."""
        self.assertIn(".weather-grid", self.app_source)
        self.assertIn("grid-template-columns: repeat(2, 1fr)", self.app_source)

    def test_page_header_responsive_stacking(self):
        """Verify page header stacks vertically and badges wrap on narrow viewports."""
        self.assertIn("flex-direction: column !important", self.app_source)
        self.assertIn("flex-wrap: wrap !important", self.app_source)

    def test_manual_weather_override_controls_and_units(self):
        """Verify all 4 manual weather controls have clear labels with documented units."""
        self.assertIn('st.number_input("Temperature (°C)"', self.app_source)
        self.assertIn('st.slider("Humidity (%)"', self.app_source)
        self.assertIn('st.number_input("Wind Speed (km/h)"', self.app_source)
        self.assertIn('st.selectbox("Weather Condition"', self.app_source)

    def test_model_performance_table_responsive_scroll_wrapper(self):
        """Verify the 5-tier evaluation comparison table is enclosed in an overflow-x auto wrapper."""
        self.assertIn("overflow-x:auto", self.app_source)


class TestSprint2MAEFormulaRendering(unittest.TestCase):
    """Testing Plan Step 5: Verification of native mathematical rendering for MAE on About Project page.

    Note: These automated tests verify source implementation and usage of Streamlit's native st.latex API;
    actual visual layout rendering in the DOM requires interactive inspection in a physical browser.
    """

    @classmethod
    def setUpClass(cls):
        with open("app.py", "r", encoding="utf-8") as f:
            cls.app_source = f.read()

    def test_mae_formula_uses_native_st_latex(self):
        """Verify that the About Project page employs native st.latex() for mathematical formatting."""
        self.assertIn('st.latex(r"\\text{MAE} = \\frac{1}{n} \\sum_{i=1}^n |y_i - \\hat{y}_i|")', self.app_source)

    def test_mae_formula_content_standard_form(self):
        """Verify standard MAE mathematical formula structure: (1/n) * sum |y_i - y_hat_i|."""
        expected_components = ["\\text{MAE}", "\\frac{1}{n}", "\\sum_{i=1}^n", "|y_i - \\hat{y}_i|"]
        for comp in expected_components:
            self.assertIn(comp, self.app_source, f"Missing mathematical component: {comp}")

    def test_mae_explanatory_text_definition(self):
        """Verify explanatory text states MAE is the average absolute difference between actual and predicted counts."""
        self.assertIn(
            "MAE measures the average absolute difference between actual and predicted rental counts",
            self.app_source,
        )

    def test_no_unrendered_latex_in_raw_html_containers(self):
        """Verify raw LaTeX delimiters ($$) are not nested inside raw HTML blocks where KaTeX is ignored."""
        # The buggy pattern was <div ...>$$\text{MAE} = ...$$</div>
        self.assertNotIn("$$\\text{MAE}", self.app_source)
        self.assertNotIn("$$\\\\text{MAE}", self.app_source)


if __name__ == "__main__":
    unittest.main()
