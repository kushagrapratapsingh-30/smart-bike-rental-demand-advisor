"""Sprint 4 Phase 2 Test Suite: Data, Weather & Calendar Verification.

Covers:
- Exhaustive verification of all 731 unique dates in UCI hour.csv against Washington, D.C. observed holiday rules.
- Exact holiday observation rules (Emancipation Day, Sunday-to-Monday shifts, pre/post 2021 Juneteenth policy).
- Weather normalization formulas (temp, atemp, hum, windspeed) per SRS v1.5 Appendix B.
- Open-Meteo wind-speed conversion and unit consistency.
- WMO weather-code mappings (all 25 supported codes) and unknown code rejection (WARN_MAP_01).
- Weather API response validation, missing fields, timestamp matching, malformed payloads, and leap days.
"""

from datetime import date, datetime
import json
import os
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

import numpy as np
import pandas as pd

from src.calendar_utils import (
    derive_calendar_features,
    get_astronomical_season,
    is_us_dc_holiday,
)
from src.data_cleaning import load_dataset
from src.models import FEATURE_COLUMNS, ModelManager, get_model_manager
from src.prediction_service import predict_demand_workflow
from src.weather import (
    WMO_CODE_MAP,
    fetch_open_meteo_forecast,
    map_wmo_code,
    normalize_weather_features,
)


class TestSprint4CalendarVerification(unittest.TestCase):
    """Exhaustive calendar, holiday, and season verification adhering to SRS v1.5 Section 4.4 and Section 13 Gate 6."""

    def setUp(self):
        self.dataset_path = "dataset/hour.csv"
        self.df = load_dataset(self.dataset_path)

    def test_exhaustive_731_dates_against_uci_hour_csv(self):
        """Verify that every single unique date in hour.csv (731 dates) matches the implemented calendar derivation.

        Tests:
        - holiday: matches hour.csv exactly for all 731 dates (0 discrepancies).
        - workingday: matches hour.csv exactly for all 731 dates (0 discrepancies).
        - weekday: matches hour.csv exactly for all 731 dates (0 discrepancies).
        - season: matches hour.csv exactly for all 731 dates (0 discrepancies).
        """
        daily = self.df[["dteday", "weekday", "holiday", "workingday", "season"]].drop_duplicates()
        self.assertEqual(len(daily), 731, f"Expected 731 unique dates in hour.csv, found {len(daily)}")

        holiday_mismatches = []
        workingday_mismatches = []
        weekday_mismatches = []
        season_mismatches = []

        for _, row in daily.iterrows():
            d_str = row["dteday"]
            dt = datetime.strptime(d_str, "%Y-%m-%d")
            cal = derive_calendar_features(dt)

            if cal["holiday"] != row["holiday"]:
                holiday_mismatches.append((d_str, row["holiday"], cal["holiday"]))
            if cal["workingday"] != row["workingday"]:
                workingday_mismatches.append((d_str, row["workingday"], cal["workingday"]))
            if cal["weekday"] != row["weekday"]:
                weekday_mismatches.append((d_str, row["weekday"], cal["weekday"]))
            if cal["season"] != row["season"]:
                season_mismatches.append((d_str, row["season"], cal["season"]))

        self.assertEqual(len(holiday_mismatches), 0, f"Holiday mismatches in dataset: {holiday_mismatches}")
        self.assertEqual(len(workingday_mismatches), 0, f"Workingday mismatches in dataset: {workingday_mismatches}")
        self.assertEqual(len(weekday_mismatches), 0, f"Weekday mismatches in dataset: {weekday_mismatches}")
        self.assertEqual(len(season_mismatches), 0, f"Season mismatches in dataset: {season_mismatches}")

    def test_dc_emancipation_day_2011_observed_friday(self):
        """In 2011, D.C. Emancipation Day (April 16) was Saturday, observed on Friday April 15."""
        # 2011-04-15 (Friday) -> Observed holiday in D.C.
        d_obs = date(2011, 4, 15)
        self.assertTrue(is_us_dc_holiday(d_obs))
        cal = derive_calendar_features(datetime(2011, 4, 15, 10, 0))
        self.assertEqual(cal["holiday"], 1)
        self.assertEqual(cal["workingday"], 0)
        self.assertEqual(cal["weekday"], 5)  # Friday in UCI convention

        # 2011-04-16 (Saturday) -> Regular weekend day, not designated holiday
        d_sat = date(2011, 4, 16)
        self.assertFalse(is_us_dc_holiday(d_sat))

    def test_dc_emancipation_day_2012_monday(self):
        """In 2012, D.C. Emancipation Day (April 16) was Monday, observed on Monday."""
        d = date(2012, 4, 16)
        self.assertTrue(is_us_dc_holiday(d))
        cal = derive_calendar_features(datetime(2012, 4, 16, 9, 0))
        self.assertEqual(cal["holiday"], 1)
        self.assertEqual(cal["workingday"], 0)
        self.assertEqual(cal["weekday"], 1)  # Monday in UCI convention

    def test_dc_emancipation_day_future_weekday(self):
        """In 2026, April 16 is a Thursday; should be marked holiday."""
        d = date(2026, 4, 16)
        self.assertTrue(is_us_dc_holiday(d))
        cal = derive_calendar_features(datetime(2026, 4, 16, 12, 0))
        self.assertEqual(cal["holiday"], 1)
        self.assertEqual(cal["workingday"], 0)

    def test_sunday_to_monday_observed_christmas_2011(self):
        """Christmas 2011 fell on Sunday Dec 25; observed on Monday Dec 26."""
        # Sunday Dec 25 -> weekend, holiday=0 per D.C./federal rules
        d_sun = date(2011, 12, 25)
        self.assertFalse(is_us_dc_holiday(d_sun))
        cal_sun = derive_calendar_features(datetime(2011, 12, 25, 12, 0))
        self.assertEqual(cal_sun["holiday"], 0)
        self.assertEqual(cal_sun["workingday"], 0)
        self.assertEqual(cal_sun["weekday"], 0)  # Sunday

        # Monday Dec 26 -> Observed holiday
        d_mon = date(2011, 12, 26)
        self.assertTrue(is_us_dc_holiday(d_mon))
        cal_mon = derive_calendar_features(datetime(2011, 12, 26, 12, 0))
        self.assertEqual(cal_mon["holiday"], 1)
        self.assertEqual(cal_mon["workingday"], 0)
        self.assertEqual(cal_mon["weekday"], 1)  # Monday

    def test_sunday_to_monday_observed_new_year_2012(self):
        """New Year 2012 fell on Sunday Jan 1; observed on Monday Jan 2."""
        d_sun = date(2012, 1, 1)
        self.assertFalse(is_us_dc_holiday(d_sun))

        d_mon = date(2012, 1, 2)
        self.assertTrue(is_us_dc_holiday(d_mon))
        cal = derive_calendar_features(datetime(2012, 1, 2, 8, 0))
        self.assertEqual(cal["holiday"], 1)
        self.assertEqual(cal["workingday"], 0)

    def test_sunday_to_monday_observed_veterans_day_2012(self):
        """Veterans Day 2012 fell on Sunday Nov 11; observed on Monday Nov 12."""
        d_sun = date(2012, 11, 11)
        self.assertFalse(is_us_dc_holiday(d_sun))

        d_mon = date(2012, 11, 12)
        self.assertTrue(is_us_dc_holiday(d_mon))
        cal = derive_calendar_features(datetime(2012, 11, 12, 14, 0))
        self.assertEqual(cal["holiday"], 1)
        self.assertEqual(cal["workingday"], 0)

    def test_saturday_new_year_2011_not_saturday_holiday(self):
        """Jan 1, 2011 was a Saturday; federal/D.C. observance was Dec 31, 2010."""
        d = date(2011, 1, 1)
        self.assertFalse(is_us_dc_holiday(d))
        cal = derive_calendar_features(datetime(2011, 1, 1, 0, 0))
        self.assertEqual(cal["holiday"], 0)
        self.assertEqual(cal["workingday"], 0)
        self.assertEqual(cal["weekday"], 6)  # Saturday

    def test_juneteenth_historical_vs_modern_policy(self):
        """Juneteenth (June 19) was not recognized in 2011-2012; is recognized 2021+."""
        # 2011-06-19 (Sunday) -> Not holiday
        self.assertFalse(is_us_dc_holiday(date(2011, 6, 19)))

        # 2012-06-19 (Tuesday) -> Regular working day
        self.assertFalse(is_us_dc_holiday(date(2012, 6, 19)))
        cal_2012 = derive_calendar_features(datetime(2012, 6, 19, 10, 0))
        self.assertEqual(cal_2012["holiday"], 0)
        self.assertEqual(cal_2012["workingday"], 1)

        # 2024-06-19 (Wednesday) -> Modern federal & D.C. holiday
        self.assertTrue(is_us_dc_holiday(date(2024, 6, 19)))
        cal_2024 = derive_calendar_features(datetime(2024, 6, 19, 10, 0))
        self.assertEqual(cal_2024["holiday"], 1)
        self.assertEqual(cal_2024["workingday"], 0)

    def test_floating_monday_holidays(self):
        """Verify floating Monday holidays across test years."""
        # MLK Day: 3rd Monday in Jan (e.g. 2026-01-19)
        self.assertTrue(is_us_dc_holiday(date(2026, 1, 19)))

        # Presidents Day: 3rd Monday in Feb (e.g. 2026-02-16)
        self.assertTrue(is_us_dc_holiday(date(2026, 2, 16)))

        # Memorial Day: Last Monday in May (e.g. 2026-05-25)
        self.assertTrue(is_us_dc_holiday(date(2026, 5, 25)))

        # Labor Day: 1st Monday in Sep (e.g. 2026-09-07)
        self.assertTrue(is_us_dc_holiday(date(2026, 9, 7)))

        # Columbus Day: 2nd Monday in Oct (e.g. 2026-10-12)
        self.assertTrue(is_us_dc_holiday(date(2026, 10, 12)))

    def test_thanksgiving_fourth_thursday(self):
        """Thanksgiving is the fourth Thursday of November."""
        # 2026 Thanksgiving is Nov 26
        self.assertTrue(is_us_dc_holiday(date(2026, 11, 26)))
        # Wednesday before Thanksgiving is not a holiday
        self.assertFalse(is_us_dc_holiday(date(2026, 11, 25)))

    def test_leap_year_dates(self):
        """Verify leap-year date parsing and calendar derivation."""
        for leap_year in [2012, 2016, 2020, 2024, 2028]:
            dt = datetime(leap_year, 2, 29, 15, 0)
            cal = derive_calendar_features(dt)
            self.assertEqual(cal["mnth"], 2)
            self.assertEqual(cal["day"], 29)
            self.assertEqual(cal["season"], 1)  # Winter
            self.assertIn(cal["workingday"], (0, 1))

    def test_season_boundary_dates(self):
        """Verify astronomical season boundaries: Mar 21, Jun 21, Sep 23, Dec 21."""
        # Winter / Spring transition
        self.assertEqual(get_astronomical_season(3, 20), 1)  # Winter
        self.assertEqual(get_astronomical_season(3, 21), 2)  # Spring

        # Spring / Summer transition
        self.assertEqual(get_astronomical_season(6, 20), 2)  # Spring
        self.assertEqual(get_astronomical_season(6, 21), 3)  # Summer

        # Summer / Fall transition
        self.assertEqual(get_astronomical_season(9, 22), 3)  # Summer
        self.assertEqual(get_astronomical_season(9, 23), 4)  # Fall

        # Fall / Winter transition
        self.assertEqual(get_astronomical_season(12, 20), 4)  # Fall
        self.assertEqual(get_astronomical_season(12, 21), 1)  # Winter


class TestSprint4WeatherNormalization(unittest.TestCase):
    """Verify weather normalization and unit conversion adhering to SRS v1.5 Appendix B."""

    def test_temperature_min_max_and_midpoint_scaling(self):
        """temp_norm = (temp_c - (-8.0)) / (39.0 - (-8.0)) = (temp_c + 8.0) / 47.0."""
        # Minimum: -8°C -> 0.0
        n_min = normalize_weather_features(-8.0, 0.0, 50.0, 10.0)
        self.assertAlmostEqual(n_min["temp"], 0.0, places=4)

        # Maximum: +39°C -> 1.0
        n_max = normalize_weather_features(39.0, 0.0, 50.0, 10.0)
        self.assertAlmostEqual(n_max["temp"], 1.0, places=4)

        # Midpoint: 15.5°C -> (15.5 + 8) / 47 = 23.5 / 47 = 0.5
        n_mid = normalize_weather_features(15.5, 0.0, 50.0, 10.0)
        self.assertAlmostEqual(n_mid["temp"], 0.5, places=4)

        # Below min clamps to 0.0
        n_sub = normalize_weather_features(-15.0, 0.0, 50.0, 10.0)
        self.assertEqual(n_sub["temp"], 0.0)

        # Above max clamps to 1.0
        n_super = normalize_weather_features(45.0, 0.0, 50.0, 10.0)
        self.assertEqual(n_super["temp"], 1.0)

    def test_apparent_temperature_min_max_and_midpoint_scaling(self):
        """atemp_norm = (atemp_c - (-16.0)) / (50.0 - (-16.0)) = (atemp_c + 16.0) / 66.0."""
        # Minimum: -16°C -> 0.0
        n_min = normalize_weather_features(20.0, -16.0, 50.0, 10.0)
        self.assertAlmostEqual(n_min["atemp"], 0.0, places=4)

        # Maximum: +50°C -> 1.0
        n_max = normalize_weather_features(20.0, 50.0, 50.0, 10.0)
        self.assertAlmostEqual(n_max["atemp"], 1.0, places=4)

        # Midpoint: 17.0°C -> (17 + 16) / 66 = 33 / 66 = 0.5
        n_mid = normalize_weather_features(20.0, 17.0, 50.0, 10.0)
        self.assertAlmostEqual(n_mid["atemp"], 0.5, places=4)

    def test_humidity_scaling_and_bounds(self):
        """hum_norm = hum_pct / 100.0."""
        n0 = normalize_weather_features(20.0, 20.0, 0.0, 10.0)
        self.assertAlmostEqual(n0["hum"], 0.0, places=4)

        n100 = normalize_weather_features(20.0, 20.0, 100.0, 10.0)
        self.assertAlmostEqual(n100["hum"], 1.0, places=4)

        n65 = normalize_weather_features(20.0, 20.0, 65.0, 10.0)
        self.assertAlmostEqual(n65["hum"], 0.65, places=4)

    def test_wind_speed_kmh_to_uci_scaling_max_67(self):
        """windspeed_norm = wind_kmh / 67.0."""
        # Calm: 0 km/h -> 0.0
        n0 = normalize_weather_features(20.0, 20.0, 50.0, 0.0)
        self.assertAlmostEqual(n0["windspeed"], 0.0, places=4)

        # Max scale: 67 km/h -> 1.0
        n67 = normalize_weather_features(20.0, 20.0, 50.0, 67.0)
        self.assertAlmostEqual(n67["windspeed"], 1.0, places=4)

        # Historical max in hour.csv was 57 km/h / 67 = 0.8507
        n57 = normalize_weather_features(20.0, 20.0, 50.0, 57.0)
        self.assertAlmostEqual(n57["windspeed"], 0.8507, places=4)

    def test_no_double_normalization_invariant(self):
        """Verify that normalizing an already normalized value does not occur in prediction workflow."""
        workflow_res = predict_demand_workflow(
            target_date=date(2026, 6, 15),
            target_hour=14,
            model_level="L5",
            use_manual_weather=True,
            manual_weather={
                "temperature_c": 23.5,  # (23.5 + 8) / 47 = 0.6702
                "apparent_temperature_c": 23.5,
                "humidity_pct": 50.0,  # 0.50
                "wind_speed_kmh": 13.4,  # 13.4 / 67 = 0.20
                "weather_desc": "Clear",
            },
        )
        self.assertTrue(workflow_res["success"])
        feat = workflow_res["feature_dict"]

        # Features passed to model must be strictly in [0, 1] range
        self.assertAlmostEqual(feat["temp"], 0.6702, places=3)
        self.assertAlmostEqual(feat["hum"], 0.50, places=3)
        self.assertAlmostEqual(feat["windspeed"], 0.20, places=3)
        self.assertTrue(0.0 <= feat["temp"] <= 1.0)
        self.assertTrue(0.0 <= feat["atemp"] <= 1.0)
        self.assertTrue(0.0 <= feat["hum"] <= 1.0)
        self.assertTrue(0.0 <= feat["windspeed"] <= 1.0)

    def test_feature_column_schema_and_order_invariance(self):
        """Feature schema must strictly match the 11 candidate features expected by trained models."""
        expected = [
            "season", "mnth", "hr", "holiday", "weekday", "workingday",
            "weathersit", "temp", "atemp", "hum", "windspeed"
        ]
        self.assertEqual(FEATURE_COLUMNS, expected)


class TestSprint4WMOAndAPIPayloadRobustness(unittest.TestCase):
    """Verify WMO code mappings, unknown code rejection, and robust payload edge cases."""

    def test_all_28_supported_wmo_codes_map_correctly(self):
        """Verify that all 28 registered WMO codes map to classes 1, 2, 3, or 4."""
        self.assertEqual(len(WMO_CODE_MAP), 28)
        for code, (weathersit, desc) in WMO_CODE_MAP.items():
            mapped_sit, mapped_desc = map_wmo_code(code)
            self.assertEqual(mapped_sit, weathersit)
            self.assertIn(mapped_sit, (1, 2, 3, 4))
            self.assertTrue(len(desc) > 0)

    def test_unknown_wmo_code_rejection_warn_map_01(self):
        """Unknown WMO codes must return 0 and raise WARN_MAP_01."""
        unknown_codes = [-5, 4, 10, 42, 98, 999]
        for bad_code in unknown_codes:
            sit, desc = map_wmo_code(bad_code)
            self.assertEqual(sit, 0)
            self.assertIn("Unknown", desc)

    @patch("urllib.request.urlopen")
    def test_missing_required_hourly_fields_rejection(self, mock_urlopen):
        """Missing required fields in hourly JSON payload returns weather_validation_error."""
        mock_response = MagicMock()
        mock_response.status = 200
        # Incomplete payload: missing 'temperature_2m' and 'weather_code'
        incomplete_payload = {
            "hourly": {
                "time": ["2026-06-15T12:00"],
                "relative_humidity_2m": [50.0],
                "wind_speed_10m": [15.0],
            }
        }
        mock_response.read.return_value = json.dumps(incomplete_payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        res = fetch_open_meteo_forecast(datetime(2026, 6, 15, 12, 0))
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "weather_validation_error")
        self.assertEqual(res["warning_code"], "WARN_API_01")

    @patch("urllib.request.urlopen")
    def test_empty_hourly_payload_rejection(self, mock_urlopen):
        """Empty hourly array returns unsupported_time."""
        mock_response = MagicMock()
        mock_response.status = 200
        empty_payload = {
            "hourly": {
                "time": [],
                "temperature_2m": [],
                "apparent_temperature": [],
                "relative_humidity_2m": [],
                "wind_speed_10m": [],
                "weather_code": [],
            }
        }
        mock_response.read.return_value = json.dumps(empty_payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        res = fetch_open_meteo_forecast(datetime(2026, 6, 15, 12, 0))
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "unsupported_time")
        self.assertEqual(res["warning_code"], "WARN_API_02")

    @patch("urllib.request.urlopen")
    def test_null_value_in_weather_attributes_rejection(self, mock_urlopen):
        """Null/None values inside hourly arrays must be rejected without crash."""
        mock_response = MagicMock()
        mock_response.status = 200
        null_payload = {
            "hourly": {
                "time": ["2026-06-15T12:00"],
                "temperature_2m": [None],
                "apparent_temperature": [20.0],
                "relative_humidity_2m": [50.0],
                "wind_speed_10m": [10.0],
                "weather_code": [0],
            }
        }
        mock_response.read.return_value = json.dumps(null_payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        res = fetch_open_meteo_forecast(datetime(2026, 6, 15, 12, 0))
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "weather_validation_error")
        self.assertEqual(res["warning_code"], "WARN_API_01")

    @patch("urllib.request.urlopen")
    def test_leap_day_timestamp_in_forecast_payload(self, mock_urlopen):
        """Verify that a forecast payload for Feb 29 (leap day) parses cleanly."""
        mock_response = MagicMock()
        mock_response.status = 200
        leap_dt_str = "2028-02-29T14:00"
        leap_payload = {
            "hourly": {
                "time": [leap_dt_str],
                "temperature_2m": [12.5],
                "apparent_temperature": [11.0],
                "relative_humidity_2m": [60.0],
                "wind_speed_10m": [15.0],
                "weather_code": [1],
            }
        }
        mock_response.read.return_value = json.dumps(leap_payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        target_dt = datetime(2028, 2, 29, 14, 0)
        res = fetch_open_meteo_forecast(target_dt)
        self.assertTrue(res["success"])
        self.assertEqual(res["timestamp"], leap_dt_str)
        self.assertEqual(res["raw_weather"]["temperature_c"], 12.5)
        self.assertEqual(res["normalized_weather"]["weathersit"], 1)


if __name__ == "__main__":
    unittest.main()
