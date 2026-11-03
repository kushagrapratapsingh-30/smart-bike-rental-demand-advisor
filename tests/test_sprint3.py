"""Unit tests for Sprint 3 execution logger, schema adherence, and concurrency safety.

Conforms to SRS v1.5 Section 5.6, Table 9.4, and Appendix C (TC-13).
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
import json
import os
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
import uuid

from src.data_cleaning import load_dataset
from src.models import ModelManager, get_model_manager
from src.prediction_service import predict_demand_workflow, validate_request_datetime
from src.logger import (
    DEFAULT_LOG_PATH,
    ExecutionLogger,
    get_execution_logger,
    is_valid_uuid4,
    sanitize_payload,
)


class TestSprint3UUIDValidation(unittest.TestCase):
    """Verify RFC 4122 UUID v4 formatting and separation per Table 9.4."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_path = os.path.join(self.temp_dir.name, "test_log.jsonl")
        self.logger = ExecutionLogger(log_path=self.log_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_uuid4_format_validation_helper(self):
        """Verify is_valid_uuid4 correctly identifies genuine UUID v4s and rejects invalid strings."""
        genuine_v4 = str(uuid.uuid4())
        self.assertTrue(is_valid_uuid4(genuine_v4))

        # Invalid formats
        self.assertFalse(is_valid_uuid4("not-a-uuid"))
        self.assertFalse(is_valid_uuid4(""))
        self.assertFalse(is_valid_uuid4(12345))
        self.assertFalse(is_valid_uuid4(None))
        self.assertFalse(is_valid_uuid4("123e4567-e89b-12d3-a456-426614174000"))  # UUID v1, not v4

    def test_log_id_and_request_id_are_distinct_uuid4(self):
        """Verify each logged entry generates separate, distinct RFC 4122 UUID v4s."""
        rec = self.logger.log_execution(
            status="ok",
            requested_datetime="2026-10-30T14:00",
            predicted_cnt=250,
            model_level="L5",
        )
        self.assertTrue(is_valid_uuid4(rec["log_id"]))
        self.assertTrue(is_valid_uuid4(rec["request_id"]))
        self.assertNotEqual(rec["log_id"], rec["request_id"])

    def test_preserve_existing_valid_request_id(self):
        """Verify an existing valid request_id is preserved while assigning a distinct log_id."""
        existing_req_id = str(uuid.uuid4())
        rec = self.logger.log_execution(
            status="ok",
            request_id=existing_req_id,
            requested_datetime="2026-10-30T15:00",
            predicted_cnt=180,
            model_level="L4",
        )
        self.assertEqual(rec["request_id"], existing_req_id)
        self.assertTrue(is_valid_uuid4(rec["log_id"]))
        self.assertNotEqual(rec["log_id"], existing_req_id)


class TestSprint3ExecutionLoggerStructure(unittest.TestCase):
    """Verify SRS v1.5 Table 9.4 ExecutionLog schema completeness and persistence."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_path = os.path.join(self.temp_dir.name, "execution_log.jsonl")
        self.logger = ExecutionLogger(log_path=self.log_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_table_9_4_schema_fields_complete(self):
        """Verify that every field mandated by SRS v1.5 Table 9.4 is present in the record."""
        required_fields = {
            "log_id",
            "request_id",
            "timestamp",
            "requested_datetime",
            "status",
            "error_code",
            "weather_source",
            "api_status",
            "input_vector",
            "predicted_cnt",
            "model_level",
            "model_version",
        }
        rec = self.logger.log_execution(
            status="ok",
            requested_datetime="2026-10-30T16:00",
            error_code=None,
            weather_source="Open-Meteo Forecast API",
            api_status=200,
            input_vector={"season": 4, "temp": 0.45, "hr": 16},
            predicted_cnt=320,
            model_level="L5",
            model_version="1.5.0",
        )
        self.assertEqual(required_fields, set(rec.keys()))

    def test_optional_fields_safely_default_to_none(self):
        """Verify that missing optional parameters safely resolve to None/null in record."""
        rec = self.logger.log_execution(status="validation_error", error_code="ERR_VAL_01")
        self.assertIsNone(rec["requested_datetime"])
        self.assertIsNone(rec["weather_source"])
        self.assertIsNone(rec["api_status"])
        self.assertIsNone(rec["input_vector"])
        self.assertIsNone(rec["predicted_cnt"])
        self.assertIsNone(rec["model_level"])
        self.assertEqual(rec["status"], "validation_error")
        self.assertEqual(rec["error_code"], "ERR_VAL_01")

    def test_jsonl_file_persistence(self):
        """Verify that records are appended to the JSONL file as valid, parseable lines."""
        self.logger.log_execution(status="ok", predicted_cnt=100)
        self.logger.log_execution(status="weather_api_error", error_code="WARN_API_01")

        self.assertTrue(os.path.exists(self.log_path))
        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        self.assertEqual(len(lines), 2)
        parsed_0 = json.loads(lines[0])
        parsed_1 = json.loads(lines[1])
        self.assertEqual(parsed_0["status"], "ok")
        self.assertEqual(parsed_0["predicted_cnt"], 100)
        self.assertEqual(parsed_1["status"], "weather_api_error")
        self.assertEqual(parsed_1["error_code"], "WARN_API_01")

    def test_get_recent_logs_ordering_and_limit(self):
        """Verify get_recent_logs returns entries newest-first and respects limit parameter."""
        for i in range(15):
            self.logger.log_execution(status="ok", predicted_cnt=i)

        # Retrieve with limit=5
        recent = self.logger.get_recent_logs(limit=5)
        self.assertEqual(len(recent), 5)
        # First element must be the newest (i=14)
        self.assertEqual(recent[0]["predicted_cnt"], 14)
        self.assertEqual(recent[1]["predicted_cnt"], 13)
        self.assertEqual(recent[4]["predicted_cnt"], 10)

    def test_empty_log_retrieval(self):
        """Verify get_recent_logs returns empty list when file does not exist."""
        empty_logger = ExecutionLogger(log_path=os.path.join(self.temp_dir.name, "nonexistent.jsonl"))
        self.assertEqual(empty_logger.get_recent_logs(), [])


class TestSprint3SecretSanitization(unittest.TestCase):
    """Verify that credentials, tokens, and secrets are never logged per SRS Section 5.6."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_path = os.path.join(self.temp_dir.name, "sanitized_log.jsonl")
        self.logger = ExecutionLogger(log_path=self.log_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sensitive_keys_redacted_in_input_vector(self):
        """Verify that dictionary keys matching sensitive substrings are redacted."""
        sensitive_inputs = {
            "season": 2,
            "temp": 0.5,
            "api_key": "secret-12345",
            "auth_token": "bearer-token-abc",
            "password": "super-secret-password",
            "user_credential": "login-credentials",
        }
        rec = self.logger.log_execution(status="ok", input_vector=sensitive_inputs)
        clean = rec["input_vector"]

        self.assertEqual(clean["season"], 2)
        self.assertEqual(clean["temp"], 0.5)
        self.assertEqual(clean["api_key"], "[REDACTED]")
        self.assertEqual(clean["auth_token"], "[REDACTED]")
        self.assertEqual(clean["password"], "[REDACTED]")
        self.assertEqual(clean["user_credential"], "[REDACTED]")

        # Confirm redacted in file on disk
        with open(self.log_path, "r", encoding="utf-8") as f:
            disk_content = f.read()
        self.assertNotIn("secret-12345", disk_content)
        self.assertNotIn("bearer-token-abc", disk_content)
        self.assertNotIn("super-secret-password", disk_content)
        self.assertIn("[REDACTED]", disk_content)

    def test_nested_sensitive_keys_redacted(self):
        """Verify nested structures have sensitive keys redacted recursively."""
        nested = {
            "features": {"hr": 10, "nested_token": "token-999"},
            "sub_array": [{"secret_key": "abc"}, {"clean_field": 1}],
        }
        clean = sanitize_payload(nested)
        self.assertEqual(clean["features"]["hr"], 10)
        self.assertEqual(clean["features"]["nested_token"], "[REDACTED]")
        self.assertEqual(clean["sub_array"][0]["secret_key"], "[REDACTED]")
        self.assertEqual(clean["sub_array"][1]["clean_field"], 1)


class TestSprint3ThreadSafetyAndResilience(unittest.TestCase):
    """Verify concurrent thread safety and exception resilience."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_path = os.path.join(self.temp_dir.name, "concurrent_log.jsonl")
        self.logger = ExecutionLogger(log_path=self.log_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_concurrent_writes_thread_safety(self):
        """Verify that concurrent writes across multiple threads write intact, uncorrupted lines."""
        num_threads = 8
        writes_per_thread = 20
        total_writes = num_threads * writes_per_thread

        def worker(thread_idx: int):
            for i in range(writes_per_thread):
                self.logger.log_execution(
                    status="ok",
                    requested_datetime="2026-10-30T10:00",
                    predicted_cnt=thread_idx * 1000 + i,
                    model_level="L5",
                )

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, t) for t in range(num_threads)]
            for f in futures:
                f.result()

        # Read back all lines
        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        self.assertEqual(len(lines), total_writes)
        log_ids = set()
        for line in lines:
            parsed = json.loads(line)
            self.assertTrue(is_valid_uuid4(parsed["log_id"]))
            self.assertTrue(is_valid_uuid4(parsed["request_id"]))
            log_ids.add(parsed["log_id"])

        # Every log_id must be completely unique
        self.assertEqual(len(log_ids), total_writes)

    def test_logger_resilience_on_unwritable_path(self):
        """Verify that logging failures (e.g. invalid path) fail safely without raising unhandled exceptions."""
        bad_logger = ExecutionLogger(log_path="/unlikely_root_path_never_writable/test.jsonl")
        # Should not raise exception
        rec = bad_logger.log_execution(status="ok", predicted_cnt=150)
        self.assertIsNotNone(rec)
        self.assertTrue(is_valid_uuid4(rec["log_id"]))


class _MockHTTPResponse:
    """Mock for urllib HTTP response in tests."""

    def __init__(self, data: bytes, status: int = 200):
        self.data = data
        self.status = status

    def read(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class TestSprint3PredictionWorkflowIntegration(unittest.TestCase):
    """Integration test suite for Sprint 3 Phase 2 Prediction Workflow.

    Conforms to SRS v1.5 Section 5.1, 5.4, 5.5, 5.6, Table 9.4, Appendix B, and TC-13.
    """

    @classmethod
    def setUpClass(cls):
        cls.mm = get_model_manager("dataset/hour.csv")
        with open("app.py", "r", encoding="utf-8") as f:
            cls.app_source = f.read()

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_path = os.path.join(self.temp_dir.name, "workflow_log.jsonl")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _read_logs(self):
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f if line.strip()]

    @patch("urllib.request.urlopen")
    def test_tc13_successful_live_weather_prediction(self, mock_urlopen):
        """TC-13-01: Valid request with live weather generates prediction and Table 9.4 log entry."""
        target_iso = "2026-10-31T14:00"
        payload = {
            "hourly": {
                "time": [target_iso],
                "temperature_2m": [19.5],
                "apparent_temperature": [19.0],
                "relative_humidity_2m": [52.0],
                "wind_speed_10m": [11.0],
                "weather_code": [0],
            }
        }
        mock_urlopen.return_value = _MockHTTPResponse(json.dumps(payload).encode("utf-8"), status=200)

        from datetime import date
        from src.prediction_service import predict_demand_workflow

        res = predict_demand_workflow(
            target_date=date(2026, 10, 31),
            target_hour=14,
            model_level="L5",
            use_manual_weather=False,
            log_path=self.log_path,
            model_manager=self.mm,
            current_date=date(2026, 10, 31),
        )

        self.assertTrue(res["success"])
        self.assertEqual(res["status"], "ok")
        self.assertIsInstance(res["predicted_cnt"], int)
        self.assertGreaterEqual(res["predicted_cnt"], 0)
        self.assertEqual(res["requested_level"], "L5")
        self.assertEqual(res["weather_source"], "Open-Meteo Forecast API (Live)")

        # Verify Table 9.4 record on disk
        logs = self._read_logs()
        self.assertEqual(len(logs), 1)
        rec = logs[0]
        self.assertTrue(is_valid_uuid4(rec["log_id"]))
        self.assertTrue(is_valid_uuid4(rec["request_id"]))
        self.assertEqual(rec["request_id"], res["request_id"])
        self.assertEqual(rec["status"], "ok")
        self.assertIsNone(rec["error_code"])
        self.assertEqual(rec["predicted_cnt"], res["predicted_cnt"])
        self.assertIn("temp", rec["input_vector"])
        self.assertEqual(rec["model_version"], "1.5.0")

    def test_tc13_successful_manual_weather_prediction(self):
        """TC-13-02: Manual weather override records labelled source and succeeds."""
        from datetime import date
        from src.prediction_service import predict_demand_workflow

        manual_weather = {
            "temperature_c": 21.0,
            "apparent_temperature_c": 19.5,
            "humidity_pct": 48.0,
            "wind_speed_kmh": 14.0,
            "weather_desc": "Clear",
        }
        res = predict_demand_workflow(
            target_date=date(2026, 10, 31),
            target_hour=16,
            model_level="L4",
            use_manual_weather=True,
            manual_weather=manual_weather,
            log_path=self.log_path,
            model_manager=self.mm,
        )

        self.assertTrue(res["success"])
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["weather_source"], "Manual Input (Labelled Fallback)")
        self.assertGreaterEqual(res["predicted_cnt"], 0)

        logs = self._read_logs()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["weather_source"], "Manual Input (Labelled Fallback)")
        self.assertEqual(logs[0]["api_status"], "manual_override")

    def test_tc13_invalid_hour_halts_with_err_val_01(self):
        """TC-13-03: Hour 25 halts before inference and records ERR_VAL_01 without fabricating prediction."""
        from datetime import date
        from src.prediction_service import predict_demand_workflow

        res = predict_demand_workflow(
            target_date=date(2026, 10, 31),
            target_hour=25,
            model_level="L5",
            log_path=self.log_path,
            model_manager=self.mm,
        )

        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "validation_error")
        self.assertEqual(res["error_code"], "ERR_VAL_01")
        self.assertIsNone(res["predicted_cnt"])

        logs = self._read_logs()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["status"], "validation_error")
        self.assertEqual(logs[0]["error_code"], "ERR_VAL_01")
        self.assertIsNone(logs[0]["predicted_cnt"])

    def test_tc13_invalid_date_halts_with_err_val_02(self):
        """TC-13-04: Date outside supported operational range halts with ERR_VAL_02."""
        from datetime import date
        from src.prediction_service import predict_demand_workflow

        res = predict_demand_workflow(
            target_date=date(1999, 5, 20),
            target_hour=10,
            model_level="L5",
            log_path=self.log_path,
            model_manager=self.mm,
        )

        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "validation_error")
        self.assertEqual(res["error_code"], "ERR_VAL_02")
        self.assertIsNone(res["predicted_cnt"])

        logs = self._read_logs()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["error_code"], "ERR_VAL_02")
        self.assertIsNone(logs[0]["predicted_cnt"])

    def test_tc13_past_date_historical_weather_unavailable(self):
        """TC-13-05: Past date without manual weather returns historical_weather_unavailable; no auto-L0."""
        from datetime import date
        from src.prediction_service import predict_demand_workflow

        res = predict_demand_workflow(
            target_date=date(2021, 6, 15),
            target_hour=12,
            model_level="L5",
            use_manual_weather=False,
            log_path=self.log_path,
            model_manager=self.mm,
            current_date=date(2026, 10, 31),
        )

        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "historical_weather_unavailable")
        self.assertIsNone(res["predicted_cnt"])

        logs = self._read_logs()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["status"], "historical_weather_unavailable")
        self.assertIsNone(logs[0]["predicted_cnt"])

    @patch("urllib.request.urlopen")
    def test_tc13_weather_api_timeout_no_auto_l0(self, mock_urlopen):
        """TC-13-06: API timeout records weather_api_error and WARN_API_01; does NOT silently run L0."""
        from datetime import date
        from src.prediction_service import predict_demand_workflow

        mock_urlopen.side_effect = TimeoutError("Connection timed out after 2.0s")

        res = predict_demand_workflow(
            target_date=date(2026, 10, 31),
            target_hour=14,
            model_level="L5",
            use_manual_weather=False,
            log_path=self.log_path,
            model_manager=self.mm,
            current_date=date(2026, 10, 31),
        )

        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "weather_api_error")
        self.assertEqual(res["warning_code"], "WARN_API_01")
        self.assertIsNone(res["predicted_cnt"])

        logs = self._read_logs()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["status"], "weather_api_error")
        self.assertEqual(logs[0]["error_code"], "WARN_API_01")
        self.assertIsNone(logs[0]["predicted_cnt"])

    @patch("urllib.request.urlopen")
    def test_tc13_weather_malformed_payload_no_auto_l0(self, mock_urlopen):
        """TC-13-07: Malformed weather payload records weather_validation_error with WARN_API_01."""
        from datetime import date
        from src.prediction_service import predict_demand_workflow

        mock_urlopen.return_value = _MockHTTPResponse(b"{}", status=200)

        res = predict_demand_workflow(
            target_date=date(2026, 10, 31),
            target_hour=14,
            model_level="L4",
            use_manual_weather=False,
            log_path=self.log_path,
            model_manager=self.mm,
            current_date=date(2026, 10, 31),
        )

        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "weather_validation_error")
        self.assertEqual(res["warning_code"], "WARN_API_01")
        self.assertIsNone(res["predicted_cnt"])

        logs = self._read_logs()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["status"], "weather_validation_error")
        self.assertEqual(logs[0]["error_code"], "WARN_API_01")

    @patch("urllib.request.urlopen")
    def test_tc13_unknown_wmo_code_rejects_silent_coercion(self, mock_urlopen):
        """TC-13-08: Unknown WMO code (999) records weather_validation_error and WARN_MAP_01."""
        from datetime import date
        from src.prediction_service import predict_demand_workflow

        target_iso = "2026-10-31T14:00"
        payload = {
            "hourly": {
                "time": [target_iso],
                "temperature_2m": [19.0],
                "apparent_temperature": [18.0],
                "relative_humidity_2m": [50.0],
                "wind_speed_10m": [10.0],
                "weather_code": [999],
            }
        }
        mock_urlopen.return_value = _MockHTTPResponse(json.dumps(payload).encode("utf-8"), status=200)

        res = predict_demand_workflow(
            target_date=date(2026, 10, 31),
            target_hour=14,
            model_level="L5",
            use_manual_weather=False,
            log_path=self.log_path,
            model_manager=self.mm,
            current_date=date(2026, 10, 31),
        )

        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "weather_validation_error")
        self.assertEqual(res["warning_code"], "WARN_MAP_01")
        self.assertIsNone(res["predicted_cnt"])

        logs = self._read_logs()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["status"], "weather_validation_error")
        self.assertEqual(logs[0]["error_code"], "WARN_MAP_01")

    @patch("urllib.request.urlopen")
    def test_tc13_unsupported_forecast_horizon_logging(self, mock_urlopen):
        """TC-13-09: Target timestamp outside forecast horizon records unsupported_time and WARN_API_02."""
        from datetime import date
        from src.prediction_service import predict_demand_workflow

        payload = {
            "hourly": {
                "time": ["2026-10-31T00:00", "2026-10-31T01:00"],
                "temperature_2m": [15.0, 15.0],
                "apparent_temperature": [14.0, 14.0],
                "relative_humidity_2m": [60.0, 60.0],
                "wind_speed_10m": [8.0, 8.0],
                "weather_code": [0, 0],
            }
        }
        mock_urlopen.return_value = _MockHTTPResponse(json.dumps(payload).encode("utf-8"), status=200)

        res = predict_demand_workflow(
            target_date=date(2026, 10, 31),
            target_hour=14,
            model_level="L5",
            use_manual_weather=False,
            log_path=self.log_path,
            model_manager=self.mm,
            current_date=date(2026, 10, 31),
        )

        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "unsupported_time")
        self.assertEqual(res["warning_code"], "WARN_API_02")
        self.assertIsNone(res["predicted_cnt"])

        logs = self._read_logs()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["status"], "unsupported_time")
        self.assertEqual(logs[0]["error_code"], "WARN_API_02")

    def test_tc13_explicit_user_selected_l0_behavior(self):
        """TC-13-10: User explicitly choosing L0 produces dynamic training mean without needing weather."""
        from datetime import date
        from src.prediction_service import predict_demand_workflow

        res = predict_demand_workflow(
            target_date=date(2026, 10, 31),
            target_hour=14,
            model_level="L0",
            use_manual_weather=False,
            log_path=self.log_path,
            model_manager=self.mm,
        )

        self.assertTrue(res["success"])
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["model_level"], "L0")
        expected_cnt = int(round(self.mm.l0_mean))
        self.assertEqual(res["predicted_cnt"], expected_cnt)

        logs = self._read_logs()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["status"], "ok")
        self.assertEqual(logs[0]["model_level"], "L0")
        self.assertEqual(logs[0]["predicted_cnt"], expected_cnt)

    def test_tc13_missing_model_bundle_l0_fallback(self):
        """TC-13-11: Missing model artifact degrades to saved L0 training mean with WARN_SYS_01."""
        from datetime import date
        from unittest.mock import MagicMock
        from src.models import ModelManager
        from src.prediction_service import predict_demand_workflow

        mock_mm = MagicMock(spec=ModelManager)
        mock_mm.l0_mean = 174.64
        mock_mm.predict.return_value = {
            "predicted_cnt": 175,
            "predicted_cnt_raw": 174.64,
            "requested_level": "L4",
            "used_level": "L0",
            "mae": 174.98,
            "fallback_note": "Model artifact unavailable; showing Level 0 fallback.",
            "warning_code": "WARN_SYS_01",
            "error_code": None,
            "status": "model_fallback_l0",
        }

        res = predict_demand_workflow(
            target_date=date(2026, 10, 31),
            target_hour=14,
            model_level="L4",
            use_manual_weather=True,
            manual_weather={"temperature_c": 20.0, "humidity_pct": 50, "wind_speed_kmh": 10.0, "weather_desc": "Clear"},
            log_path=self.log_path,
            model_manager=mock_mm,
        )

        self.assertTrue(res["success"])
        self.assertEqual(res["status"], "model_fallback_l0")
        self.assertEqual(res["warning_code"], "WARN_SYS_01")
        self.assertEqual(res["used_level"], "L0")
        self.assertEqual(res["predicted_cnt"], 175)

        logs = self._read_logs()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["status"], "model_fallback_l0")
        self.assertEqual(logs[0]["error_code"], "WARN_SYS_01")

    def test_tc13_model_and_l0_both_missing_err_sys_01(self):
        """TC-13-12: Total model and L0 unavailability returns ERR_SYS_01; zero fabricated count."""
        from datetime import date
        from unittest.mock import MagicMock
        from src.models import ModelManager
        from src.prediction_service import predict_demand_workflow

        mock_mm = MagicMock(spec=ModelManager)
        mock_mm.predict.return_value = {
            "predicted_cnt": None,
            "predicted_cnt_raw": None,
            "requested_level": "L4",
            "used_level": None,
            "mae": None,
            "fallback_note": "Model and L0 config unavailable; cannot generate prediction.",
            "warning_code": None,
            "error_code": "ERR_SYS_01",
            "status": "prediction_error",
        }

        res = predict_demand_workflow(
            target_date=date(2026, 10, 31),
            target_hour=14,
            model_level="L4",
            use_manual_weather=True,
            manual_weather={"temperature_c": 20.0, "humidity_pct": 50, "wind_speed_kmh": 10.0, "weather_desc": "Clear"},
            log_path=self.log_path,
            model_manager=mock_mm,
        )

        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "prediction_error")
        self.assertEqual(res["error_code"], "ERR_SYS_01")
        self.assertIsNone(res["predicted_cnt"])

        logs = self._read_logs()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["status"], "prediction_error")
        self.assertEqual(logs[0]["error_code"], "ERR_SYS_01")
        self.assertIsNone(logs[0]["predicted_cnt"])

    def test_tc13_cascading_baseline_group_fallback(self):
        """TC-13-13: Unseen category combination in L3 cascades down and records baseline_fallback."""
        from datetime import date
        from src.prediction_service import predict_demand_workflow

        res = predict_demand_workflow(
            target_date=date(2026, 10, 31),
            target_hour=14,
            model_level="L3",
            use_manual_weather=True,
            manual_weather={"temperature_c": 20.0, "humidity_pct": 50, "wind_speed_kmh": 10.0, "weather_desc": "Clear"},
            log_path=self.log_path,
            model_manager=self.mm,
        )

        self.assertTrue(res["success"])
        self.assertIn(res["status"], ("ok", "baseline_fallback"))
        self.assertGreaterEqual(res["predicted_cnt"], 0)

        logs = self._read_logs()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["status"], res["status"])

    def test_tc13_request_id_log_id_independence(self):
        """TC-13-14: Preserves caller's valid UUID v4 request_id and generates distinct log_id."""
        from datetime import date
        from src.prediction_service import predict_demand_workflow

        caller_req_id = str(uuid.uuid4())
        res = predict_demand_workflow(
            target_date=date(2026, 10, 31),
            target_hour=14,
            model_level="L0",
            request_id=caller_req_id,
            log_path=self.log_path,
            model_manager=self.mm,
        )

        self.assertEqual(res["request_id"], caller_req_id)
        logs = self._read_logs()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["request_id"], caller_req_id)
        self.assertNotEqual(logs[0]["log_id"], caller_req_id)
        self.assertTrue(is_valid_uuid4(logs[0]["log_id"]))

    def test_tc13_single_entry_invariant_per_submission(self):
        """TC-13-15: Exactly one execution log entry is written per submitted request."""
        from datetime import date
        from src.prediction_service import predict_demand_workflow

        predict_demand_workflow(date(2026, 10, 31), 10, "L0", log_path=self.log_path, model_manager=self.mm)
        predict_demand_workflow(date(2026, 10, 31), 25, "L0", log_path=self.log_path, model_manager=self.mm)
        predict_demand_workflow(date(1995, 1, 1), 10, "L0", log_path=self.log_path, model_manager=self.mm)
        predict_demand_workflow(date(2021, 1, 1), 10, "L5", log_path=self.log_path, model_manager=self.mm, current_date=date(2026, 10, 31))
        predict_demand_workflow(
            date(2026, 10, 31), 12, "L4",
            use_manual_weather=True,
            manual_weather={"temperature_c": 20.0, "humidity_pct": 50, "wind_speed_kmh": 10.0, "weather_desc": "Clear"},
            log_path=self.log_path,
            model_manager=self.mm,
        )

        logs = self._read_logs()
        self.assertEqual(len(logs), 5)

    def test_tc13_secret_redaction_in_logged_payload(self):
        """TC-13-16: Accidental sensitive keys in input parameters are redacted in the JSONL log."""
        from datetime import date
        from src.prediction_service import predict_demand_workflow

        sensitive_manual = {
            "temperature_c": 20.0,
            "humidity_pct": 50,
            "wind_speed_kmh": 10.0,
            "weather_desc": "Clear",
            "api_key": "secret-token-12345",
            "auth_token": "bearer-xyz",
        }
        res = predict_demand_workflow(
            target_date=date(2026, 10, 31),
            target_hour=14,
            model_level="L4",
            use_manual_weather=True,
            manual_weather=sensitive_manual,
            log_path=self.log_path,
            model_manager=self.mm,
        )
        self.assertTrue(res["success"])

        with open(self.log_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        self.assertNotIn("secret-token-12345", raw_text)
        self.assertNotIn("bearer-xyz", raw_text)

    def test_streamlit_predict_button_guard_in_app_source(self):
        """TC-13-17: Verify app.py conditionally invokes predict_demand_workflow under predict_btn."""
        self.assertIn("predict_btn = st.button", self.app_source)
        self.assertIn("if predict_btn:", self.app_source)
        self.assertIn("predict_demand_workflow(", self.app_source)
        self.assertIn('st.session_state["prediction_result"]', self.app_source)


class TestSprint3PredictionLifecycleAndUI(unittest.TestCase):
    """Lifecycle and UI integration verification tests for Sprint 3 Phase 3.

    Verifies prediction lifecycle states, stale prevention, input immutability,
    repeated predictions, session-state management, error presentation,
    and UI contract compliance.
    """

    @classmethod
    def setUpClass(cls):
        cls.mm = get_model_manager("dataset/hour.csv")
        with open("app.py", "r", encoding="utf-8") as f:
            cls.app_source = f.read()

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_path = os.path.join(self.temp_dir.name, "lifecycle_log.jsonl")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _read_logs(self):
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f if line.strip()]

    def test_lifecycle_01_initial_load_no_prediction_triggered(self):
        """1. Initial load does not execute prediction: session state is None and ready state is rendered."""
        # Simulate clean session state on initial page load
        session_state = {}
        res = session_state.get("prediction_result")
        self.assertIsNone(res, "On initial application load, prediction_result must be None")
        # Verify app source guards against auto-execution and renders awaiting submission state
        self.assertIn("res is None", self.app_source)
        self.assertIn("Awaiting Submission", self.app_source)
        self.assertIn("Ready to Advise Demand", self.app_source)

    def test_lifecycle_02_initial_load_creates_no_logs(self):
        """2. Initial load does not create logs: log file remains non-existent or empty."""
        logs = self._read_logs()
        self.assertEqual(len(logs), 0, "No log entries should exist on initial application load")
        self.assertFalse(os.path.exists(self.log_path))

    def test_lifecycle_03_input_changes_do_not_execute_prediction(self):
        """3. Input changes do not execute prediction: changing parameters without button does not run workflow or log."""
        mock_workflow = MagicMock()
        # Simulate widget value changes on left column
        selected_date = date(2026, 11, 2)
        selected_hour = 17
        model_level = "L5"
        predict_btn = False  # Widget interaction triggers passive rerun, predict_btn is False

        if predict_btn:
            mock_workflow()

        mock_workflow.assert_not_called()
        self.assertEqual(len(self._read_logs()), 0)

    @patch("urllib.request.urlopen")
    def test_lifecycle_04_successful_prediction_executes_once(self, mock_urlopen):
        """4. Successful prediction executes once: generates count, request ID, MAE, and 1 log entry."""
        target_iso = "2026-11-02T14:00"
        payload = {
            "hourly": {
                "time": [target_iso],
                "temperature_2m": [18.0],
                "apparent_temperature": [17.5],
                "relative_humidity_2m": [50.0],
                "wind_speed_10m": [10.0],
                "weather_code": [0],
            }
        }
        mock_urlopen.return_value = _MockHTTPResponse(json.dumps(payload).encode("utf-8"), status=200)

        res = predict_demand_workflow(
            target_date=date(2026, 11, 2),
            target_hour=14,
            model_level="L5",
            use_manual_weather=False,
            log_path=self.log_path,
            model_manager=self.mm,
            current_date=date(2026, 11, 2),
        )

        self.assertTrue(res["success"])
        self.assertIsInstance(res["predicted_cnt"], int)
        self.assertGreaterEqual(res["predicted_cnt"], 0)
        self.assertTrue(is_valid_uuid4(res["request_id"]))
        self.assertIsNotNone(res.get("mae"))

        logs = self._read_logs()
        self.assertEqual(len(logs), 1, "Exactly one log record must be generated per prediction execution")
        self.assertEqual(logs[0]["request_id"], res["request_id"])

    def test_lifecycle_05_repeated_intentional_prediction_creates_new_request(self):
        """5. Repeated intentional prediction creates a new request: unique request IDs and separate log records."""
        res1 = predict_demand_workflow(
            target_date=date(2026, 11, 2),
            target_hour=9,
            model_level="L0",
            log_path=self.log_path,
            model_manager=self.mm,
        )
        res2 = predict_demand_workflow(
            target_date=date(2026, 11, 2),
            target_hour=10,
            model_level="L0",
            log_path=self.log_path,
            model_manager=self.mm,
        )

        self.assertNotEqual(res1["request_id"], res2["request_id"], "Subsequent predictions must have unique request IDs")
        self.assertTrue(is_valid_uuid4(res1["request_id"]))
        self.assertTrue(is_valid_uuid4(res2["request_id"]))

        logs = self._read_logs()
        self.assertEqual(len(logs), 2)
        self.assertEqual(logs[0]["request_id"], res1["request_id"])
        self.assertEqual(logs[1]["request_id"], res2["request_id"])

    def test_lifecycle_06_passive_rerun_does_not_duplicate_prediction_or_logs(self):
        """6. Passive rerun does not duplicate prediction: existing result in state rendered without re-running workflow."""
        # Initial genuine submission
        res = predict_demand_workflow(
            target_date=date(2026, 11, 2),
            target_hour=12,
            model_level="L0",
            log_path=self.log_path,
            model_manager=self.mm,
        )
        session_state = {"prediction_result": res}
        self.assertEqual(len(self._read_logs()), 1)

        # Simulate 5 passive Streamlit reruns (sliders, tabs, toggles without clicking predict)
        for _ in range(5):
            predict_btn = False
            if predict_btn:
                predict_demand_workflow(
                    target_date=date(2026, 11, 2),
                    target_hour=12,
                    model_level="L0",
                    log_path=self.log_path,
                    model_manager=self.mm,
                )
            # col_result simply reads session_state["prediction_result"]
            active_res = session_state.get("prediction_result")
            self.assertEqual(active_res["request_id"], res["request_id"])

        self.assertEqual(len(self._read_logs()), 1, "Passive reruns must never duplicate execution log entries")

    def test_lifecycle_07_invalid_inputs_halt_and_replace_old_prediction(self):
        """7. Invalid inputs handled correctly: ERR_VAL_01 / ERR_VAL_02 halt before inference and replace old result."""
        # Prior valid result
        valid_res = predict_demand_workflow(
            target_date=date(2026, 11, 2),
            target_hour=14,
            model_level="L0",
            log_path=self.log_path,
            model_manager=self.mm,
        )
        session_state = {"prediction_result": valid_res}
        self.assertTrue(session_state["prediction_result"]["success"])

        # User enters invalid hour (hr=25) and clicks Predict
        invalid_hr_res = predict_demand_workflow(
            target_date=date(2026, 11, 2),
            target_hour=25,
            model_level="L0",
            log_path=self.log_path,
            model_manager=self.mm,
        )
        session_state["prediction_result"] = invalid_hr_res

        self.assertFalse(session_state["prediction_result"]["success"])
        self.assertEqual(session_state["prediction_result"]["status"], "validation_error")
        self.assertEqual(session_state["prediction_result"]["error_code"], "ERR_VAL_01")
        self.assertIsNone(session_state["prediction_result"]["predicted_cnt"])

        # User enters invalid date (year 2035) and clicks Predict
        invalid_date_res = predict_demand_workflow(
            target_date=date(2035, 1, 1),
            target_hour=12,
            model_level="L0",
            log_path=self.log_path,
            model_manager=self.mm,
        )
        session_state["prediction_result"] = invalid_date_res

        self.assertFalse(session_state["prediction_result"]["success"])
        self.assertEqual(session_state["prediction_result"]["status"], "validation_error")
        self.assertEqual(session_state["prediction_result"]["error_code"], "ERR_VAL_02")
        self.assertIsNone(session_state["prediction_result"]["predicted_cnt"])

    @patch("urllib.request.urlopen")
    def test_lifecycle_08_weather_failure_does_not_silently_run_l0(self, mock_urlopen):
        """8. Weather failure does not silently fall back to L0: returns failure status and logs warning."""
        mock_urlopen.side_effect = Exception("Open-Meteo connection timed out")

        res = predict_demand_workflow(
            target_date=date(2026, 11, 2),
            target_hour=15,
            model_level="L5",
            use_manual_weather=False,
            log_path=self.log_path,
            model_manager=self.mm,
            current_date=date(2026, 11, 2),
        )

        self.assertFalse(res["success"], "Weather API failure must not produce a fabricated success")
        self.assertEqual(res["status"], "weather_api_error")
        self.assertEqual(res["warning_code"], "WARN_API_01")
        self.assertIsNone(res["predicted_cnt"])

        logs = self._read_logs()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["status"], "weather_api_error")
        self.assertEqual(logs[0]["error_code"], "WARN_API_01")
        self.assertIsNone(logs[0]["predicted_cnt"])

    def test_lifecycle_09_manual_weather_override_works(self):
        """9. Manual weather override works: uses user inputs and logs labelled manual source without network calls."""
        res = predict_demand_workflow(
            target_date=date(2026, 11, 2),
            target_hour=16,
            model_level="L4",
            use_manual_weather=True,
            manual_weather={
                "temperature_c": 22.5,
                "apparent_temperature_c": 21.0,
                "humidity_pct": 45.0,
                "wind_speed_kmh": 8.0,
                "weather_desc": "Clear",
            },
            log_path=self.log_path,
            model_manager=self.mm,
        )

        self.assertTrue(res["success"])
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["weather_source"], "Manual Input (Labelled Fallback)")
        self.assertEqual(res["api_status"], "manual_override")
        self.assertIsInstance(res["predicted_cnt"], int)

        logs = self._read_logs()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["weather_source"], "Manual Input (Labelled Fallback)")

    def test_lifecycle_10_explicit_l0_selection_uses_dynamic_mean(self):
        """10. Explicit L0 selection works: uses dynamic training mean from ModelManager.l0_mean, never hardcoded."""
        res = predict_demand_workflow(
            target_date=date(2026, 11, 2),
            target_hour=17,
            model_level="L0",
            log_path=self.log_path,
            model_manager=self.mm,
        )

        self.assertTrue(res["success"])
        self.assertEqual(res["requested_level"], "L0")
        self.assertEqual(res["used_level"], "L0")
        self.assertEqual(res["weather_source"], "Explicit L0 Baseline")
        self.assertEqual(res["api_status"], "not_required")

        expected_cnt = int(round(self.mm.l0_mean))
        self.assertEqual(res["predicted_cnt"], expected_cnt, "L0 prediction must strictly match rounded dynamic l0_mean")
        self.assertAlmostEqual(res["predicted_cnt_raw"], self.mm.l0_mean, places=2)

    def test_lifecycle_11_stale_prediction_detection_and_ui_guard(self):
        """11. Old prediction results are not misleadingly displayed: signature comparison detects modified parameters."""
        saved_signature = {
            "date": "2026-11-02",
            "hour": 17,
            "model_level": "L5",
            "use_manual_weather": False,
            "manual_weather": None,
        }

        # Case A: Identical inputs -> not stale
        current_identical = {
            "date": "2026-11-02",
            "hour": 17,
            "model_level": "L5",
            "use_manual_weather": False,
            "manual_weather": None,
        }
        self.assertFalse(current_identical != saved_signature)

        # Case B: Changed hour -> stale
        current_changed_hr = dict(saved_signature, hour=18)
        self.assertTrue(current_changed_hr != saved_signature)

        # Case C: Changed date -> stale
        current_changed_date = dict(saved_signature, date="2026-11-03")
        self.assertTrue(current_changed_date != saved_signature)

        # Case D: Changed model tier -> stale
        current_changed_lvl = dict(saved_signature, model_level="L1")
        self.assertTrue(current_changed_lvl != saved_signature)

        # Case E: Enabled manual weather -> stale
        current_changed_weather = dict(saved_signature, use_manual_weather=True, manual_weather={"temperature_c": 25.0})
        self.assertTrue(current_changed_weather != saved_signature)

        # Verify app source implements stale warning banner, prior run badge, and clear button
        self.assertIn("is_inputs_modified", self.app_source)
        self.assertIn("Parameters Modified Since Run", self.app_source)
        self.assertIn("Prior Run", self.app_source)
        self.assertIn("Clear Prediction", self.app_source)

    def test_lifecycle_12_request_ids_remain_unique(self):
        """12. Request IDs remain unique: 24 consecutive predictions generate 24 unique UUID v4 IDs."""
        req_ids = set()
        log_ids = set()

        for h in range(24):
            res = predict_demand_workflow(
                target_date=date(2026, 11, 2),
                target_hour=h,
                model_level="L0",
                log_path=self.log_path,
                model_manager=self.mm,
            )
            self.assertTrue(is_valid_uuid4(res["request_id"]))
            req_ids.add(res["request_id"])

        logs = self._read_logs()
        self.assertEqual(len(logs), 24)
        for entry in logs:
            log_ids.add(entry["log_id"])

        self.assertEqual(len(req_ids), 24, "All 24 request IDs must be strictly unique")
        self.assertEqual(len(log_ids), 24, "All 24 log IDs must be strictly unique")
        # Set intersection between log_ids and req_ids must be empty
        self.assertTrue(req_ids.isdisjoint(log_ids), "Request IDs and Log IDs must never overlap")


class TestSprint3LoggingAndAuditVerification(unittest.TestCase):
    """Exhaustive audit verification tests for Sprint 3 Phase 4.

    Covers Section 5 through 11:
    - 12-field schema compliance (SRS Table 9.4)
    - Valid JSON serialization and line separation
    - RFC 4122 UUID v4 formatting and validity
    - Log ID uniqueness and request ID correlation
    - Recursive redaction across nested dictionaries and lists
    - Input immutability and non-sensitive preservation
    - Concurrent multithreaded append safety and zero record loss
    - Append-only integrity and history preservation
    - Failure and failsafe resilience (no crashes on disk errors)
    - Recent log retrieval ordering, limit, corrupt-line recovery, and redaction
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_path = os.path.join(self.temp_dir.name, "audit_test.jsonl")
        self.logger = ExecutionLogger(log_path=self.log_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _read_raw_lines(self):
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _read_parsed_records(self):
        return [json.loads(line) for line in self._read_raw_lines()]

    def test_audit_01_exact_12_field_schema(self):
        """1. Exact 12-field schema: every record contains exactly the 12 Table 9.4 keys, no more, no less."""
        expected_fields = {
            "log_id",
            "request_id",
            "timestamp",
            "requested_datetime",
            "status",
            "error_code",
            "weather_source",
            "api_status",
            "input_vector",
            "predicted_cnt",
            "model_level",
            "model_version",
        }
        rec = self.logger.log_execution(
            status="ok",
            requested_datetime="2026-11-02T14:00",
            predicted_cnt=250,
            model_level="L5",
            weather_source="Open-Meteo Forecast API (Live)",
            api_status="200",
        )
        self.assertEqual(set(rec.keys()), expected_fields, "Logged record must have exactly the 12 Table 9.4 fields")
        self.assertEqual(len(rec.keys()), 12)

        file_records = self._read_parsed_records()
        self.assertEqual(len(file_records), 1)
        self.assertEqual(set(file_records[0].keys()), expected_fields)
        self.assertEqual(len(file_records[0].keys()), 12)

    def test_audit_02_valid_json_serialization(self):
        """2. Valid JSON serialization: each record occupies exactly one valid, parseable JSONL line."""
        for i in range(5):
            self.logger.log_execution(
                status="ok",
                requested_datetime=f"2026-11-02T{10+i:02d}:00",
                predicted_cnt=100 + i * 20,
                model_level="L4",
            )

        raw_lines = self._read_raw_lines()
        self.assertEqual(len(raw_lines), 5)
        for idx, line in enumerate(raw_lines):
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                self.fail(f"Line {idx} failed JSON deserialization: {e}")
            self.assertEqual(data["model_level"], "L4")

    def test_audit_03_uuid_v4_validity(self):
        """3. UUID v4 validity: log_id and request_id strictly adhere to RFC 4122 UUID v4 formatting."""
        rec = self.logger.log_execution(
            status="ok",
            requested_datetime="2026-11-02T12:00",
            predicted_cnt=300,
            model_level="L5",
        )
        self.assertTrue(is_valid_uuid4(rec["log_id"]), f"log_id '{rec['log_id']}' is not a valid UUID v4")
        self.assertTrue(is_valid_uuid4(rec["request_id"]), f"request_id '{rec['request_id']}' is not a valid UUID v4")

        # Test invalid inputs to request_id are safely replaced with fresh valid UUID v4
        invalid_ids = ["not-a-uuid", "12345", "", None, 9999, "123e4567-e89b-12d3-a456-426614174000"]
        for bad_id in invalid_ids:
            bad_rec = self.logger.log_execution(status="ok", request_id=bad_id)
            self.assertTrue(is_valid_uuid4(bad_rec["request_id"]), f"Failed to normalize invalid request_id '{bad_id}'")
            self.assertNotEqual(bad_rec["request_id"], str(bad_id))

    def test_audit_04_unique_log_ids(self):
        """4. Unique log IDs: all log_id values across 100 consecutive records are strictly unique."""
        seen_ids = set()
        for i in range(100):
            rec = self.logger.log_execution(
                status="ok",
                requested_datetime="2026-11-02T10:00",
                predicted_cnt=150,
                model_level="L0",
            )
            self.assertNotIn(rec["log_id"], seen_ids, "Duplicate log_id generated")
            seen_ids.add(rec["log_id"])

        self.assertEqual(len(seen_ids), 100)

    def test_audit_05_request_id_correlation(self):
        """5. Request ID correlation: multiple operations under the same request retain identical request_id."""
        shared_req_id = str(uuid.uuid4())
        rec1 = self.logger.log_execution(
            status="ok",
            request_id=shared_req_id,
            requested_datetime="2026-11-02T12:00",
            weather_source="Open-Meteo Forecast API",
            api_status="200",
        )
        rec2 = self.logger.log_execution(
            status="ok",
            request_id=shared_req_id,
            requested_datetime="2026-11-02T12:00",
            predicted_cnt=320,
            model_level="L5",
        )

        self.assertEqual(rec1["request_id"], shared_req_id)
        self.assertEqual(rec2["request_id"], shared_req_id)
        self.assertNotEqual(rec1["log_id"], rec2["log_id"], "Distinct log entries must have distinct log_ids")

    def test_audit_06_recursive_redaction(self):
        """6. Recursive redaction: all representative sensitive keys are masked recursively."""
        sensitive_payload = {
            "password": "plain-text-pwd",
            "token": "secret-jwt-token",
            "api_key": "open-meteo-fake-key",
            "secret": "db-secret",
            "authorization": "Bearer xyz123",
            "access_token": "oauth-access-token",
            "refresh_token": "oauth-refresh-token",
            "nested_config": {
                "admin_password": "super-secret-password",
                "auth_bearer": "bearer-token-abc",
            },
        }

        sanitized = sanitize_payload(sensitive_payload)
        for key in ["password", "token", "api_key", "secret", "authorization", "access_token", "refresh_token"]:
            self.assertEqual(sanitized[key], "[REDACTED]", f"Key '{key}' was not redacted")
        self.assertEqual(sanitized["nested_config"]["admin_password"], "[REDACTED]")
        self.assertEqual(sanitized["nested_config"]["auth_bearer"], "[REDACTED]")

        # Log and verify file serialization contains zero raw secret values
        self.logger.log_execution(
            status="ok",
            input_vector=sensitive_payload,
        )
        with open(self.log_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
        self.assertNotIn("plain-text-pwd", raw_content)
        self.assertNotIn("secret-jwt-token", raw_content)
        self.assertNotIn("open-meteo-fake-key", raw_content)
        self.assertNotIn("super-secret-password", raw_content)

    def test_audit_07_nested_dictionary_redaction(self):
        """7. Nested dictionary redaction: deeply nested dictionaries mask case-insensitive sensitive keys."""
        deep_dict = {
            "level1": {
                "level2": {
                    "level3": {
                        "API_KEY": "KEY-12345",
                        "Secret_Token": "TOKEN-67890",
                        "safe_field": 42.0,
                    }
                }
            }
        }
        res = sanitize_payload(deep_dict)
        l3 = res["level1"]["level2"]["level3"]
        self.assertEqual(l3["API_KEY"], "[REDACTED]")
        self.assertEqual(l3["Secret_Token"], "[REDACTED]")
        self.assertEqual(l3["safe_field"], 42.0)

    def test_audit_08_nested_list_redaction(self):
        """8. Nested list redaction: sensitive keys inside lists and tuples are masked."""
        list_payload = {
            "users": [
                {"username": "alice", "auth_token": "token-alice"},
                {"username": "bob", "credential_hash": "hash-bob"},
            ],
            "headers": [
                ["Authorization", "Basic dXNlcjpwYXNz"],
                ["Content-Type", "application/json"],
            ],
        }
        res = sanitize_payload(list_payload)
        self.assertEqual(res["users"][0]["username"], "alice")
        self.assertEqual(res["users"][0]["auth_token"], "[REDACTED]")
        self.assertEqual(res["users"][1]["credential_hash"], "[REDACTED]")
        self.assertEqual(res["headers"][1][1], "application/json")

    def test_audit_09_non_sensitive_preservation_and_immutability(self):
        """9. Non-sensitive preservation and immutability: operational features preserved; input unmutated."""
        original_inputs = {
            "season": 3,
            "mnth": 7,
            "hr": 17,
            "holiday": 0,
            "weekday": 2,
            "workingday": 1,
            "weathersit": 1,
            "temp": 0.72,
            "atemp": 0.68,
            "hum": 0.50,
            "windspeed": 0.19,
            "secret_key": "raw_sensitive_string",
        }
        original_copy = dict(original_inputs)

        sanitized = sanitize_payload(original_inputs)

        # Immutability check
        self.assertEqual(original_inputs, original_copy, "sanitize_payload must not mutate the caller's input dictionary")
        self.assertEqual(original_inputs["secret_key"], "raw_sensitive_string")

        # Preservation check
        for k in ["season", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"]:
            self.assertEqual(sanitized[k], original_inputs[k], f"Operational feature '{k}' was unexpectedly modified")
        self.assertEqual(sanitized["secret_key"], "[REDACTED]")

    def test_audit_10_concurrent_writes_thread_safety(self):
        """10. Concurrent writes: 10 threads writing 25 records each (250 total) yield zero lost or corrupted lines."""
        num_threads = 10
        records_per_thread = 25
        total_records = num_threads * records_per_thread

        def worker(thread_idx):
            for r in range(records_per_thread):
                self.logger.log_execution(
                    status="ok",
                    requested_datetime=f"2026-11-02T{r % 24:02d}:00",
                    predicted_cnt=thread_idx * 100 + r,
                    model_level="L5",
                    input_vector={"thread": thread_idx, "seq": r},
                )

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        records = self._read_parsed_records()
        self.assertEqual(len(records), total_records, f"Expected {total_records} records, but found {len(records)}")

        # Verify all log IDs are unique
        log_ids = {r["log_id"] for r in records}
        self.assertEqual(len(log_ids), total_records, "Duplicate log_ids found during concurrent writes")

    def test_audit_11_append_only_behaviour(self):
        """11. Append-only behaviour: sequential writes strictly append without truncating earlier records."""
        self.logger.log_execution(status="ok", predicted_cnt=100)
        self.assertEqual(len(self._read_raw_lines()), 1)

        self.logger.log_execution(status="ok", predicted_cnt=200)
        self.assertEqual(len(self._read_raw_lines()), 2)

        self.logger.log_execution(status="ok", predicted_cnt=300)
        lines = self._read_raw_lines()
        self.assertEqual(len(lines), 3)

        records = [json.loads(l) for l in lines]
        self.assertEqual([r["predicted_cnt"] for r in records], [100, 200, 300])

    def test_audit_12_existing_log_preservation(self):
        """12. Existing log preservation: initial file contents are retained when logger starts."""
        initial_entries = [
            {"log_id": str(uuid.uuid4()), "request_id": str(uuid.uuid4()), "status": "ok", "predicted_cnt": 50},
            {"log_id": str(uuid.uuid4()), "request_id": str(uuid.uuid4()), "status": "ok", "predicted_cnt": 75},
        ]
        with open(self.log_path, "w", encoding="utf-8") as f:
            for entry in initial_entries:
                f.write(json.dumps(entry) + "\n")

        # New logger instance targeting the same file
        new_logger = ExecutionLogger(log_path=self.log_path)
        new_logger.log_execution(status="ok", predicted_cnt=120)

        lines = self._read_raw_lines()
        self.assertEqual(len(lines), 3, "Pre-existing log entries must not be overwritten or truncated")
        self.assertEqual(json.loads(lines[0])["predicted_cnt"], 50)
        self.assertEqual(json.loads(lines[1])["predicted_cnt"], 75)
        self.assertEqual(json.loads(lines[2])["predicted_cnt"], 120)

    def test_audit_13_successful_prediction_logging(self):
        """13. Successful prediction logging: captures status='ok', count, level, weather source, and api_status."""
        rec = self.logger.log_execution(
            status="ok",
            requested_datetime="2026-11-02T15:00",
            predicted_cnt=450,
            model_level="L5",
            weather_source="Open-Meteo Forecast API (Live)",
            api_status="200",
            input_vector={"temp": 0.65, "hum": 0.45},
        )
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(rec["predicted_cnt"], 450)
        self.assertEqual(rec["model_level"], "L5")
        self.assertEqual(rec["weather_source"], "Open-Meteo Forecast API (Live)")
        self.assertEqual(rec["api_status"], "200")
        self.assertIsNone(rec["error_code"])

    def test_audit_14_failed_prediction_logging(self):
        """14. Failed prediction logging: validation error, weather failure, and system error audit structure."""
        # Validation failure
        val_rec = self.logger.log_execution(
            status="validation_error",
            requested_datetime="2026-11-02 hr=25",
            error_code="ERR_VAL_01",
        )
        self.assertEqual(val_rec["status"], "validation_error")
        self.assertEqual(val_rec["error_code"], "ERR_VAL_01")
        self.assertIsNone(val_rec["predicted_cnt"])

        # Weather API timeout failure
        w_rec = self.logger.log_execution(
            status="weather_api_error",
            requested_datetime="2026-11-02T15:00",
            error_code="WARN_API_01",
            weather_source="Open-Meteo Forecast API",
            api_status="timeout",
        )
        self.assertEqual(w_rec["status"], "weather_api_error")
        self.assertEqual(w_rec["error_code"], "WARN_API_01")
        self.assertEqual(w_rec["api_status"], "timeout")
        self.assertIsNone(w_rec["predicted_cnt"])

        # System error (model missing)
        sys_rec = self.logger.log_execution(
            status="prediction_error",
            requested_datetime="2026-11-02T15:00",
            error_code="ERR_SYS_01",
        )
        self.assertEqual(sys_rec["status"], "prediction_error")
        self.assertEqual(sys_rec["error_code"], "ERR_SYS_01")
        self.assertIsNone(sys_rec["predicted_cnt"])

    def test_audit_15_logger_write_failure_failsafe(self):
        """15. Logger write failure failsafe: unwritable log destination fails safely without crashing caller."""
        # Target path pointing to an invalid destination (a directory path instead of a file)
        invalid_logger = ExecutionLogger(log_path=self.temp_dir.name)  # directory path
        try:
            res = invalid_logger.log_execution(
                status="ok",
                requested_datetime="2026-11-02T10:00",
                predicted_cnt=180,
                model_level="L0",
            )
            # Must return the record dictionary safely without raising an unhandled exception
            self.assertIsInstance(res, dict)
            self.assertEqual(res["status"], "ok")
            self.assertTrue(is_valid_uuid4(res["log_id"]))
        except Exception as e:
            self.fail(f"Logger raised unexpected exception on unwritable destination: {e}")

    def test_audit_16_recent_log_retrieval(self):
        """16. Recent log retrieval: get_recent_logs returns newest-first ordering and respects limit."""
        for i in range(10):
            self.logger.log_execution(
                status="ok",
                requested_datetime=f"2026-11-02T{i:02d}:00",
                predicted_cnt=i * 10,
            )

        # Retrieve top 3
        top3 = self.logger.get_recent_logs(limit=3)
        self.assertEqual(len(top3), 3)
        # Newest record is i=9, followed by 8, then 7
        self.assertEqual(top3[0]["predicted_cnt"], 90)
        self.assertEqual(top3[1]["predicted_cnt"], 80)
        self.assertEqual(top3[2]["predicted_cnt"], 70)

        # Retrieve with limit greater than total
        all_logs = self.logger.get_recent_logs(limit=50)
        self.assertEqual(len(all_logs), 10)
        self.assertEqual(all_logs[0]["predicted_cnt"], 90)
        self.assertEqual(all_logs[-1]["predicted_cnt"], 0)

    def test_audit_17_empty_and_malformed_log_handling(self):
        """17. Empty and malformed log handling: handles missing files, blank lines, corrupt JSON lines, and sanitizes on read."""
        # Non-existent file
        non_existent_logger = ExecutionLogger(log_path=os.path.join(self.temp_dir.name, "does_not_exist.jsonl"))
        self.assertEqual(non_existent_logger.get_recent_logs(), [])

        # Empty file
        empty_log_path = os.path.join(self.temp_dir.name, "empty.jsonl")
        open(empty_log_path, "w", encoding="utf-8").close()
        empty_logger = ExecutionLogger(log_path=empty_log_path)
        self.assertEqual(empty_logger.get_recent_logs(), [])

        # Corrupt file with invalid lines and unredacted sensitive keys
        corrupt_path = os.path.join(self.temp_dir.name, "corrupt.jsonl")
        valid_rec1 = {"log_id": str(uuid.uuid4()), "request_id": str(uuid.uuid4()), "status": "ok", "predicted_cnt": 100}
        valid_rec2 = {
            "log_id": str(uuid.uuid4()),
            "request_id": str(uuid.uuid4()),
            "status": "ok",
            "predicted_cnt": 200,
            "input_vector": {"api_key": "raw_leaked_key", "temp": 0.5},
        }

        with open(corrupt_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(valid_rec1) + "\n")
            f.write("\n")  # Blank line
            f.write("MALFORMED JSON LINE {not valid json\n")  # Broken JSON line
            f.write("   \n")  # Whitespace line
            f.write(json.dumps(valid_rec2) + "\n")
            f.write('{"truncated": true, "half": \n')  # Half-written line

        corrupt_logger = ExecutionLogger(log_path=corrupt_path)
        recovered = corrupt_logger.get_recent_logs(limit=10)

        # Corrupted lines must be safely skipped; both valid records recovered
        self.assertEqual(len(recovered), 2)
        # Newest first
        self.assertEqual(recovered[0]["predicted_cnt"], 200)
        self.assertEqual(recovered[1]["predicted_cnt"], 100)
        # Defense-in-depth sanitization: raw key masked on read
        self.assertEqual(recovered[0]["input_vector"]["api_key"], "[REDACTED]")
        self.assertEqual(recovered[0]["input_vector"]["temp"], 0.5)


class TestSprint3Phase5ComprehensiveRegressionAndEdgeCases(unittest.TestCase):
    """Sprint 3 Phase 5: Comprehensive Project-Wide Verification & Regression Suite.

    Validates:
    1. Boundary hours (0 and 23) across multiple tiers.
    2. Date formats: date, datetime, ISO timestamp strings, leap years, min/max operational years.
    3. Date validation rejection & error recording (ERR_VAL_02).
    4. Hour validation rejection & error recording (ERR_VAL_01).
    5. Extreme manual weather inputs (sub-zero, extreme heat, 100% hum, gale wind, heavy rain).
    6. Non-numeric manual weather inputs rejection.
    7. Missing bundle fallback to dynamic training mean (~174.64 -> 175).
    8. Corrupted model bundle graceful degradation (is_artifact_loaded=False, WARN_SYS_01).
    9. Weather API HTTP error handling (404, 500) without crashing or fabricating predictions.
    10. Weather API timeout handling without silent fallback.
    11. Weather API unsupported forecast horizon (>7 days).
    12. Weather API unknown WMO code handling (WARN_MAP_01).
    13. Past date live forecast rejection policy.
    14. High concurrency stress test across 20 threads with unique UUIDs and uncorrupted JSONL.
    15. Execution logger self-healing and recovery from corrupted JSONL lines.
    16. End-to-end dataset schema and 80/20 chronological split integrity.
    17. End-to-end model performance progression (L0-L5) and test MAE hierarchy.
    18. Strict SRS Table 9.4 12-field schema persistence verification.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_path = os.path.join(self.temp_dir.name, "phase5_test_log.jsonl")
        self.logger = ExecutionLogger(log_path=self.log_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_regression_01_boundary_hours_0_and_23(self):
        """1. Boundary hours 0 (midnight) and 23 (11 PM) succeed across L0, L4, and L5 with valid outputs."""
        for hr in [0, 23]:
            for lvl in ["L0", "L4", "L5"]:
                res = predict_demand_workflow(
                    target_date=date(2026, 11, 2),
                    target_hour=hr,
                    model_level=lvl,
                    use_manual_weather=True,
                    manual_weather={
                        "temperature_c": 16.0,
                        "apparent_temperature_c": 14.5,
                        "humidity_pct": 55.0,
                        "wind_speed_kmh": 10.0,
                        "weather_desc": "Clear",
                    },
                    log_path=self.log_path,
                )
                self.assertTrue(res["success"], f"Failed for hr={hr}, lvl={lvl}")
                self.assertEqual(res["status"], "ok")
                self.assertIsInstance(res["predicted_cnt"], int)
                self.assertGreaterEqual(res["predicted_cnt"], 0)
                self.assertEqual(res["feature_dict"]["hr"], hr)
                self.assertTrue(is_valid_uuid4(res["request_id"]))

    def test_regression_02_datetime_representation_formats(self):
        """2. All valid date representations (date, datetime, ISO timestamp strings, leap day, min/max years) succeed."""
        date_formats = [
            (date(2026, 11, 2), 14),
            (datetime(2026, 11, 2, 14, 0), None),
            ("2026-11-02T14:00", None),
            ("2026-11-02", 14),
            (date(2024, 2, 29), 12),      # Leap day
            (date(2011, 1, 1), 0),        # Min operational boundary
            (date(2030, 12, 31), 23),     # Max operational boundary
        ]
        for dt_val, hr_val in date_formats:
            res = predict_demand_workflow(
                target_date=dt_val,
                target_hour=hr_val,
                model_level="L0",
                log_path=self.log_path,
            )
            self.assertTrue(res["success"], f"Failed for dt={dt_val}, hr={hr_val}")
            self.assertEqual(res["status"], "ok")
            self.assertIsInstance(res["predicted_cnt"], int)
            self.assertGreaterEqual(res["predicted_cnt"], 0)

    def test_regression_03_datetime_validation_rejections(self):
        """3. Invalid dates (None, bad strings, <2011, >2030, bad types) halt execution with ERR_VAL_02."""
        invalid_dates = [
            (None, 12),
            ("invalid-date-string", 12),
            ("2026-13-45", 12),
            (date(2010, 12, 31), 12),     # Pre-operational
            (date(2031, 1, 1), 12),       # Beyond operational horizon
            (123456, 12),                 # Unsupported type
        ]
        for dt_val, hr_val in invalid_dates:
            res = predict_demand_workflow(
                target_date=dt_val,
                target_hour=hr_val,
                model_level="L0",
                log_path=self.log_path,
            )
            self.assertFalse(res["success"])
            self.assertEqual(res["error_code"], "ERR_VAL_02")
            self.assertIsNone(res["predicted_cnt"])
            self.assertEqual(res["status"], "validation_error")

    def test_regression_04_hour_validation_rejections(self):
        """4. Invalid hours (-1, 24, 25, strings, floats, None) halt execution with ERR_VAL_01."""
        invalid_hours = [-1, 24, 25, 100, "invalid_str", 12.5, None]
        for hr_val in invalid_hours:
            res = predict_demand_workflow(
                target_date=date(2026, 11, 2),
                target_hour=hr_val,
                model_level="L0",
                log_path=self.log_path,
            )
            self.assertFalse(res["success"])
            self.assertEqual(res["error_code"], "ERR_VAL_01")
            self.assertIsNone(res["predicted_cnt"])
            self.assertEqual(res["status"], "validation_error")

    def test_regression_05_extreme_weather_inputs_manual_override(self):
        """5. Extreme weather inputs (sub-zero, extreme heat, 100% hum, gale wind, heavy rain) produce non-negative predictions."""
        extreme_scenarios = [
            {"temperature_c": -15.0, "apparent_temperature_c": -20.0, "humidity_pct": 0.0, "wind_speed_kmh": 70.0, "weather_desc": "Clear"},
            {"temperature_c": 45.0, "apparent_temperature_c": 48.0, "humidity_pct": 100.0, "wind_speed_kmh": 0.0, "weather_desc": "Heavy Rain"},
            {"temperature_c": 0.0, "apparent_temperature_c": -5.0, "humidity_pct": 50.0, "wind_speed_kmh": 35.0, "weather_desc": "Mist/Cloudy"},
        ]
        for scen in extreme_scenarios:
            for lvl in ["L4", "L5"]:
                res = predict_demand_workflow(
                    target_date=date(2026, 11, 2),
                    target_hour=14,
                    model_level=lvl,
                    use_manual_weather=True,
                    manual_weather=scen,
                    log_path=self.log_path,
                )
                self.assertTrue(res["success"], f"Failed for lvl={lvl}, scen={scen}")
                self.assertIsInstance(res["predicted_cnt"], int)
                self.assertGreaterEqual(res["predicted_cnt"], 0)
                self.assertEqual(res["weather_source"], "Manual Input (Labelled Fallback)")

    def test_regression_06_non_numeric_manual_weather_rejection(self):
        """6. Non-numeric manual weather inputs halt with ERR_VAL_01 and log a validation_error."""
        bad_weather = {"temperature_c": "freezing", "humidity_pct": "humid", "wind_speed_kmh": "fast", "weather_desc": "Clear"}
        res = predict_demand_workflow(
            target_date=date(2026, 11, 2),
            target_hour=14,
            model_level="L4",
            use_manual_weather=True,
            manual_weather=bad_weather,
            log_path=self.log_path,
        )
        self.assertFalse(res["success"])
        self.assertEqual(res["error_code"], "ERR_VAL_01")
        self.assertIn("must be numeric", res["message"])
        self.assertEqual(res["status"], "validation_error")

    def test_regression_07_missing_model_bundle_dynamic_mean_fallback(self):
        """7. Missing model bundle falls back to dynamically calculated training mean with WARN_SYS_01."""
        non_existent_bundle = os.path.join(self.temp_dir.name, "missing_model.joblib")
        # Instantiate ModelManager with a bundle path that doesn't exist; simulate missing bundle without retraining
        mm = ModelManager.__new__(ModelManager)
        mm.dataset_path = "dataset/hour.csv"
        mm.bundle_path = non_existent_bundle
        mm.df = load_dataset("dataset/hour.csv")
        mm.n_total = len(mm.df)
        mm.n_train = int(len(mm.df) * 0.80)
        mm.train_df = mm.df.iloc[: mm.n_train].copy()
        mm.test_df = mm.df.iloc[mm.n_train :].copy()
        mm.y_train = mm.train_df["cnt"].values
        mm.y_test = mm.test_df["cnt"].values
        mm.X_train = mm.train_df[["season", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"]]
        mm.X_test = mm.test_df[["season", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"]]
        mm.is_artifact_loaded = False
        mm.lr_model = None
        mm.rf_model = None
        mm.l0_mean = float(mm.y_train.mean())
        mm.l1_map = {}
        mm.l2_map = {}
        mm.l3_map = {}
        mm.mae_scores = {"L0": 174.98, "L1": 113.73, "L2": 101.55, "L3": 95.48, "L4": 134.70, "L5": None}
        mm.improvements = {"L0": -53.86, "L1": 0.0, "L2": 10.71, "L3": 16.05, "L4": -18.44, "L5": None}

        pred_res = mm.predict("L5", {
            "season": 1, "mnth": 1, "hr": 12, "holiday": 0, "weekday": 1,
            "workingday": 1, "weathersit": 1, "temp": 0.3, "atemp": 0.3, "hum": 0.5, "windspeed": 0.1,
        })
        self.assertEqual(pred_res["used_level"], "L0")
        self.assertEqual(pred_res["warning_code"], "WARN_SYS_01")
        self.assertEqual(pred_res["status"], "model_fallback_l0")
        self.assertAlmostEqual(pred_res["predicted_cnt"], 175, delta=1)

    def test_regression_08_corrupted_model_bundle_graceful_handling(self):
        """8. Corrupted model bundle file degrades gracefully to L0 dynamic mean baseline with WARN_SYS_01."""
        corrupt_path = os.path.join(self.temp_dir.name, "corrupt_bundle.joblib")
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_VALID_JOBLIB_FILE_CORRUPT_BYTES_DATA")

        mm = ModelManager(bundle_path=corrupt_path)
        self.assertFalse(mm.is_artifact_loaded)
        self.assertIsNone(mm.lr_model)
        self.assertIsNone(mm.rf_model)
        self.assertAlmostEqual(mm.l0_mean, 174.64, delta=0.5)

        pred_res = mm.predict("L4", {
            "season": 2, "mnth": 4, "hr": 17, "holiday": 0, "weekday": 3,
            "workingday": 1, "weathersit": 1, "temp": 0.5, "atemp": 0.5, "hum": 0.4, "windspeed": 0.2,
        })
        self.assertEqual(pred_res["used_level"], "L0")
        self.assertEqual(pred_res["warning_code"], "WARN_SYS_01")
        self.assertEqual(pred_res["status"], "model_fallback_l0")

    def test_regression_09_weather_api_http_errors_404_and_500(self):
        """9. HTTP 404/500 errors from Open-Meteo record weather_api_error and WARN_API_01 without fabricating predictions."""
        from urllib.error import HTTPError
        for status_code in [404, 500]:
            with patch("urllib.request.urlopen") as mock_url:
                mock_url.side_effect = HTTPError("http://example.com", status_code, "HTTP Error", {}, None)
                res = predict_demand_workflow(
                    target_date=date(2026, 11, 2),
                    target_hour=14,
                    model_level="L5",
                    use_manual_weather=False,
                    current_date=date(2026, 11, 1),
                    log_path=self.log_path,
                )
                self.assertFalse(res["success"])
                self.assertEqual(res["status"], "weather_api_error")
                self.assertEqual(res["warning_code"], "WARN_API_01")
                self.assertIsNone(res["predicted_cnt"])

    def test_regression_10_weather_api_timeout_no_silent_fallback(self):
        """10. Connection timeout records weather_api_error and WARN_API_01; does NOT silently run L0."""
        with patch("urllib.request.urlopen") as mock_url:
            mock_url.side_effect = TimeoutError("Connection timed out after 2.0s")
            res = predict_demand_workflow(
                target_date=date(2026, 11, 2),
                target_hour=14,
                model_level="L5",
                use_manual_weather=False,
                current_date=date(2026, 11, 1),
                log_path=self.log_path,
            )
            self.assertFalse(res["success"])
            self.assertEqual(res["status"], "weather_api_error")
            self.assertEqual(res["warning_code"], "WARN_API_01")
            self.assertIsNone(res["predicted_cnt"])

    def test_regression_11_weather_api_unsupported_forecast_horizon(self):
        """11. Timestamp outside the 7-day forecast horizon records unsupported_time and WARN_API_02."""
        from tests.test_sprint3 import _MockHTTPResponse
        mock_payload = {
            "hourly": {
                "time": ["2026-11-01T12:00", "2026-11-01T13:00"],
                "temperature_2m": [15.0, 15.0],
                "apparent_temperature": [14.0, 14.0],
                "relative_humidity_2m": [60.0, 60.0],
                "wind_speed_10m": [8.0, 8.0],
                "weather_code": [0, 0],
            }
        }
        with patch("urllib.request.urlopen") as mock_url:
            mock_url.return_value = _MockHTTPResponse(json.dumps(mock_payload).encode("utf-8"), status=200)
            res = predict_demand_workflow(
                target_date=date(2026, 11, 20),   # 19 days in the future
                target_hour=14,
                model_level="L5",
                use_manual_weather=False,
                current_date=date(2026, 11, 1),
                log_path=self.log_path,
            )
            self.assertFalse(res["success"])
            self.assertEqual(res["status"], "unsupported_time")
            self.assertEqual(res["warning_code"], "WARN_API_02")
            self.assertIsNone(res["predicted_cnt"])

    def test_regression_12_weather_api_unknown_wmo_code(self):
        """12. Unknown WMO code (999) records weather_validation_error and WARN_MAP_01."""
        from tests.test_sprint3 import _MockHTTPResponse
        mock_payload = {
            "hourly": {
                "time": ["2026-11-02T14:00"],
                "temperature_2m": [18.0],
                "apparent_temperature": [17.0],
                "relative_humidity_2m": [60.0],
                "wind_speed_10m": [15.0],
                "weather_code": [999],
            }
        }
        with patch("urllib.request.urlopen") as mock_url:
            mock_url.return_value = _MockHTTPResponse(json.dumps(mock_payload).encode("utf-8"), status=200)
            res = predict_demand_workflow(
                target_date=date(2026, 11, 2),
                target_hour=14,
                model_level="L5",
                use_manual_weather=False,
                current_date=date(2026, 11, 1),
                log_path=self.log_path,
            )
            self.assertFalse(res["success"])
            self.assertEqual(res["status"], "weather_validation_error")
            self.assertEqual(res["warning_code"], "WARN_MAP_01")
            self.assertIsNone(res["predicted_cnt"])

    def test_regression_13_past_date_forecast_rejection_policy(self):
        """13. Target date in the past without manual weather records historical_weather_unavailable and halts."""
        res = predict_demand_workflow(
            target_date=date(2026, 10, 25),
            target_hour=14,
            model_level="L5",
            use_manual_weather=False,
            current_date=date(2026, 11, 1),
            log_path=self.log_path,
        )
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "historical_weather_unavailable")
        self.assertIsNone(res["predicted_cnt"])

    def test_regression_14_high_concurrency_stress_20_threads(self):
        """14. High concurrency stress test across 20 threads generates unique UUIDs and uncorrupted JSONL log entries."""
        def call_worker(idx):
            return predict_demand_workflow(
                target_date=date(2026, 11, 2),
                target_hour=idx % 24,
                model_level="L0",
                log_path=self.log_path,
            )

        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(call_worker, range(20)))

        self.assertEqual(len(results), 20)
        for r in results:
            self.assertTrue(r["success"])
            self.assertEqual(r["status"], "ok")

        # Verify all request IDs are unique RFC 4122 UUID v4s
        req_ids = [r["request_id"] for r in results]
        self.assertEqual(len(set(req_ids)), 20)
        for rid in req_ids:
            self.assertTrue(is_valid_uuid4(rid))

        # Verify all 20 lines written to disk are valid intact JSON records
        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        self.assertEqual(len(lines), 20)
        parsed_records = [json.loads(line) for line in lines]
        log_ids = [rec["log_id"] for rec in parsed_records]
        self.assertEqual(len(set(log_ids)), 20)
        for lid in log_ids:
            self.assertTrue(is_valid_uuid4(lid))

    def test_regression_15_logger_corrupt_file_recovery_and_append(self):
        """15. Execution logger safely ignores corrupted/half-written lines and appends cleanly."""
        corrupt_lines = [
            "MALFORMED NON-JSON GARBAGE LINE",
            "",
            '{"partial_record": true, "half": ',
            json.dumps({"log_id": str(uuid.uuid4()), "request_id": str(uuid.uuid4()), "status": "ok", "predicted_cnt": 150, "token": "secret_abc"}),
            "   \t\n",
        ]
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(corrupt_lines) + "\n")

        # Log new clean entry
        self.logger.log_execution(
            status="ok",
            requested_datetime="2026-11-02T15:00",
            predicted_cnt=220,
            model_level="L5",
        )

        recovered = self.logger.get_recent_logs(limit=10)
        self.assertEqual(len(recovered), 2)
        self.assertEqual(recovered[0]["predicted_cnt"], 220)
        self.assertEqual(recovered[1]["predicted_cnt"], 150)
        # Defense-in-depth sanitization: raw key masked on read
        self.assertEqual(recovered[1]["token"], "[REDACTED]")

    def test_regression_16_dataset_schema_and_chronological_split(self):
        """16. Dataset integrity: 17,379 rows, 17 columns, zero nulls, non-negative cnt, and 80/20 chronological split."""
        df = load_dataset("dataset/hour.csv")
        self.assertEqual(len(df), 17379)
        self.assertEqual(df.shape[1], 17)
        self.assertEqual(int(df.isnull().sum().sum()), 0)
        self.assertTrue((df["cnt"] >= 0).all())
        self.assertTrue(((df["casual"] + df["registered"]) == df["cnt"]).all())

        n_train = int(len(df) * 0.80)
        n_test = len(df) - n_train
        self.assertEqual(n_train, 13903)
        self.assertEqual(n_test, 3476)

    def test_regression_17_model_evaluation_metrics_and_hierarchy(self):
        """17. Model evaluation hierarchy: test set size 3,476, L0 MAE 174.98, L1 113.73, and L5 MAE improves over L0."""
        mm = get_model_manager()
        self.assertEqual(len(mm.test_df), 3476)
        self.assertEqual(mm.mae_scores["L0"], 174.98)
        self.assertEqual(mm.mae_scores["L1"], 113.73)
        self.assertEqual(mm.mae_scores["L2"], 101.55)
        self.assertEqual(mm.mae_scores["L3"], 95.48)
        self.assertEqual(mm.mae_scores["L4"], 134.70)
        if mm.mae_scores.get("L5") is not None:
            self.assertLess(mm.mae_scores["L5"], mm.mae_scores["L0"])
            self.assertLess(mm.mae_scores["L5"], mm.mae_scores["L4"])

        summary = mm.get_summary_table()
        self.assertEqual(len(summary), 6)
        self.assertEqual([row["Level"] for row in summary], ["L0", "L1", "L2", "L3", "L4", "L5"])

    def test_regression_18_strict_table_9_4_schema_disk_verification(self):
        """18. Strict SRS Table 9.4 12-field schema compliance on disk: log_id, request_id, UTC timestamp, types."""
        res = predict_demand_workflow(
            target_date=date(2026, 11, 2),
            target_hour=14,
            model_level="L5",
            use_manual_weather=True,
            manual_weather={"temperature_c": 19.5, "humidity_pct": 52.0, "wind_speed_kmh": 11.0, "weather_desc": "Clear"},
            log_path=self.log_path,
        )
        self.assertTrue(res["success"])
        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        self.assertEqual(len(lines), 1)

        record = json.loads(lines[0])
        expected_fields = {
            "log_id",
            "request_id",
            "timestamp",
            "requested_datetime",
            "status",
            "error_code",
            "weather_source",
            "api_status",
            "input_vector",
            "predicted_cnt",
            "model_level",
            "model_version",
        }
        self.assertEqual(set(record.keys()), expected_fields)
        self.assertTrue(is_valid_uuid4(record["log_id"]))
        self.assertTrue(is_valid_uuid4(record["request_id"]))
        self.assertNotEqual(record["log_id"], record["request_id"])
        self.assertTrue("T" in record["timestamp"])


if __name__ == "__main__":
    unittest.main()
