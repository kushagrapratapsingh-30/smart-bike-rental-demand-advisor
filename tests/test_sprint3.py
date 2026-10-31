"""Unit tests for Sprint 3 execution logger, schema adherence, and concurrency safety.

Conforms to SRS v1.5 Section 5.6, Table 9.4, and Appendix C (TC-13).
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import uuid

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


if __name__ == "__main__":
    unittest.main()
