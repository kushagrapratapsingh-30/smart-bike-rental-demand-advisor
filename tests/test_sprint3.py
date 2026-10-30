"""Unit tests for Sprint 3 execution logger, schema adherence, and concurrency safety.

Conforms to SRS v1.5 Section 5.6, Table 9.4, and Appendix C (TC-13).
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
import tempfile
import unittest
import uuid

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


if __name__ == "__main__":
    unittest.main()
