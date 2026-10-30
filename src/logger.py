"""Structured Execution Logger conforming to SRS v1.5 Section 5.6 and Table 9.4.

This module provides thread-safe, append-only JSONL execution logging for every
prediction request, validation failure, API timeout, and baseline fallback.
"""

from datetime import datetime, timezone
import json
import os
import threading
from typing import Any, Dict, List, Optional, Union
import uuid

DEFAULT_LOG_PATH = "logs/execution_log.jsonl"
SENSITIVE_KEY_SUBSTRINGS = ("key", "token", "secret", "password", "auth", "credential", "bearer")


def is_valid_uuid4(val: Any) -> bool:
    """Validate whether the given string is a standard RFC 4122 UUID v4."""
    if not isinstance(val, str):
        return False
    try:
        parsed = uuid.UUID(val, version=4)
        return str(parsed) == val and parsed.version == 4
    except (ValueError, AttributeError, TypeError):
        return False


def sanitize_payload(obj: Any) -> Any:
    """Recursively sanitize dictionary or collection to mask credentials and secrets."""
    if isinstance(obj, dict):
        sanitized = {}
        for k, v in obj.items():
            key_lower = str(k).lower()
            if any(sub in key_lower for sub in SENSITIVE_KEY_SUBSTRINGS):
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = sanitize_payload(v)
        return sanitized
    elif isinstance(obj, (list, tuple)):
        return [sanitize_payload(item) for item in obj]
    elif isinstance(obj, set):
        return [sanitize_payload(item) for item in sorted(list(obj), key=str)]
    return obj


class ExecutionLogger:
    """Thread-safe append-only execution logger conforming to SRS v1.5 Table 9.4."""

    def __init__(self, log_path: str = DEFAULT_LOG_PATH):
        self.log_path = log_path
        self._lock = threading.Lock()

    def log_execution(
        self,
        status: str,
        requested_datetime: Optional[Union[datetime, str]] = None,
        request_id: Optional[str] = None,
        error_code: Optional[str] = None,
        weather_source: Optional[str] = None,
        api_status: Optional[Union[str, int]] = None,
        input_vector: Optional[Dict[str, Any]] = None,
        predicted_cnt: Optional[Union[int, float]] = None,
        model_level: Optional[str] = None,
        model_version: Optional[str] = "1.5.0",
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Record a single execution outcome into append-only JSONL log.

        Guarantees:
        - log_id is a freshly generated RFC 4122 UUID v4.
        - request_id is either validated as UUID v4 (if provided) or newly generated as UUID v4.
        - All Table 9.4 fields are populated (None / null when unavailable).
        - Sensitive keys or secrets in input_vector are redacted.
        - Thread-safe write with fail-safe error handling.
        """
        # Ensure log_id is always a new RFC 4122 UUID v4
        log_id = str(uuid.uuid4())

        # Ensure request_id is a valid UUID v4
        if request_id is None or not is_valid_uuid4(request_id):
            final_request_id = str(uuid.uuid4())
        else:
            final_request_id = request_id

        # Format timestamps as standard ISO 8601 strings
        if timestamp is None:
            ts_str = datetime.now(timezone.utc).isoformat()
        elif isinstance(timestamp, datetime):
            ts_str = timestamp.isoformat()
        else:
            ts_str = str(timestamp)

        if requested_datetime is None:
            req_dt_str = None
        elif isinstance(requested_datetime, datetime):
            req_dt_str = requested_datetime.isoformat()
        else:
            req_dt_str = str(requested_datetime)

        # Sanitize input vector
        clean_input = sanitize_payload(input_vector) if input_vector is not None else None

        # Build record matching Table 9.4 schema exactly
        record = {
            "log_id": log_id,
            "request_id": final_request_id,
            "timestamp": ts_str,
            "requested_datetime": req_dt_str,
            "status": str(status) if status is not None else "unknown",
            "error_code": str(error_code) if error_code is not None else None,
            "weather_source": str(weather_source) if weather_source is not None else None,
            "api_status": str(api_status) if api_status is not None else None,
            "input_vector": clean_input,
            "predicted_cnt": int(round(predicted_cnt)) if isinstance(predicted_cnt, (int, float)) else None,
            "model_level": str(model_level) if model_level is not None else None,
            "model_version": str(model_version) if model_version is not None else "1.5.0",
        }

        # Safe append-only file persistence
        try:
            log_dir = os.path.dirname(self.log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            line = json.dumps(record, default=str, ensure_ascii=False) + "\n"
            with self._lock:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception:
            # Fallback guarantee: logging errors must never crash calling code
            pass

        return record

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent log entries ordered newest first."""
        if not os.path.exists(self.log_path):
            return []

        entries = []
        try:
            with self._lock:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entries.append(json.loads(line))
                        except Exception:
                            continue
        except Exception:
            return []

        # Return most recent entries (newest first)
        return list(reversed(entries))[:limit]


_global_logger_instance: Optional[ExecutionLogger] = None


def get_execution_logger(log_path: str = DEFAULT_LOG_PATH) -> ExecutionLogger:
    """Get or create singleton ExecutionLogger instance."""
    global _global_logger_instance
    if _global_logger_instance is None or _global_logger_instance.log_path != log_path:
        _global_logger_instance = ExecutionLogger(log_path=log_path)
    return _global_logger_instance


def log_prediction_request(
    status: str,
    requested_datetime: Optional[Union[datetime, str]] = None,
    request_id: Optional[str] = None,
    error_code: Optional[str] = None,
    weather_source: Optional[str] = None,
    api_status: Optional[Union[str, int]] = None,
    input_vector: Optional[Dict[str, Any]] = None,
    predicted_cnt: Optional[Union[int, float]] = None,
    model_level: Optional[str] = None,
    model_version: Optional[str] = "1.5.0",
    timestamp: Optional[datetime] = None,
    log_path: str = DEFAULT_LOG_PATH,
) -> Dict[str, Any]:
    """Convenience module function to log a request outcome using the default logger."""
    logger = get_execution_logger(log_path=log_path)
    return logger.log_execution(
        status=status,
        requested_datetime=requested_datetime,
        request_id=request_id,
        error_code=error_code,
        weather_source=weather_source,
        api_status=api_status,
        input_vector=input_vector,
        predicted_cnt=predicted_cnt,
        model_level=model_level,
        model_version=model_version,
        timestamp=timestamp,
    )


def get_recent_logs(limit: int = 50, log_path: str = DEFAULT_LOG_PATH) -> List[Dict[str, Any]]:
    """Convenience module function to retrieve recent logs using the default logger."""
    logger = get_execution_logger(log_path=log_path)
    return logger.get_recent_logs(limit=limit)
