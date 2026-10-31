"""Prediction workflow service coordinating validation, calendar features, weather, model inference, and logging.

Conforms to SRS v1.5 Section 5.1 (Request Lifecycle), Section 5.4 (Fallback Rules),
Section 5.5 (Failure Handling), Section 5.6 (Logging), Section 5.8 (ER Model),
and Table 9.4 (ExecutionLog).
"""

from datetime import date, datetime, timezone
import os
from typing import Any, Dict, Optional, Tuple, Union
import uuid

from src.calendar_utils import derive_calendar_features
from src.logger import DEFAULT_LOG_PATH, ExecutionLogger, get_execution_logger, is_valid_uuid4
from src.models import FEATURE_COLUMNS, ModelManager, get_model_manager
from src.validation import (
    VALID_WEATHER,
    validate_hour,
    validate_inputs,
    validate_weather,
)
from src.weather import (
    fetch_open_meteo_forecast,
    normalize_weather_features,
)


def validate_request_datetime(target_date: Any) -> Tuple[bool, Optional[str], Optional[str]]:
    """Validate requested date adhering to SRS Appendix B ERR_VAL_02.

    Returns:
        (is_valid: bool, error_code: Optional[str], error_message: Optional[str])
    """
    if target_date is None:
        return False, "ERR_VAL_02", "ERR_VAL_02: Requested date is required."

    if isinstance(target_date, datetime):
        d = target_date.date()
    elif isinstance(target_date, date):
        d = target_date
    elif isinstance(target_date, str):
        try:
            d = datetime.fromisoformat(target_date).date()
        except Exception:
            return False, "ERR_VAL_02", "ERR_VAL_02: Requested date/time is invalid or unparseable."
    else:
        return False, "ERR_VAL_02", "ERR_VAL_02: Requested date/time type is not supported."

    # Supported operational range: 2011 to 2030 (dataset inception to operational horizon)
    if d.year < 2011 or d.year > 2030:
        return False, "ERR_VAL_02", f"ERR_VAL_02: Requested date '{d.isoformat()}' is outside supported range (2011-2030)."

    return True, None, "Valid date."


def predict_demand_workflow(
    target_date: Union[date, datetime, str],
    target_hour: Optional[Any] = None,
    model_level: str = "L5",
    use_manual_weather: bool = False,
    manual_weather: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    log_path: str = DEFAULT_LOG_PATH,
    model_manager: Optional[ModelManager] = None,
    current_date: Optional[date] = None,
) -> Dict[str, Any]:
    """Execute the end-to-end prediction workflow conforming to SRS v1.5 Section 5.1.

    Orchestrates:
    1. Validation of date and hour (ERR_VAL_01, ERR_VAL_02).
    2. Calendar predictor derivation.
    3. Weather resolution (Open-Meteo live retrieval or labelled manual override).
    4. Model inference via ModelManager (L0-L5) with cascading fallback.
    5. Exactly one execution log entry written to Table 9.4 JSONL storage.
    6. Non-negative output clipping and error/warning taxonomy enforcement.
    """
    logger = get_execution_logger(log_path=log_path)

    # 1. Ensure valid RFC 4122 UUID v4 request_id
    if request_id is not None and is_valid_uuid4(request_id):
        final_request_id = request_id
    else:
        final_request_id = str(uuid.uuid4())

    requested_level = str(model_level).upper() if model_level else "L5"

    # 2. Date Validation (SRS Appendix B ERR_VAL_02)
    is_date_valid, date_err_code, date_err_msg = validate_request_datetime(target_date)
    if not is_date_valid:
        req_dt_repr = str(target_date)
        logger.log_execution(
            status="validation_error",
            requested_datetime=req_dt_repr,
            request_id=final_request_id,
            error_code=date_err_code,
            weather_source=None,
            api_status=None,
            input_vector=None,
            predicted_cnt=None,
            model_level=None,
        )
        return {
            "success": False,
            "status": "validation_error",
            "error_code": date_err_code,
            "warning_code": None,
            "message": date_err_msg,
            "predicted_cnt": None,
            "request_id": final_request_id,
            "requested_datetime": req_dt_repr,
            "weather_source": None,
        }

    # Resolve date object
    if isinstance(target_date, datetime):
        d = target_date.date()
    elif isinstance(target_date, date):
        d = target_date
    else:
        d = datetime.fromisoformat(str(target_date)).date()

    # 3. Hour Validation (SRS Appendix B ERR_VAL_01)
    if target_hour is not None:
        hr_val = target_hour
    elif isinstance(target_date, datetime):
        hr_val = target_date.hour
    else:
        hr_val = None

    is_hr_valid, hr_err_msg = validate_hour(hr_val)
    if not is_hr_valid:
        req_dt_repr = f"{d.isoformat()} hr={hr_val}"
        logger.log_execution(
            status="validation_error",
            requested_datetime=req_dt_repr,
            request_id=final_request_id,
            error_code="ERR_VAL_01",
            weather_source=None,
            api_status=None,
            input_vector=None,
            predicted_cnt=None,
            model_level=None,
        )
        return {
            "success": False,
            "status": "validation_error",
            "error_code": "ERR_VAL_01",
            "warning_code": None,
            "message": hr_err_msg,
            "predicted_cnt": None,
            "request_id": final_request_id,
            "requested_datetime": req_dt_repr,
            "weather_source": None,
        }

    # Validated local target datetime
    target_dt = datetime(d.year, d.month, d.day, int(hr_val), 0)
    target_iso = target_dt.strftime("%Y-%m-%dT%H:00")

    # 4. Derive Calendar Features
    cal = derive_calendar_features(target_dt)

    # Validate derived season and workingday bounds
    is_inputs_valid, inputs_err_msg = validate_inputs(
        hour=cal["hr"],
        season=cal["season"],
        weather=1,  # Weather validated separately
        working_day=cal["workingday"],
    )
    if not is_inputs_valid:
        logger.log_execution(
            status="validation_error",
            requested_datetime=target_iso,
            request_id=final_request_id,
            error_code="ERR_VAL_01",
            weather_source=None,
            api_status=None,
            input_vector=None,
            predicted_cnt=None,
            model_level=None,
        )
        return {
            "success": False,
            "status": "validation_error",
            "error_code": "ERR_VAL_01",
            "warning_code": None,
            "message": inputs_err_msg,
            "predicted_cnt": None,
            "request_id": final_request_id,
            "requested_datetime": target_iso,
            "weather_source": None,
        }

    mm = model_manager or get_model_manager()

    # 5. Explicit L0 Baseline Selection (SRS Section 5.4)
    # L0 predicts historical training mean and does not depend on weather features.
    if requested_level == "L0":
        feature_dict = {
            "season": cal["season"],
            "mnth": cal["mnth"],
            "hr": cal["hr"],
            "holiday": cal["holiday"],
            "weekday": cal["weekday"],
            "workingday": cal["workingday"],
            "weathersit": 1,
            "temp": 0.0,
            "atemp": 0.0,
            "hum": 0.0,
            "windspeed": 0.0,
        }
        pred_res = mm.predict("L0", feature_dict)
        if pred_res.get("error_code") == "ERR_SYS_01" or pred_res["predicted_cnt"] is None:
            logger.log_execution(
                status="prediction_error",
                requested_datetime=target_iso,
                request_id=final_request_id,
                error_code="ERR_SYS_01",
                weather_source="Explicit L0 Baseline",
                api_status="not_required",
                input_vector=feature_dict,
                predicted_cnt=None,
                model_level=None,
            )
            return {
                "success": False,
                "status": "prediction_error",
                "error_code": "ERR_SYS_01",
                "warning_code": None,
                "message": pred_res.get("fallback_note", "Model and L0 configuration unavailable; cannot generate prediction."),
                "predicted_cnt": None,
                "request_id": final_request_id,
                "requested_datetime": target_iso,
                "weather_source": "Explicit L0 Baseline",
            }

        pred_cnt = pred_res["predicted_cnt"]
        logger.log_execution(
            status="ok",
            requested_datetime=target_iso,
            request_id=final_request_id,
            error_code=None,
            weather_source="Explicit L0 Baseline",
            api_status="not_required",
            input_vector=feature_dict,
            predicted_cnt=pred_cnt,
            model_level="L0",
        )
        return {
            "success": True,
            "status": "ok",
            "error_code": None,
            "warning_code": None,
            "predicted_cnt": pred_cnt,
            "predicted_cnt_raw": pred_res.get("predicted_cnt_raw"),
            "requested_level": "L0",
            "used_level": "L0",
            "model_level": "L0",
            "mae": pred_res.get("mae"),
            "fallback_note": pred_res.get("fallback_note", ""),
            "request_id": final_request_id,
            "requested_datetime": target_iso,
            "weather_source": "Explicit L0 Baseline",
            "api_status": "not_required",
            "raw_weather": None,
            "normalized_weather": None,
            "calendar_features": cal,
            "feature_dict": feature_dict,
        }

    # 6. Weather Resolution for Weather-Dependent Models (L1-L5)
    if use_manual_weather:
        # Manual Weather Override
        if not isinstance(manual_weather, dict):
            manual_weather = {}

        m_temp = manual_weather.get("temperature_c", 20.0)
        m_hum = manual_weather.get("humidity_pct", 55.0)
        m_wind = manual_weather.get("wind_speed_kmh", 12.0)
        m_desc = manual_weather.get("weather_desc", "Clear")

        is_w_valid, w_err_msg = validate_weather(m_desc)
        if not is_w_valid:
            logger.log_execution(
                status="validation_error",
                requested_datetime=target_iso,
                request_id=final_request_id,
                error_code="ERR_VAL_01",
                weather_source="Manual Input (Labelled Fallback)",
                api_status="manual_override",
                input_vector=None,
                predicted_cnt=None,
                model_level=requested_level,
            )
            return {
                "success": False,
                "status": "validation_error",
                "error_code": "ERR_VAL_01",
                "warning_code": None,
                "message": w_err_msg,
                "predicted_cnt": None,
                "request_id": final_request_id,
                "requested_datetime": target_iso,
                "weather_source": "Manual Input (Labelled Fallback)",
            }

        try:
            temp_c = float(m_temp)
            atemp_c = float(manual_weather.get("apparent_temperature_c", temp_c - 2.0))
            hum_pct = float(m_hum)
            wind_kmh = float(m_wind)
        except (ValueError, TypeError):
            logger.log_execution(
                status="validation_error",
                requested_datetime=target_iso,
                request_id=final_request_id,
                error_code="ERR_VAL_01",
                weather_source="Manual Input (Labelled Fallback)",
                api_status="manual_override",
                input_vector=None,
                predicted_cnt=None,
                model_level=requested_level,
            )
            return {
                "success": False,
                "status": "validation_error",
                "error_code": "ERR_VAL_01",
                "warning_code": None,
                "message": "Manual weather values must be numeric numbers.",
                "predicted_cnt": None,
                "request_id": final_request_id,
                "requested_datetime": target_iso,
                "weather_source": "Manual Input (Labelled Fallback)",
            }

        weathersit = VALID_WEATHER.get(m_desc, 1)
        norm_weather = normalize_weather_features(temp_c, atemp_c, hum_pct, wind_kmh)
        norm_weather["weathersit"] = weathersit

        raw_weather = {
            "temperature_c": temp_c,
            "apparent_temperature_c": atemp_c,
            "humidity_pct": hum_pct,
            "wind_speed_kmh": wind_kmh,
            "weather_desc": m_desc,
            "weathersit": weathersit,
        }
        weather_source = "Manual Input (Labelled Fallback)"
        api_status = "manual_override"

    else:
        # Live Weather Retrieval via Open-Meteo
        today = current_date if current_date is not None else date.today()

        # Check Past Date Policy (SRS Section 4.1 & 5.5)
        if target_dt.date() < today:
            logger.log_execution(
                status="historical_weather_unavailable",
                requested_datetime=target_iso,
                request_id=final_request_id,
                error_code=None,
                weather_source=None,
                api_status="past_date_unsupported",
                input_vector=None,
                predicted_cnt=None,
                model_level=requested_level,
            )
            return {
                "success": False,
                "status": "historical_weather_unavailable",
                "error_code": None,
                "warning_code": None,
                "message": "Past date requested. Live forecast is unavailable for past dates. Use Manual Weather Override or select Level 0 baseline.",
                "predicted_cnt": None,
                "request_id": final_request_id,
                "requested_datetime": target_iso,
                "weather_source": None,
            }

        weather_res = fetch_open_meteo_forecast(target_dt)

        if not weather_res["success"]:
            # Weather retrieval failed. Per SRS Section 5.4, do NOT silently run L0 and do NOT fabricate weather.
            fail_status = weather_res.get("status", "weather_api_error")
            warn_code = weather_res.get("warning_code", "WARN_API_01")
            api_stat = "timeout" if "timeout" in weather_res.get("message", "").lower() else "api_error"
            if fail_status == "unsupported_time":
                api_stat = "unsupported_time"
            elif fail_status == "weather_validation_error":
                api_stat = "validation_error"

            logger.log_execution(
                status=fail_status,
                requested_datetime=target_iso,
                request_id=final_request_id,
                error_code=warn_code,
                weather_source="Open-Meteo Forecast API",
                api_status=api_stat,
                input_vector=None,
                predicted_cnt=None,
                model_level=requested_level,
            )
            return {
                "success": False,
                "status": fail_status,
                "error_code": None,
                "warning_code": warn_code,
                "message": weather_res.get("message", "Live weather could not be retrieved. Enter weather manually or use Level 0 baseline."),
                "predicted_cnt": None,
                "request_id": final_request_id,
                "requested_datetime": target_iso,
                "weather_source": "Open-Meteo Forecast API",
            }

        raw_weather = weather_res["raw_weather"]
        norm_weather = weather_res["normalized_weather"]
        weather_source = "Open-Meteo Forecast API (Live)"
        api_status = "200"

    # 7. Model Inference
    feature_dict = {
        "season": cal["season"],
        "mnth": cal["mnth"],
        "hr": cal["hr"],
        "holiday": cal["holiday"],
        "weekday": cal["weekday"],
        "workingday": cal["workingday"],
        "weathersit": norm_weather["weathersit"],
        "temp": norm_weather["temp"],
        "atemp": norm_weather["atemp"],
        "hum": norm_weather["hum"],
        "windspeed": norm_weather["windspeed"],
    }

    pred_res = mm.predict(requested_level, feature_dict)
    pred_cnt = pred_res.get("predicted_cnt")
    used_level = pred_res.get("used_level")
    pred_status = pred_res.get("status", "ok")
    warn_code = pred_res.get("warning_code")
    err_code = pred_res.get("error_code")

    # Critical failure: Model and L0 both unavailable (TC-07)
    if pred_status == "prediction_error" or err_code == "ERR_SYS_01" or pred_cnt is None:
        logger.log_execution(
            status="prediction_error",
            requested_datetime=target_iso,
            request_id=final_request_id,
            error_code="ERR_SYS_01",
            weather_source=weather_source,
            api_status=api_status,
            input_vector=feature_dict,
            predicted_cnt=None,
            model_level=None,
        )
        return {
            "success": False,
            "status": "prediction_error",
            "error_code": "ERR_SYS_01",
            "warning_code": None,
            "message": pred_res.get("fallback_note", "Model and L0 configuration unavailable; cannot generate prediction."),
            "predicted_cnt": None,
            "request_id": final_request_id,
            "requested_datetime": target_iso,
            "weather_source": weather_source,
        }

    # Map status to authorized SRS Section 5.5 values
    if pred_status == "model_fallback_l0":
        final_status = "model_fallback_l0"
        logged_code = warn_code or "WARN_SYS_01"
    elif pred_status == "baseline_fallback":
        final_status = "baseline_fallback"
        logged_code = warn_code
    elif pred_status == "model_fallback_l4":
        # Per SRS Section 5.5, L4 ML prediction is valid ("ok") with model_level="L4"
        final_status = "ok"
        logged_code = None
    else:
        final_status = "ok"
        logged_code = None

    # Exactly one log entry written for completed prediction
    logger.log_execution(
        status=final_status,
        requested_datetime=target_iso,
        request_id=final_request_id,
        error_code=logged_code,
        weather_source=weather_source,
        api_status=api_status,
        input_vector=feature_dict,
        predicted_cnt=pred_cnt,
        model_level=used_level,
    )

    return {
        "success": True,
        "status": final_status,
        "error_code": err_code,
        "warning_code": warn_code,
        "predicted_cnt": pred_cnt,
        "predicted_cnt_raw": pred_res.get("predicted_cnt_raw"),
        "requested_level": requested_level,
        "used_level": used_level,
        "model_level": used_level,
        "mae": pred_res.get("mae"),
        "fallback_note": pred_res.get("fallback_note", ""),
        "request_id": final_request_id,
        "requested_datetime": target_iso,
        "weather_source": weather_source,
        "api_status": api_status,
        "raw_weather": raw_weather,
        "normalized_weather": norm_weather,
        "calendar_features": cal,
        "feature_dict": feature_dict,
    }
