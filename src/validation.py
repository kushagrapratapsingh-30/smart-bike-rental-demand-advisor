"""Input and dataset validation rules for bike demand estimation."""

from typing import Any, Dict, List, Tuple, Union

import pandas as pd

# Season coding per SRS v1.5: 1 = Winter, 2 = Spring, 3 = Summer, 4 = Fall
VALID_SEASONS = {
    "Winter": 1,
    "Spring": 2,
    "Summer": 3,
    "Fall": 4,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
}

VALID_WEATHER = {
    "Clear": 1,
    "Mist/Cloudy": 2,
    "Light Rain/Snow": 3,
    "Heavy Rain": 4,
    "Heavy Rain/Ice": 4,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
}

VALID_WORKING_DAY = {
    "Yes": 1,
    "No": 0,
    True: 1,
    False: 0,
    1: 1,
    0: 0,
}

REQUIRED_DATASET_COLUMNS = [
    "instant", "dteday", "season", "yr", "mnth", "hr",
    "holiday", "weekday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed",
    "casual", "registered", "cnt"
]


def validate_hour(hour: Any) -> Tuple[bool, str]:
    """Validate that hour is an integer between 0 and 23 (ERR_VAL_01)."""
    try:
        if isinstance(hour, bool) or not isinstance(hour, (int, float)):
            return False, "ERR_VAL_01: Hour must be an integer between 0 and 23."

        hour_val = int(hour)
        if hour != hour_val:
            return False, "ERR_VAL_01: Hour must be an integer between 0 and 23."

        if hour_val < 0 or hour_val > 23:
            return False, "ERR_VAL_01: Hour must be an integer between 0 and 23."

        return True, "Valid hour."
    except (ValueError, TypeError):
        return False, "ERR_VAL_01: Hour must be an integer between 0 and 23."


def validate_season(season: Any) -> Tuple[bool, str]:
    """Validate season name or numeric code (1=Winter, 2=Spring, 3=Summer, 4=Fall)."""
    if season not in VALID_SEASONS:
        return False, f"ERR_VAL_SEASON: Invalid season '{season}'. Must be one of: Winter, Spring, Summer, Fall (or 1, 2, 3, 4)."
    return True, "Valid season."


def validate_weather(weather: Any) -> Tuple[bool, str]:
    """Validate weather category or numeric code."""
    if weather not in VALID_WEATHER:
        return False, f"ERR_VAL_WEATHER: Invalid weather '{weather}'. Must be one of: Clear, Mist/Cloudy, Light Rain/Snow, Heavy Rain (or 1, 2, 3, 4)."
    return True, "Valid weather."


def validate_working_day(working_day: Any) -> Tuple[bool, str]:
    """Validate working day flag."""
    if working_day not in VALID_WORKING_DAY:
        return False, f"ERR_VAL_WORKINGDAY: Invalid working day value '{working_day}'. Must be 'Yes' or 'No'."
    return True, "Valid working day."


def validate_inputs(
    hour: Any,
    season: Any,
    weather: Any,
    working_day: Any
) -> Tuple[bool, str]:
    """Validate all input fields, returning the first error encountered."""
    ok, msg = validate_hour(hour)
    if not ok:
        return False, msg

    ok, msg = validate_season(season)
    if not ok:
        return False, msg

    ok, msg = validate_weather(weather)
    if not ok:
        return False, msg

    ok, msg = validate_working_day(working_day)
    if not ok:
        return False, msg

    return True, "All inputs validated successfully."


def sanitize_inputs(
    hour: Any,
    season: Any,
    weather: Any,
    working_day: Any
) -> Dict[str, Union[int, str]]:
    """Convert valid inputs into mapped codes for model inference."""
    is_valid, error_msg = validate_inputs(hour, season, weather, working_day)
    if not is_valid:
        raise ValueError(error_msg)

    return {
        "hour": int(hour),
        "season": str(season),
        "season_code": VALID_SEASONS[season],
        "weather": str(weather),
        "weather_code": VALID_WEATHER[weather],
        "working_day": str(working_day),
        "working_day_flag": VALID_WORKING_DAY[working_day],
    }


def validate_dataset_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate DataFrame columns, ranges, and target consistency.
    Enforces C-2 constraint: casual + registered == cnt.
    """
    errors: List[str] = []

    if not isinstance(df, pd.DataFrame):
        return False, ["Input is not a valid pandas DataFrame."]

    missing_cols = [col for col in REQUIRED_DATASET_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0].to_dict()
    if cols_with_nulls:
        errors.append(f"Missing/null values detected in columns: {cols_with_nulls}")

    if "hr" in df.columns:
        invalid_hr = df[~df["hr"].isin(range(0, 24))]
        if len(invalid_hr) > 0:
            errors.append(f"Hour (hr) contains {len(invalid_hr)} records outside valid range 0–23.")

    if "season" in df.columns:
        invalid_season = df[~df["season"].isin([1, 2, 3, 4])]
        if len(invalid_season) > 0:
            errors.append(f"Season contains {len(invalid_season)} records outside valid domain 1–4.")

    if "weathersit" in df.columns:
        invalid_weather = df[~df["weathersit"].isin([1, 2, 3, 4])]
        if len(invalid_weather) > 0:
            errors.append(f"Weather (weathersit) contains {len(invalid_weather)} records outside valid domain 1–4.")

    for col in ["cnt", "casual", "registered"]:
        if col in df.columns:
            non_numeric = pd.to_numeric(df[col], errors="coerce").isnull()
            if non_numeric.any():
                errors.append(f"Column '{col}' contains non-numeric values.")
            else:
                negatives = (df[col] < 0).sum()
                if negatives > 0:
                    errors.append(f"Column '{col}' contains {negatives} negative rental counts.")

    # Target consistency check (Constraint C-2)
    if {"casual", "registered", "cnt"}.issubset(df.columns):
        mismatches = ((df["casual"] + df["registered"]) != df["cnt"]).sum()
        if mismatches > 0:
            errors.append(f"Target consistency failure: {mismatches} records violate casual + registered == cnt.")

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_dataset_record(record: Union[Dict[str, Any], pd.Series]) -> Tuple[bool, List[str]]:
    """Validate a single row against schema and domain constraints."""
    errors: List[str] = []

    if "hr" in record:
        hr_val = record["hr"]
        if not isinstance(hr_val, (int, float)) or int(hr_val) != hr_val or hr_val < 0 or hr_val > 23:
            errors.append(f"Invalid hour '{hr_val}'. Must be an integer between 0 and 23.")

    if "season" in record:
        s_val = record["season"]
        if s_val not in VALID_SEASONS:
            errors.append(f"Invalid season '{s_val}'. Must be one of 1, 2, 3, 4 or Winter, Spring, Summer, Fall.")

    if "weathersit" in record:
        w_val = record["weathersit"]
        if w_val not in VALID_WEATHER:
            errors.append(f"Invalid weathersit '{w_val}'. Must be in 1, 2, 3, 4.")

    for col in ["cnt", "casual", "registered"]:
        if col in record:
            val = record[col]
            if not isinstance(val, (int, float)) or val < 0:
                errors.append(f"Column '{col}' value '{val}' must be a non-negative numeric value.")

    # Target consistency check (Constraint C-2)
    if {"casual", "registered", "cnt"}.issubset(record.keys() if isinstance(record, dict) else record.index):
        if record["casual"] + record["registered"] != record["cnt"]:
            errors.append(f"Consistency mismatch: casual ({record['casual']}) + registered ({record['registered']}) != cnt ({record['cnt']}).")

    return len(errors) == 0, errors


if __name__ == "__main__":
    print("=== Input Validation Self-Check ===")
    res1, msg1 = validate_inputs(17, "Summer", "Clear", "Yes")
    print(f"Valid inputs (17, Summer): {res1}")

    res2, msg2 = validate_inputs(25, "Summer", "Clear", "Yes")
    print(f"Invalid hour (25): {msg2}")

    res3, msg3 = validate_inputs(-1, "Winter", "Mist/Cloudy", "No")
    print(f"Negative hour (-1): {msg3}")

    res4, msg4 = validate_inputs(12, "Monsoon", "Clear", "Yes")
    print(f"Invalid season: {msg4}")

    print("Sanitized (Winter):", sanitize_inputs(8, "Winter", "Mist/Cloudy", "Yes"))
    print("Sanitized (Summer):", sanitize_inputs(8, "Summer", "Clear", "Yes"))
