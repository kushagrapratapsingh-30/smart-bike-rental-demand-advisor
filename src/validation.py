"""
Input Validation and Sanitization Module
Sprint I Deliverable - Smart Bike Rental Demand Advisor

Enforces server-side boundary checks and input validation rules (FR-1.2, NFR-2, Appendix B).
Rejects invalid or out-of-bounds parameters before pipeline/baseline invocation.
"""

from typing import Tuple, Dict, Any, Union

# Allowed categorical sets
VALID_SEASONS = {
    "Spring": 1,
    "Summer": 2,
    "Fall": 3,
    "Winter": 4,
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


def validate_hour(hour: Any) -> Tuple[bool, str]:
    """
    Validate that the hour is an integer bounded strictly between 0 and 23 inclusive.
    Returns (is_valid, message).
    """
    try:
        # Check if numerical
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
    """Validate that the season belongs to the allowed category domain."""
    if season not in VALID_SEASONS:
        return False, f"ERR_VAL_02: Invalid season '{season}'. Must be one of: Spring, Summer, Fall, Winter."
    return True, "Valid season."


def validate_weather(weather: Any) -> Tuple[bool, str]:
    """Validate that weather severity belongs to the allowed domain."""
    if weather not in VALID_WEATHER:
        return False, f"ERR_VAL_03: Invalid weather '{weather}'. Must be one of: Clear, Mist/Cloudy, Light Rain/Snow, Heavy Rain."
    return True, "Valid weather."


def validate_working_day(working_day: Any) -> Tuple[bool, str]:
    """Validate working day flag (Yes/No, 1/0, True/False)."""
    if working_day not in VALID_WORKING_DAY:
        return False, f"ERR_VAL_04: Invalid working day value '{working_day}'. Must be 'Yes' or 'No'."
    return True, "Valid working day."


def validate_inputs(
    hour: Any,
    season: Any,
    weather: Any,
    working_day: Any
) -> Tuple[bool, str]:
    """
    Aggregate validation function running all boundary checks.
    Halts on first validation failure and returns error code and message.
    """
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
    """
    Sanitize validated inputs into a standard dictionary representation.
    Raises ValueError if validation fails.
    """
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


if __name__ == "__main__":
    print("=" * 60)
    print("INPUT VALIDATION CHECKS (TESTING FR-1.2, NFR-2)")
    print("=" * 60)

    # Test Case 1: Valid inputs
    res1, msg1 = validate_inputs(17, "Summer", "Clear", "Yes")
    print(f"Test 1 (Valid - Hour 17, Summer): valid={res1}, msg='{msg1}'")

    # Test Case 2: Out of bound hour (> 23)
    res2, msg2 = validate_inputs(25, "Summer", "Clear", "Yes")
    print(f"Test 2 (Invalid - Hour 25): valid={res2}, msg='{msg2}'")

    # Test Case 3: Negative hour (< 0)
    res3, msg3 = validate_inputs(-1, "Winter", "Mist/Cloudy", "No")
    print(f"Test 3 (Invalid - Hour -1): valid={res3}, msg='{msg3}'")

    # Test Case 4: Invalid season
    res4, msg4 = validate_inputs(12, "Monsoon", "Clear", "Yes")
    print(f"Test 4 (Invalid - Season Monsoon): valid={res4}, msg='{msg4}'")

    # Test Case 5: Sanitization demo
    clean_dict = sanitize_inputs(8, "Winter", "Mist/Cloudy", "Yes")
    print(f"Test 5 (Sanitized Output): {clean_dict}")
    print("=" * 60)
