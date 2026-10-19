"""Calendar utilities for deriving hour, weekday, month, season, holiday, and workingday."""

from datetime import date, datetime
from typing import Any, Dict


def is_us_dc_holiday(d: date) -> bool:
    """Check if date is an observed Washington, D.C. federal holiday."""
    m, day, y = d.month, d.day, d.year
    weekday = d.weekday()  # Monday=0, Sunday=6

    # Fixed date holidays (simple rule or observed)
    if m == 1 and day == 1:  # New Year's Day
        return True
    if m == 6 and day == 19:  # Juneteenth
        return True
    if m == 7 and day == 4:  # Independence Day
        return True
    if m == 11 and day == 11:  # Veterans Day
        return True
    if m == 12 and day == 25:  # Christmas Day
        return True

    # Floating Monday holidays
    # MLK Day: 3rd Monday in January
    if m == 1 and weekday == 0 and 15 <= day <= 21:
        return True
    # Washington's Birthday / Presidents Day: 3rd Monday in February
    if m == 2 and weekday == 0 and 15 <= day <= 21:
        return True
    # Memorial Day: Last Monday in May
    if m == 5 and weekday == 0 and day >= 25:
        return True
    # Labor Day: 1st Monday in September
    if m == 9 and weekday == 0 and day <= 7:
        return True
    # Columbus Day: 2nd Monday in October
    if m == 10 and weekday == 0 and 8 <= day <= 14:
        return True
    # Thanksgiving Day: 4th Thursday in November (Thursday is weekday 3)
    if m == 11 and weekday == 3 and 22 <= day <= 28:
        return True

    return False


def get_astronomical_season(m: int, day: int) -> int:
    """Map calendar date to season per SRS v1.5: 1=Winter, 2=Spring, 3=Summer, 4=Fall."""
    if (m == 12 and day >= 21) or (m in (1, 2)) or (m == 3 and day <= 20):
        return 1  # Winter
    elif (m == 3 and day >= 21) or (m in (4, 5)) or (m == 6 and day <= 20):
        return 2  # Spring
    elif (m == 6 and day >= 21) or (m in (7, 8)) or (m == 9 and day <= 22):
        return 3  # Summer
    else:
        return 4  # Fall


def derive_calendar_features(dt: datetime) -> Dict[str, Any]:
    """Derive all calendar predictors strictly adhering to UCI schema and SRS v1.5."""
    m = dt.month
    day = dt.day
    hr = dt.hour

    # UCI weekday convention: 0=Sunday, 1=Monday, ..., 6=Saturday
    # Python weekday: 0=Monday, ..., 6=Sunday
    uci_weekday = (dt.weekday() + 1) % 7

    season = get_astronomical_season(m, day)
    holiday = 1 if is_us_dc_holiday(dt.date()) else 0
    workingday = 1 if (uci_weekday not in (0, 6) and holiday == 0) else 0

    season_names = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}

    return {
        "hr": hr,
        "mnth": m,
        "day": day,
        "year": dt.year,
        "weekday": uci_weekday,
        "weekday_name": dt.strftime("%A"),
        "season": season,
        "season_name": season_names[season],
        "holiday": holiday,
        "workingday": workingday,
        "dteday": dt.strftime("%Y-%m-%d"),
    }
