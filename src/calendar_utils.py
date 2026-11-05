"""Calendar utilities for deriving hour, weekday, month, season, holiday, and workingday."""

from datetime import date, datetime
from typing import Any, Dict


def is_us_dc_holiday(d: date) -> bool:
    """Check if date is an observed Washington, D.C. government / federal holiday.

    Adheres strictly to SRS v1.5 Section 4.4, Section 13 Release Gate 6,
    and District of Columbia Official Code § 1-612.02.

    Observation Rules (5 U.S.C. § 6103 & D.C. Code):
    - When a holiday falls on a Sunday, the next day (Monday) is observed.
    - When a holiday falls on a Saturday, the preceding day (Friday) is observed.
    - D.C. Emancipation Day (April 16) is an official D.C. public holiday.
    - Juneteenth (June 19) is recognized for modern dates (2021+) per Public Law 117-17;
      it was not a public holiday during the 2011–2012 historical dataset period.
    """
    m, day, y = d.month, d.day, d.year
    weekday = d.weekday()  # Monday=0, Sunday=6

    # 1. New Year's Day (January 1)
    if m == 1 and day == 1 and weekday < 5:
        return True
    if m == 1 and day == 2 and weekday == 0:  # Jan 1 fell on Sunday -> observed Monday Jan 2
        return True
    if m == 12 and day == 31 and weekday == 4:  # Jan 1 falls on Saturday -> observed Friday Dec 31
        return True

    # 2. Martin Luther King Jr. Day (3rd Monday in January)
    if m == 1 and weekday == 0 and 15 <= day <= 21:
        return True

    # 3. Washington's Birthday / Presidents Day (3rd Monday in February)
    if m == 2 and weekday == 0 and 15 <= day <= 21:
        return True

    # 4. D.C. Emancipation Day (April 16, D.C. Code § 1-612.02)
    if m == 4 and day == 16 and weekday < 5:
        return True
    if m == 4 and day == 15 and weekday == 4:  # Apr 16 is Saturday -> observed Friday Apr 15
        return True
    if m == 4 and day == 17 and weekday == 0:  # Apr 16 is Sunday -> observed Monday Apr 17
        return True

    # 5. Memorial Day (Last Monday in May)
    if m == 5 and weekday == 0 and day >= 25:
        return True

    # 6. Juneteenth National Independence Day (June 19) - recognized 2021+
    if y >= 2021:
        if m == 6 and day == 19 and weekday < 5:
            return True
        if m == 6 and day == 18 and weekday == 4:  # June 19 is Saturday -> observed Friday June 18
            return True
        if m == 6 and day == 20 and weekday == 0:  # June 19 is Sunday -> observed Monday June 20
            return True

    # 7. Independence Day (July 4)
    if m == 7 and day == 4 and weekday < 5:
        return True
    if m == 7 and day == 3 and weekday == 4:  # July 4 is Saturday -> observed Friday July 3
        return True
    if m == 7 and day == 5 and weekday == 0:  # July 4 is Sunday -> observed Monday July 5
        return True

    # 8. Labor Day (1st Monday in September)
    if m == 9 and weekday == 0 and day <= 7:
        return True

    # 9. Columbus Day (2nd Monday in October)
    if m == 10 and weekday == 0 and 8 <= day <= 14:
        return True

    # 10. Veterans Day (November 11)
    if m == 11 and day == 11 and weekday < 5:
        return True
    if m == 11 and day == 10 and weekday == 4:  # Nov 11 is Saturday -> observed Friday Nov 10
        return True
    if m == 11 and day == 12 and weekday == 0:  # Nov 11 is Sunday -> observed Monday Nov 12
        return True

    # 11. Thanksgiving Day (4th Thursday in November)
    if m == 11 and weekday == 3 and 22 <= day <= 28:
        return True

    # 12. Christmas Day (December 25)
    if m == 12 and day == 25 and weekday < 5:
        return True
    if m == 12 and day == 24 and weekday == 4:  # Dec 25 is Saturday -> observed Friday Dec 24
        return True
    if m == 12 and day == 26 and weekday == 0:  # Dec 25 is Sunday -> observed Monday Dec 26
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
