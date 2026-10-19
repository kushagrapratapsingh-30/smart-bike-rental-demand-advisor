"""Open-Meteo weather client, WMO code mapping, and normalization per SRS v1.5."""

from datetime import datetime
import json
from typing import Any, Dict, Optional, Tuple
import urllib.error
import urllib.request

# Fixed coordinates for Washington, D.C. per SRS v1.5 Section 2.3
LATITUDE = 38.8951
LONGITUDE = -77.0364
TIMEZONE = "America/New_York"
TIMEOUT_SECONDS = 2.0

# WMO Weather Interpretation Codes (WW) mapping to UCI weathersit (1-4)
# per SRS v1.5 Section 4.3
WMO_CODE_MAP = {
    0: (1, "Clear sky"),
    1: (1, "Mainly clear"),
    2: (1, "Partly cloudy"),
    3: (2, "Overcast"),
    45: (2, "Fog"),
    48: (2, "Depositing rime fog"),
    51: (3, "Light drizzle"),
    53: (3, "Moderate drizzle"),
    55: (3, "Dense drizzle"),
    56: (3, "Light freezing drizzle"),
    57: (3, "Dense freezing drizzle"),
    61: (3, "Slight rain"),
    63: (3, "Moderate rain"),
    65: (4, "Heavy rain"),
    66: (3, "Light freezing rain"),
    67: (4, "Heavy freezing rain"),
    71: (3, "Slight snow fall"),
    73: (3, "Moderate snow fall"),
    75: (4, "Heavy snow fall"),
    77: (4, "Snow grains"),
    80: (3, "Slight rain showers"),
    81: (3, "Moderate rain showers"),
    82: (4, "Violent rain showers"),
    85: (3, "Slight snow showers"),
    86: (4, "Heavy snow showers"),
    95: (3, "Thunderstorm"),
    96: (4, "Thunderstorm with slight hail"),
    99: (4, "Thunderstorm with heavy hail"),
}


def map_wmo_code(wmo_code: int) -> Tuple[int, str]:
    """Map Open-Meteo WMO weather code to UCI weathersit class (1-4)."""
    if wmo_code in WMO_CODE_MAP:
        return WMO_CODE_MAP[wmo_code]
    # For unknown code, default to Class 2 with unverified mapping warning
    return 2, f"Unverified condition (WMO {wmo_code})"


def normalize_weather_features(
    temp_c: float,
    atemp_c: float,
    hum_pct: float,
    wind_kmh: float
) -> Dict[str, float]:
    """Normalize weather values to [0, 1] per SRS v1.5 Appendix B."""
    # temp_norm = (tempactual - tmin) / (tmax - tmin), tmin=-8°C, tmax=+39°C (diff=47)
    temp_norm = (temp_c - (-8.0)) / (39.0 - (-8.0))

    # atemp_norm = (atempactual - amin) / (amax - amin), amin=-16°C, amax=+50°C (diff=66)
    atemp_norm = (atemp_c - (-16.0)) / (50.0 - (-16.0))

    # hum_norm = hum% / 100
    hum_norm = hum_pct / 100.0

    # windspeed_norm = windspeedactual / 67 (after converting to km/h)
    windspeed_norm = wind_kmh / 67.0

    return {
        "temp": float(round(max(0.0, min(1.0, temp_norm)), 4)),
        "atemp": float(round(max(0.0, min(1.0, atemp_norm)), 4)),
        "hum": float(round(max(0.0, min(1.0, hum_norm)), 4)),
        "windspeed": float(round(max(0.0, min(1.0, windspeed_norm)), 4)),
    }


def fetch_open_meteo_forecast(target_dt: datetime) -> Dict[str, Any]:
    """Fetch hourly weather from Open-Meteo for Washington, D.C. with 2.0s timeout."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&hourly=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m"
        f"&timezone={TIMEZONE.replace('/', '%2F')}"
    )

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "SmartBikeRentalDemandAdvisor/1.5 (Student Project)"}
    )

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
            if response.status != 200:
                return {
                    "success": False,
                    "status": "weather_api_error",
                    "warning_code": "WARN_API_01",
                    "message": "Live weather could not be retrieved. Enter weather manually or use the Level 0 baseline.",
                }
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError:
        return {
            "success": False,
            "status": "weather_api_error",
            "warning_code": "WARN_API_01",
            "message": "Live weather could not be retrieved. Enter weather manually or use the Level 0 baseline.",
        }
    except Exception:
        return {
            "success": False,
            "status": "weather_api_error",
            "warning_code": "WARN_API_01",
            "message": "Live weather could not be retrieved. Enter weather manually or use the Level 0 baseline.",
        }

    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])

    # Target timestamp in ISO format: 'YYYY-MM-DDTHH:00'
    target_iso = target_dt.strftime("%Y-%m-%dT%H:00")

    if target_iso not in times:
        return {
            "success": False,
            "status": "unsupported_time",
            "warning_code": "WARN_API_02",
            "message": "Requested date/time is outside forecast coverage (7-day window). Enter weather manually or use Level 0 baseline.",
        }

    idx = times.index(target_iso)
    temp_c = float(hourly["temperature_2m"][idx])
    atemp_c = float(hourly["apparent_temperature"][idx])
    hum_pct = float(hourly["relative_humidity_2m"][idx])
    wind_kmh = float(hourly["wind_speed_10m"][idx])
    wmo_code = int(hourly["weather_code"][idx])

    weathersit, weather_desc = map_wmo_code(wmo_code)
    normalized = normalize_weather_features(temp_c, atemp_c, hum_pct, wind_kmh)

    return {
        "success": True,
        "status": "ok",
        "weather_source": "Open-Meteo Forecast API",
        "timestamp": target_iso,
        "raw_weather": {
            "temperature_c": temp_c,
            "apparent_temperature_c": atemp_c,
            "humidity_pct": hum_pct,
            "wind_speed_kmh": wind_kmh,
            "wmo_code": wmo_code,
            "weather_desc": weather_desc,
            "weathersit": weathersit,
        },
        "normalized_weather": {
            **normalized,
            "weathersit": weathersit,
        },
    }
