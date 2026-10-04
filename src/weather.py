"""Weather data from Open-Meteo (free, no API key).

Returns one row per day: date, rain_mm, et0_mm, tmax_c, tmin_c, is_forecast.
et0_mm is reference evapotranspiration (how much water a reference crop uses).
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import pandas as pd
import requests

URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY = [
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "temperature_2m_max",
    "temperature_2m_min",
]
SOIL_MOISTURE_DAILY = ["soil_moisture_0_to_7cm_mean", "soil_moisture_7_to_28cm_mean"]
CACHE_DIR = Path("data")
CACHE_MAX_AGE_S = 3 * 3600  # reuse downloaded data for 3 hours



def _parse(payload: dict) -> pd.DataFrame:
    """Convert the Open-Meteo JSON into a clean daily table."""
    d = payload["daily"]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(d["time"]),
            "rain_mm": d["precipitation_sum"],
            "et0_mm": d["et0_fao_evapotranspiration"],
            "tmax_c": d["temperature_2m_max"],
            "tmin_c": d["temperature_2m_min"],
        }
    )
    df["rain_mm"] = df["rain_mm"].fillna(0.0)
    return df


def _add_flags(df: pd.DataFrame) -> pd.DataFrame:
    today = pd.Timestamp.today().normalize()
    df = df.copy()
    df["is_forecast"] = df["date"] > today
    return df


def fetch_weather(
    lat: float,
    lon: float,
    past_days: int = 30,
    forecast_days: int = 6,
    use_cache: bool = True,
    cache_dir: Path = CACHE_DIR,
    timeout_s: int = 20,
) -> pd.DataFrame:
    """Daily weather for a point. past_days max is 92 on this endpoint."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # window in the name: a 30-day download must never be served to a 90-day request
    cache = cache_dir / f"weather_{lat:.3f}_{lon:.3f}_p{past_days}_f{forecast_days}.csv"

    if use_cache and cache.exists() and time.time() - cache.stat().st_mtime < CACHE_MAX_AGE_S:
        return _add_flags(pd.read_csv(cache, parse_dates=["date"]))

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(DAILY),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "auto",
    }
    try:
        r = requests.get(URL, params=params, timeout=timeout_s)
        r.raise_for_status()
        df = _parse(r.json())
    except (requests.RequestException, KeyError, ValueError) as e:
        if cache.exists():  # offline fallback: use the last good download
            warnings.warn(f"Weather download failed ({e}); using stale cache.")
            return _add_flags(pd.read_csv(cache, parse_dates=["date"]))
        raise

    df.to_csv(cache, index=False)
    return _add_flags(df)


def _archive_cache(prefix: str, lat: float, lon: float, start, end, cache_dir: Path) -> Path:
    """One cache file per request window. The archive is history, so it never expires."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{prefix}_{lat:.3f}_{lon:.3f}_{start}_{end}.csv"


def fetch_weather_archive(lat: float, lon: float, start, end, use_cache: bool = True,
                          cache_dir: Path = CACHE_DIR, timeout_s: int = 20) -> pd.DataFrame:
    """Observed daily weather for a past window, from the ERA5 archive (same columns as fetch_weather).

    Used to replay the water balance over history for validation, so both the weather and
    the soil moisture compared against it come from the same reanalysis.
    """
    cache = _archive_cache("archive_weather", lat, lon, start, end, cache_dir)
    if use_cache and cache.exists():
        return _add_flags(pd.read_csv(cache, parse_dates=["date"]))

    params = {
        "latitude": lat, "longitude": lon, "start_date": str(start), "end_date": str(end),
        "daily": ",".join(DAILY), "timezone": "auto",
    }
    try:
        r = requests.get(ARCHIVE_URL, params=params, timeout=timeout_s)
        r.raise_for_status()
        df = _parse(r.json())
    except (requests.RequestException, KeyError, ValueError) as e:
        if cache.exists():
            warnings.warn(f"Archive weather download failed ({e}); using stale cache.")
            return _add_flags(pd.read_csv(cache, parse_dates=["date"]))
        raise

    df.to_csv(cache, index=False)
    return _add_flags(df)


def fetch_soil_moisture(lat: float, lon: float, start, end, use_cache: bool = True,
                        cache_dir: Path = CACHE_DIR, timeout_s: int = 30) -> pd.DataFrame:
    """Daily ERA5-Land soil water content in m3/m3 for 0-7 cm and 7-28 cm.

    This is the independent dataset the water balance is validated against: it is an
    ECMWF land-surface model, not our own simple bucket.
    """
    cache = _archive_cache("soil_moisture", lat, lon, start, end, cache_dir)
    if use_cache and cache.exists():
        return pd.read_csv(cache, parse_dates=["date"])

    params = {
        "latitude": lat, "longitude": lon, "start_date": str(start), "end_date": str(end),
        "daily": ",".join(SOIL_MOISTURE_DAILY), "timezone": "auto",
    }
    try:
        r = requests.get(ARCHIVE_URL, params=params, timeout=timeout_s)
        r.raise_for_status()
        d = r.json()["daily"]
        df = pd.DataFrame({
            "date": pd.to_datetime(d["time"]),
            "sm_0_7_m3m3": d["soil_moisture_0_to_7cm_mean"],
            "sm_7_28_m3m3": d["soil_moisture_7_to_28cm_mean"],
        })
    except (requests.RequestException, KeyError, ValueError) as e:
        if cache.exists():
            warnings.warn(f"Soil moisture download failed ({e}); using stale cache.")
            return pd.read_csv(cache, parse_dates=["date"])
        raise

    df.to_csv(cache, index=False)
    return df


def summarize(df: pd.DataFrame) -> dict:
    """Headline numbers the decision engine will use later."""
    past = df[~df["is_forecast"]].tail(7)
    future = df[df["is_forecast"]].head(5)
    return {
        "rain_last_7d_mm": round(float(past["rain_mm"].sum()), 1),
        "et0_last_7d_mm": round(float(past["et0_mm"].sum()), 1),
        "rain_next_5d_mm": round(float(future["rain_mm"].sum()), 1),
        "days_in_table": int(len(df)),
        "last_observed_date": str(past["date"].max().date()) if len(past) else None,
    }


if __name__ == "__main__":
    cfg = json.loads(Path("config/field.json").read_text())
    table = fetch_weather(cfg["latitude"], cfg["longitude"])
    print(table.tail(12).to_string(index=False))
    print(summarize(table))