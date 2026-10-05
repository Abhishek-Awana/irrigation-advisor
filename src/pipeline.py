"""The complete flow for one field: data in -> decision -> words and maps out.

  weather (Open-Meteo)  ┐
  soil (SoilGrids)      │
  satellite (Sentinel-2)├─> engine.decide() ─> explain.explain()  (Hindi / English)
  irrigation log        ┘                   └> maps.build_maps()  (field vs neighbours)

Every input is optional at run time: if one source is offline the advice is still given,
with lower confidence, and the reason is listed in `errors`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import engine, irrigation_log, soil, weather

CONFIG_PATH = Path("config/field.json")
WEATHER_PAST_DAYS = 90
WEATHER_FORECAST_DAYS = 6


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text())


def save_config(cfg: dict, path: Path = CONFIG_PATH) -> None:
    Path(path).write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")


def field_geometry_and_area(cfg: dict):
    """(field polygon, area in m2). Area falls back to config `field_area_m2` if the polygon fails."""
    from src.satellite import field_area_m2, load_field_geometry

    geom = load_field_geometry()
    return geom, cfg.get("field_area_m2") or round(field_area_m2(geom))


def today_et0_mm(wx: pd.DataFrame, today=None) -> float | None:
    today = pd.Timestamp(today if today is not None else pd.Timestamp.today()).normalize()
    row = wx[pd.to_datetime(wx["date"]) == today]
    return float(row["et0_mm"].iloc[0]) if len(row) and pd.notna(row["et0_mm"].iloc[0]) else None


def run_advice(cfg: dict, today=None, use_satellite: bool = True, log_path: Path = irrigation_log.LOG_PATH,
               fetch_weather=weather.fetch_weather, fetch_soil=soil.field_soil, satellite_series=None) -> dict:
    """Gather every input and ask the engine for today's decision.

    Returns {"decision", "weather", "series", "soil", "geom", "area_m2", "log", "errors"}.
    """
    errors: list[str] = []
    lat, lon = cfg["latitude"], cfg["longitude"]
    crop = engine.load_crop(cfg["crop"])

    wx = fetch_weather(lat, lon, past_days=WEATHER_PAST_DAYS, forecast_days=WEATHER_FORECAST_DAYS)

    soil_params = None
    try:
        soil_params = fetch_soil(lat, lon, bottom_cm=engine.soil_bottom_cm(crop))
    except Exception as e:  # noqa: BLE001
        errors.append(f"field soil unavailable, using the texture-class table: {e}")

    geom, area_m2 = None, cfg.get("field_area_m2")
    try:
        geom, area_m2 = field_geometry_and_area(cfg)
    except Exception as e:  # noqa: BLE001
        errors.append(f"field outline unavailable: {e}")

    series, sat = None, None
    if use_satellite:
        try:
            if satellite_series is None:
                from src.timeseries import build_series

                series = build_series(geom)
            else:
                series = satellite_series(geom)
            from src.timeseries import satellite_summary

            sat = satellite_summary(series, today)
        except Exception as e:  # noqa: BLE001
            errors.append(f"satellite unavailable: {e}")

    flow = cfg.get("pump_flow_lps")
    log = irrigation_log.load_log(log_path)
    last_irr, events = irrigation_log.engine_inputs(log, area_m2, flow)
    if last_irr is None:
        last_irr = cfg.get("last_irrigation_date")

    decision = engine.decide(wx, cfg["crop"], cfg["sowing_date"], last_irr, cfg.get("soil"), sat,
                             today=today, irrigations=events, soil_params=soil_params,
                             field_area_m2=area_m2, pump_flow_lps=flow)
    return {"decision": decision, "weather": wx, "series": series, "soil": soil_params, "geom": geom,
            "area_m2": area_m2, "log": log, "errors": errors}


if __name__ == "__main__":
    from src.explain import explain

    cfg = load_config()
    result = run_advice(cfg)
    for e in result["errors"]:
        print("!", e)
    print(json.dumps(result["decision"], indent=2, default=str))
    print()
    print(explain(result["decision"], "hi" if cfg.get("language", "").lower() == "hindi" else "en",
                  model=cfg.get("llm_model", "gemma3:4b"))["text"])
