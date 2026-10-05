import pandas as pd
import pytest

from src import irrigation_log, pipeline

TODAY = pd.Timestamp("2026-10-04")
CFG = {"crop": "sugarcane", "sowing_date": "2026-04-15", "latitude": 29.3733, "longitude": 77.7721,
       "last_irrigation_date": None, "pump_flow_lps": 10.0, "field_area_m2": 10_000}
SOIL = {"texture_class": "clay_loam", "theta_fc_m3m3": 0.3248, "theta_wp_m3m3": 0.1914,
        "taw_mm_per_m": 133.4, "source": "test"}


def fake_weather(lat, lon, past_days, forecast_days):
    dates = pd.date_range(TODAY - pd.Timedelta(days=past_days - 1), TODAY + pd.Timedelta(days=forecast_days))
    return pd.DataFrame({"date": dates, "rain_mm": 0.0, "et0_mm": 4.0, "is_forecast": dates > TODAY})


@pytest.fixture
def no_outline(monkeypatch):
    monkeypatch.setattr(pipeline, "field_geometry_and_area", lambda cfg: (None, cfg["field_area_m2"]))


def test_full_flow_with_irrigation_log(tmp_path, no_outline):
    log = tmp_path / "log.csv"
    irrigation_log.add_entry(TODAY - pd.Timedelta(days=4), path=log)
    r = pipeline.run_advice(CFG, today=TODAY, use_satellite=False, log_path=log,
                            fetch_weather=fake_weather, fetch_soil=lambda lat, lon, bottom_cm: SOIL)
    d = r["decision"]
    assert d["soil"] == "clay_loam" and d["action"] == "wait"
    assert d["depletion_mm"] == pytest.approx(4 * 4.0 * 1.25, abs=0.5)  # 4 days at Kc 1.25 x ET0 4
    assert not any("no irrigation date" in n for n in d["confidence_notes"])
    assert r["errors"] == []


def test_offline_soil_and_satellite_still_give_advice(tmp_path, no_outline):
    def broken_soil(*a, **k):
        raise ConnectionError("SoilGrids down")

    def broken_series(geom):
        raise ConnectionError("STAC down")

    r = pipeline.run_advice(CFG, today=TODAY, log_path=tmp_path / "none.csv", fetch_weather=fake_weather,
                            fetch_soil=broken_soil, satellite_series=broken_series)
    assert r["decision"]["action"] in ("irrigate_now", "irrigate_soon", "wait")
    assert len(r["errors"]) == 2
    assert r["decision"]["irrigation_depth_mm"]["water_m3"] is not None


def test_today_et0():
    wx = fake_weather(0, 0, 3, 1)
    assert pipeline.today_et0_mm(wx, TODAY) == 4.0
    assert pipeline.today_et0_mm(wx, TODAY + pd.Timedelta(days=30)) is None


def test_config_round_trip(tmp_path):
    p = tmp_path / "field.json"
    pipeline.save_config({"farmer_name": "रमेश", "crop": "wheat"}, p)
    assert pipeline.load_config(p)["farmer_name"] == "रमेश"
