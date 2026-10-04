import pandas as pd
import pytest

from src import validate

TODAY = pd.Timestamp("2026-10-04")
SOWING = TODAY - pd.Timedelta(days=170)  # sugarcane, comfortably mid-season


def weather(rain=None, past=25, et0=5.0):
    """Daily table ending TODAY. rain = {day_offset: mm}."""
    dates = pd.date_range(TODAY - pd.Timedelta(days=past - 1), TODAY)
    rain = rain or {}
    return pd.DataFrame({
        "date": dates,
        "rain_mm": [float(rain.get((d - TODAY).days, 0.0)) for d in dates],
        "et0_mm": et0,
        "is_forecast": False,
    })


def model_for(rain=None, **kw):
    wx = weather(rain, **kw)
    return validate.model_series(wx, "sugarcane", SOWING, soil_name="loam",
                                 start=wx["date"].min() - pd.Timedelta(days=1))


def obs_from(model, thickness_m=0.21, start_theta=0.45):
    """ERA5-Land stub: the same water loss in mm, expressed as theta for that layer.

    0.07 m and 0.21 m are the real ERA5-Land layer thicknesses, so the mm/day drying rate
    of the stub matches the model's exactly when the model is right.
    """
    return pd.DataFrame({
        "date": model["date"],
        "sm_0_7_m3m3": start_theta - model["dr_mm"] / (0.07 * 1000),
        "sm_7_28_m3m3": start_theta - model["dr_mm"] / (thickness_m * 1000),
    })


def test_model_series_reproduces_the_dry_down():
    m = model_for()
    assert m["theta_m3m3"].is_monotonic_decreasing  # no rain, only evaporation
    assert m["theta_m3m3"].iloc[0] == pytest.approx(0.29, abs=0.01)  # starts at loam field capacity
    assert m["theta_m3m3"].iloc[-1] > 0.14  # never drier than the wilting point


def test_model_series_wets_up_after_a_heavy_rain():
    m = model_for(rain={-10: 60}).set_index("date")["theta_m3m3"]
    rain_day = TODAY - pd.Timedelta(days=10)
    assert m[rain_day] > m[rain_day - pd.Timedelta(days=1)]  # refills more than a day of use
    assert m[rain_day] > m[rain_day + pd.Timedelta(days=1)]  # a spike, then it dries again


def test_compare_reports_near_perfect_when_signals_agree():
    m = model_for(rain={-12: 35, -5: 25})  # vary the signal, or a correlation means nothing
    metrics = validate.compare(m, obs_from(m), "sm_7_28_m3m3", "7-28 cm")
    assert metrics["enough_data"] is True
    assert metrics["anomaly_r"] == pytest.approx(1.0, abs=1e-6)  # identical day-to-day changes
    assert metrics["anomaly_rmse_mm_per_day"] == pytest.approx(0.0, abs=1e-6)  # same mm/day lost
    assert metrics["drying_ratio"] == pytest.approx(1.0, abs=0.01)  # same mm/day of water loss
    assert metrics["wet_event_hit_rate"] == 1.0
    assert metrics["level_bias_m3m3"] == metrics["level_bias_m3m3"]  # level offset still reported
    assert validate.verdict({"7-28 cm": metrics})["passed"] is True


def test_compare_is_undefined_when_the_soil_never_changes():
    flat = pd.DataFrame({"date": model_for()["date"], "theta_m3m3": 0.25, "dr_mm": 0.0,
                         "root_depth_m": 1.2, "rain_mm": 0.0, "et0_mm": 0.0})
    metrics = validate.compare(flat, obs_from(flat), "sm_7_28_m3m3", "7-28 cm")
    assert metrics["anomaly_r"] != metrics["anomaly_r"]  # NaN: no movement, no verdict
    assert metrics["drying_ratio"] is None
    assert validate.verdict({"7-28 cm": metrics})["passed"] is False


def test_compare_flags_a_model_that_moves_the_wrong_way():
    m = model_for(rain={-10: 40, -3: 30})
    inverted = obs_from(m)
    inverted["sm_7_28_m3m3"] = 0.45 + m["dr_mm"] / (0.21 * 1000)  # wets when we dry
    bad = validate.compare(m, inverted, "sm_7_28_m3m3", "7-28 cm")
    assert bad["anomaly_r"] == pytest.approx(-1.0, abs=1e-6)
    assert bad["wet_event_hit_rate"] == 0.0  # every event agrees in the wrong direction
    v = validate.verdict({"7-28 cm": bad})
    assert v["passed"] is False and v["checks"]["7-28 cm"]["anomaly_r_ok"] is False



def test_compare_refuses_to_score_thin_data():
    m = model_for(past=8)
    metrics = validate.compare(m, obs_from(m), "sm_7_28_m3m3", "7-28 cm")
    assert metrics["enough_data"] is False
    v = validate.verdict({"7-28 cm": metrics})
    assert v["passed"] is False and "overlapping days" in v["checks"]["7-28 cm"]["why"]


def test_verdict_passes_on_agreeing_signals_and_reports_the_numbers():
    m = model_for(rain={-12: 35, -4: 25})
    layers = {label: validate.compare(m, obs_from(m), col, label)
              for label, col in validate.LAYERS.items()}
    v = validate.verdict(layers)
    assert v["passed"] is True
    assert set(v["checks"]) == {"0-7 cm", "7-28 cm"}
    for label in ("0-7 cm", "7-28 cm"):
        assert v["checks"][label]["anomaly_r"] >= validate.MIN_ANOMALY_R
        # the drying rate is reported, but deliberately not part of the pass/fail
        assert v["checks"][label]["informational_drying"]["ratio"] == pytest.approx(1.0, abs=0.01)
        assert "not like-for-like" in v["checks"][label]["informational_drying"]["why_not_required"]


def test_plot_validation_writes_a_file(tmp_path):
    m = model_for(rain={-5: 30})
    out = validate.plot_validation(m, obs_from(m), tmp_path / "v.png")
    assert (tmp_path / "v.png").exists()
    assert out.endswith("v.png")
