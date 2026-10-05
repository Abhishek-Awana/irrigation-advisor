import json
from pathlib import Path

import pandas as pd
import pytest

from src import engine

TODAY = pd.Timestamp("2026-10-04")
SOWING = TODAY - pd.Timedelta(days=70)  # mid-season: Kc 1.15, roots 1.0 m
# loam: TAW 150 mm, RAW 82.5 mm. ET0 5 mm/day -> ETc 5.75 mm/day.


@pytest.fixture
def crops_file(tmp_path):
    crops = json.loads(Path("config/crops.json").read_text())
    crops["wheat"]["verified"] = True
    p = tmp_path / "crops.json"
    p.write_text(json.dumps(crops))
    return p


def weather(rain=None, past=30, future=5, et0=5.0):
    """Daily table ending `future` days after TODAY. rain = {day_offset: mm}."""
    dates = pd.date_range(TODAY - pd.Timedelta(days=past - 1), TODAY + pd.Timedelta(days=future))
    rain = rain or {}
    return pd.DataFrame({
        "date": dates,
        "rain_mm": [float(rain.get((d - TODAY).days, 0.0)) for d in dates],
        "et0_mm": et0,
        "is_forecast": dates > TODAY,
    })


def run(days_since_irrigation, crops_file, rain=None, satellite=None, **kw):
    return engine.decide(weather(rain, **kw), "wheat", SOWING, TODAY - pd.Timedelta(days=days_since_irrigation),
                         soil="loam", satellite=satellite, today=TODAY, crops_path=crops_file)


def test_long_dry_spell_means_irrigate_now(crops_file):
    r = run(16, crops_file)  # 16 x 5.75 = 92 mm > 82.5
    assert r["action"] == "irrigate_now" and r["days_to_irrigation"] == 0


def test_stress_point_in_two_days_means_irrigate_soon(crops_file):
    r = run(13, crops_file)  # 74.75 today, 86.25 two days later
    assert r["action"] == "irrigate_soon" and r["days_to_irrigation"] == 2


def test_recently_irrigated_means_wait_and_extrapolates(crops_file):
    r = run(3, crops_file)
    assert r["action"] == "wait" and r["days_to_irrigation"] == 12 and r["basis"] == "extrapolated"


def test_heavy_rain_yesterday_resets_the_clock(crops_file):
    r = run(16, crops_file, rain={-1: 60})  # would be irrigate_now without the rain
    assert r["action"] == "wait" and r["depletion_mm"] < 50


def test_forecast_rain_delays_irrigation(crops_file):
    r = run(13, crops_file, rain={1: 40})  # baseline is irrigate_soon
    assert r["action"] == "wait" and r["days_to_irrigation"] > 2


def test_not_sown_yet(crops_file):
    r = engine.decide(weather(), "wheat", TODAY + pd.Timedelta(days=10), today=TODAY, crops_path=crops_file)
    assert r["action"] == "not_sown"


def test_falling_leaf_water_index_raises_a_watch_flag(crops_file):
    sat = {"available": True, "age_days": 2, "ndmi_change": -0.08, "days_between": 6}
    assert run(3, crops_file, satellite=sat)["action"] == "watch"
    calm = {**sat, "ndmi_change": -0.01}
    assert run(3, crops_file, satellite=calm)["action"] == "wait"


def test_satellite_never_overrides_irrigate_now(crops_file):
    sat = {"available": True, "age_days": 2, "ndmi_change": 0.0, "days_between": 6}
    assert run(16, crops_file, satellite=sat)["action"] == "irrigate_now"


def test_confidence_levels(crops_file):
    good = {"available": True, "age_days": 2, "ndmi_change": 0.0, "days_between": 6}
    assert run(3, crops_file, satellite=good)["confidence"] == "high"
    assert run(3, crops_file)["confidence"] == "medium"  # no satellite view
    short = run(20, crops_file, past=5)  # history shorter than the irrigation record
    assert short["confidence"] == "low"


def test_kc_curve():
    crop = engine.load_crop("wheat", Path("config/crops.json"))
    assert engine.stage_and_kc(0, crop) == ("initial", 0.3)
    assert engine.stage_and_kc(60, crop) == ("mid-season", 1.15)
    assert engine.stage_and_kc(130, crop) == ("finished", None)
    assert 0.3 < engine.stage_and_kc(40, crop)[1] < 1.15


def test_heavy_rain_makes_a_short_weather_history_harmless(crops_file):
    # irrigation 40 days ago is outside the 10-day weather table, but 300 mm of rain erases the unknown start
    r = engine.decide(weather({-3: 300}, past=10), "wheat", SOWING, TODAY - pd.Timedelta(days=40),
                      soil="loam", today=TODAY, crops_path=crops_file)
    assert not any("history" in n for n in r["confidence_notes"])
    assert r["confidence"] != "low"


def test_future_irrigation_date_is_treated_as_unknown(crops_file):
    r = engine.decide(weather(), "wheat", SOWING, TODAY + pd.Timedelta(days=5), soil="loam",
                      today=TODAY, crops_path=crops_file)
    assert any("no irrigation date" in n for n in r["confidence_notes"])


def test_sugarcane_in_october_is_mid_season():
    crop = engine.load_crop("sugarcane", Path("config/crops.json"))
    das = (TODAY - pd.Timestamp("2026-04-15")).days  # ~172 days after planting
    p = engine.day_params(das, crop, engine.SOIL_TAW_MM_PER_M["loam"])
    assert p["stage"] == "mid-season" and p["kc"] == 1.25
    assert abs(p["raw"] - 0.65 * 150 * 1.2) < 1e-6  # roots at 1.2 m, p = 0.65


def test_dry_spell_extension_uses_recent_observed_et0():
    w = weather(et0=6.0)
    w.loc[w["is_forecast"], "et0_mm"] = 2.0  # cloudy forecast days must not set the long-term rate
    ext = engine.extend_weather(w)
    assert abs(ext[ext["extrapolated"]]["et0_mm"].iloc[0] - 6.0) < 1e-9


# --- Step 4B: soil moisture and how much water to apply -------------------------------

def test_taw_table_still_matches_the_named_soil_classes():
    for name in engine.SOIL_TAW_MM_PER_M:
        assert engine.SOIL_TAW_MM_PER_M[name] == round(
            (engine.SOIL_THETA_FC_M3M3[name] - engine.SOIL_THETA_WP_M3M3[name]) * 1000)
    assert engine.SOIL_TAW_MM_PER_M["loam"] == 150  # existing crops must not shift


def test_soil_moisture_spans_field_capacity_to_wilting_point():
    loam = engine.soil_water_params("loam")
    fc, wp = loam["theta_fc_m3m3"], loam["theta_wp_m3m3"]
    assert engine.soil_moisture_m3m3(0.0, 1.0, fc, wp) == pytest.approx(fc)
    assert engine.soil_moisture_m3m3(150.0, 1.0, fc, wp) == pytest.approx(wp)  # 150 mm over 1 m
    # 75 mm missing from a 1 m root zone = half of the 150 mm the loam can hold
    assert engine.soil_moisture_m3m3(75.0, 1.0, fc, wp) == pytest.approx((fc + wp) / 2)
    assert engine.soil_moisture_m3m3(500.0, 1.0, fc, wp) == pytest.approx(wp)  # never below wp


def test_soil_moisture_depends_on_root_depth():
    loam = engine.soil_water_params("loam")
    fc, wp = loam["theta_fc_m3m3"], loam["theta_wp_m3m3"]
    shallow = engine.soil_moisture_m3m3(60.0, 0.3, fc, wp)
    deep = engine.soil_moisture_m3m3(60.0, 1.2, fc, wp)
    assert shallow < deep < fc  # the same mm dries a shallow root zone far more


def test_water_available_pct():
    assert engine.water_available_pct(0.0, 150.0) == 100
    assert engine.water_available_pct(150.0, 150.0) == 0
    assert engine.water_available_pct(75.0, 150.0) == 50
    assert engine.water_available_pct(10.0, 0.0) == 0  # no soil capacity, no claim


def test_irrigation_depth_refills_the_root_zone():
    d = engine.irrigation_depth_mm(60.0, 150.0)
    assert d["needed"] is True
    assert d["net_mm"] == 60.0
    assert d["gross_mm"] == pytest.approx(60.0 / engine.IRRIGATION_EFFICIENCY, abs=0.1)
    assert d["efficiency"] == engine.IRRIGATION_EFFICIENCY


def test_irrigation_depth_caps_at_capacity_and_skips_trivial_top_ups():
    assert engine.irrigation_depth_mm(500.0, 150.0)["net_mm"] == 150.0  # cannot store more
    tiny = engine.irrigation_depth_mm(2.0, 150.0)
    assert tiny["needed"] is False and tiny["gross_mm"] == 0.0


def test_decide_reports_soil_moisture_and_the_depth_to_apply(crops_file):
    r = run(16, crops_file)  # 92 mm depleted from a 150 mm loam root zone
    assert r["action"] == "irrigate_now"
    d = r["irrigation_depth_mm"]
    assert d["needed"] is True
    assert d["net_mm"] == pytest.approx(r["depletion_mm"], abs=0.2)
    assert d["gross_mm"] == pytest.approx(d["net_mm"] / engine.IRRIGATION_EFFICIENCY, abs=0.2)
    assert r["water_available_pct"] == pytest.approx(100 * (1 - r["depletion_mm"] / r["root_zone_capacity_mm"]), abs=1)
    assert 0.14 <= r["soil_moisture_m3m3"] <= 0.29
    assert r["root_depth_m"] == pytest.approx(1.0)  # wheat at 70 days, roots at max
    assert any("apply about" in reason for reason in r["reasons"])


def test_decide_uses_the_fields_own_soil_when_given(crops_file):
    field = {"texture_class": "clay_loam", "theta_fc_m3m3": 0.3177, "theta_wp_m3m3": 0.1809,
             "taw_mm_per_m": 136.8, "source": "test SoilGrids profile"}
    r = engine.decide(weather(), "wheat", SOWING, TODAY - pd.Timedelta(days=16), soil="loam",
                      today=TODAY, crops_path=crops_file, soil_params=field)
    assert r["soil"] == "clay_loam" and r["soil_source"] == "test SoilGrids profile"
    assert r["root_zone_capacity_mm"] == pytest.approx(136.8, abs=0.2)  # not the 150 mm loam guess
    assert r["field_capacity_m3m3"] == 0.3177
    assert not any("soil type assumed" in n for n in r["confidence_notes"])


def test_unknown_soil_name_is_rejected():
    with pytest.raises(KeyError):
        engine.soil_water_params("moon_dust")


def test_waiting_advice_still_says_what_a_refill_would_take(crops_file):
    r = run(3, crops_file)  # only 17 mm down: wait
    assert r["action"] == "wait"
    assert r["irrigation_depth_mm"]["net_mm"] == pytest.approx(r["depletion_mm"], abs=0.2)
    assert any("If you irrigate today" in reason for reason in r["reasons"])


def test_soil_bottom_cm_follows_max_root_depth():
    assert engine.soil_bottom_cm({"root_depth_m": {"min": 0.3, "max": 1.2}}) == 120


def test_pump_hours_from_depth_area_and_flow():
    # 84 mm on 1 hectare = 840 m3; at 10 L/s that is 84,000 s = 23.3 h
    assert engine.pump_hours(84.0, 10_000, 10.0) == pytest.approx(23.3)
    assert engine.pump_hours(84.0, None, 10.0) is None
    assert engine.pump_hours(84.0, 10_000, 0) is None


def test_decide_gives_volume_and_pump_hours_when_field_is_known(crops_file):
    r = engine.decide(weather(), "wheat", SOWING, TODAY - pd.Timedelta(days=16), soil="loam",
                      today=TODAY, crops_path=crops_file, field_area_m2=10_000, pump_flow_lps=10.0)
    d = r["irrigation_depth_mm"]
    assert d["water_m3"] == pytest.approx(d["gross_mm"] * 10, abs=0.1)
    assert d["pump_hours"] == engine.pump_hours(d["gross_mm"], 10_000, 10.0)
    assert r["actions"][0] == {"code": "apply_depth", "gross_mm": d["gross_mm"], "pump_hours": d["pump_hours"]}


def test_actions_for_waiting_and_unknown_irrigation():
    crop = {"water_saving_tips": ["trash_mulch"]}
    depth = {"needed": False, "gross_mm": 0.0}
    acts = engine.farmer_actions("wait", depth, 12, 0.0, 90, irrigation_known=False, crop=crop)
    codes = [a["code"] for a in acts]
    assert codes == ["soil_nearly_full", "recheck_in", "trash_mulch", "log_irrigation"]
    assert acts[1]["days"] == 7  # never more than a week between checks


def test_actions_when_irrigating_with_rain_forecast():
    depth = {"needed": True, "gross_mm": 80.0, "pump_hours": 5.0}
    acts = engine.farmer_actions("irrigate_now", depth, 0, 15.0, 30, irrigation_known=True, crop={})
    assert [a["code"] for a in acts] == ["apply_depth", "irrigate_cool_hours", "rain_forecast_counted"]


def test_watch_asks_for_a_hand_check():
    acts = engine.farmer_actions("watch", {"needed": False}, 8, 0.0, 50, True, {})
    assert acts == [{"code": "hand_check"}]
