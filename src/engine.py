"""Step 4: rules + water-balance decision engine (FAO-56 style, single crop coefficient).

Each day:  Dr = Dr_yesterday + ETc - effective_rain - irrigation, kept between 0 and TAW.
  Dr  = depletion of the root zone (mm of water missing)
  TAW = total available water in the root zone (soil capacity x root depth)
  RAW = p x TAW, the depletion at which the crop starts to feel stress
Irrigate when Dr reaches RAW. The LLM never decides: it only explains this result (Step 7).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

CROPS_PATH = Path("config/crops.json")
# Typical water content at the permanent wilting point and at field capacity, m3/m3.
# FAO-56 Table 19 style averages per texture class. Confirm with a local soil test, or use
# src/soil.py (SoilGrids + Saxton & Rawls), which measures it for the field itself.
SOIL_THETA_WP_M3M3 = {
    "sand": 0.05, "loamy_sand": 0.07, "sandy_loam": 0.10, "loam": 0.14, "silt_loam": 0.16,
    "silt": 0.16, "sandy_clay_loam": 0.18, "clay_loam": 0.20, "silty_clay_loam": 0.22,
    "sandy_clay": 0.24, "silty_clay": 0.26, "clay": 0.28,
}
SOIL_THETA_FC_M3M3 = {
    "sand": 0.13, "loamy_sand": 0.17, "sandy_loam": 0.21, "loam": 0.29, "silt_loam": 0.33,
    "silt": 0.34, "sandy_clay_loam": 0.34, "clay_loam": 0.36, "silty_clay_loam": 0.38,
    "sandy_clay": 0.38, "silty_clay": 0.40, "clay": 0.42,
}
# Available water per metre of soil, mm. Derived so the two tables can never drift apart.
SOIL_TAW_MM_PER_M = {
    s: round((SOIL_THETA_FC_M3M3[s] - SOIL_THETA_WP_M3M3[s]) * 1000) for s in SOIL_THETA_WP_M3M3
}
DEFAULT_SOIL = "loam"
IRRIGATION_EFFICIENCY = 0.7  # surface (flood/furrow): ~30% is lost in conveyance and runoff
MIN_IRRIGATION_MM = 5.0  # a smaller top-up is not worth the water and the labour

MAX_EXTEND_DAYS = 21  # beyond the forecast: assume no rain and recent ET0
SOON_DAYS = 2
SAT_MAX_AGE_DAYS = 10
SAT_MAX_GAP_DAYS = 15
NDMI_DROP = -0.05  # heuristic: not calibrated yet, tune with the farmer's logs
RAIN_WORTH_NOTING_MM = 10.0  # forecast rain big enough to mention next to an irrigation
NEARLY_FULL_PCT = 80  # root zone this full: more water now is wasted
MAX_RECHECK_DAYS = 7


def load_crop(name: str, path: Path = CROPS_PATH) -> dict:
    crops = json.loads(Path(path).read_text())
    if name not in crops:
        raise KeyError(f"crop '{name}' is not in {path}")
    return crops[name]


def stage_and_kc(das: int, crop: dict):
    """Growth stage and crop coefficient Kc for `das` days after sowing (FAO-56 piecewise curve)."""
    s, kc = crop["stages_days"], crop["kc"]
    ini, dev, mid, late = s["initial"], s["development"], s["mid"], s["late"]
    if das < 0:
        return None, None
    if das < ini:
        return "initial", kc["ini"]
    if das < ini + dev:
        return "development", kc["ini"] + (das - ini) / dev * (kc["mid"] - kc["ini"])
    if das < ini + dev + mid:
        return "mid-season", kc["mid"]
    if das < ini + dev + mid + late:
        return "late-season", kc["mid"] + (das - ini - dev - mid) / late * (kc["end"] - kc["mid"])
    return "finished", None


def root_depth_m(das: int, crop: dict) -> float:
    """Roots deepen linearly until the end of the development stage."""
    s, r = crop["stages_days"], crop["root_depth_m"]
    f = min(max(das / (s["initial"] + s["development"]), 0), 1)
    return r["min"] + f * (r["max"] - r["min"])


def soil_bottom_cm(crop: dict) -> int:
    """Soil depth to average over: the deepest the roots get (TAW must cover the whole root zone)."""
    return int(round(crop["root_depth_m"]["max"] * 100))


def effective_rain_mm(rain_mm: float) -> float:
    """Small showers (<2 mm) are lost to evaporation; otherwise ~80% reaches the soil."""
    return 0.0 if rain_mm < 2 else 0.8 * rain_mm


def day_params(das: int, crop: dict, taw_per_m: float) -> dict:
    stage, kc = stage_and_kc(das, crop)
    taw = taw_per_m * root_depth_m(das, crop)
    return {"stage": stage, "kc": kc, "taw": taw, "raw": crop["depletion_fraction_p"] * taw}


def soil_water_params(soil: str | None = None, soil_params: dict | None = None) -> dict:
    """Which water-holding numbers to use: the field's own, or a texture-class guess.

    soil_params comes from src/soil.py (SoilGrids + Saxton & Rawls) and carries
    theta_fc_m3m3, theta_wp_m3m3 and taw_mm_per_m measured for this field.
    """
    if soil_params:
        return {
            "name": soil_params.get("texture_class", soil or DEFAULT_SOIL),
            "theta_fc_m3m3": soil_params["theta_fc_m3m3"],
            "theta_wp_m3m3": soil_params["theta_wp_m3m3"],
            "taw_mm_per_m": soil_params["taw_mm_per_m"],
            "source": soil_params.get("source", "field soil data"),
            "assumed": False,
        }
    name = soil or DEFAULT_SOIL
    if name not in SOIL_TAW_MM_PER_M:
        raise KeyError(f"soil '{name}' is unknown; use one of {sorted(SOIL_TAW_MM_PER_M)}")
    return {
        "name": name,
        "theta_fc_m3m3": SOIL_THETA_FC_M3M3[name],
        "theta_wp_m3m3": SOIL_THETA_WP_M3M3[name],
        "taw_mm_per_m": SOIL_TAW_MM_PER_M[name],
        "source": "texture-class table (FAO-56 typical values)",
        "assumed": soil is None,
    }


def soil_moisture_m3m3(dr_mm: float, root_zone_m: float, theta_fc_m3m3: float,
                       theta_wp_m3m3: float) -> float:
    """Root-zone water content now: field capacity minus the water the crop has used."""
    theta = theta_fc_m3m3 - dr_mm / (root_zone_m * 1000.0)
    return min(max(theta, theta_wp_m3m3), theta_fc_m3m3)


def water_available_pct(dr_mm: float, taw_mm: float) -> int:
    """How full the root zone is: 100% = field capacity, 0% = wilting point."""
    return int(round(100 * (1 - dr_mm / taw_mm))) if taw_mm else 0


def irrigation_depth_mm(dr_mm: float, taw_mm: float, efficiency: float = IRRIGATION_EFFICIENCY,
                        min_mm: float = MIN_IRRIGATION_MM) -> dict:
    """How much water to put on.

    net_mm   refills the root zone back to field capacity (FAO-56: replace what is depleted)
    gross_mm the depth to actually release, because surface irrigation loses a share
    """
    net_mm = min(max(dr_mm, 0.0), taw_mm)
    if net_mm < min_mm:
        return {"needed": False, "net_mm": 0.0, "gross_mm": 0.0, "efficiency": efficiency}
    return {
        "needed": True,
        "net_mm": round(net_mm, 1),
        "gross_mm": round(net_mm / efficiency, 1),
        "efficiency": efficiency,
    }


def pump_hours(gross_mm: float, area_m2: float | None, flow_lps: float | None) -> float | None:
    """Hours of pumping to put `gross_mm` on `area_m2` with a pump giving `flow_lps` litres per second."""
    if not area_m2 or not flow_lps or flow_lps <= 0:
        return None
    volume_l = gross_mm * area_m2  # 1 mm over 1 m2 is 1 litre
    return round(volume_l / flow_lps / 3600, 1)


def farmer_actions(action: str, depth: dict, days: int | None, rain_next_5d_mm: float,
                   water_pct: int, irrigation_known: bool, crop: dict) -> list[dict]:
    """What the farmer can do, as fixed codes plus numbers. src/explain.py turns them into words.

    Deterministic on purpose: the LLM may rephrase these, never invent new ones.
    """
    out: list[dict] = []
    irrigate = action in ("irrigate_now", "irrigate_soon")
    if irrigate and depth.get("needed"):
        out.append({"code": "apply_depth", "gross_mm": depth["gross_mm"], "pump_hours": depth.get("pump_hours")})
        out.append({"code": "irrigate_cool_hours"})
        if rain_next_5d_mm >= RAIN_WORTH_NOTING_MM:
            out.append({"code": "rain_forecast_counted", "rain_mm": round(rain_next_5d_mm)})
    if action == "watch":
        out.append({"code": "hand_check"})
    if action == "wait":
        if water_pct >= NEARLY_FULL_PCT:
            out.append({"code": "soil_nearly_full", "water_pct": water_pct})
        recheck = MAX_RECHECK_DAYS if days is None else max(1, min(days - SOON_DAYS, MAX_RECHECK_DAYS))
        out.append({"code": "recheck_in", "days": recheck})
    if action in ("irrigate_now", "irrigate_soon", "watch", "wait"):
        out.extend({"code": tip} for tip in crop.get("water_saving_tips", []))
    if not irrigation_known:
        out.append({"code": "log_irrigation"})
    return out


def extend_weather(df: pd.DataFrame, days: int = MAX_EXTEND_DAYS) -> pd.DataFrame:
    """Append days after the forecast: no rain, ET0 = average of the last 7 observed days.

    (Not the last forecast days: those can be unusually cloudy and would understate water use.)
    """
    observed = df[~df["is_forecast"].astype(bool)] if "is_forecast" in df.columns else df
    et0 = (observed if len(observed) else df)["et0_mm"].tail(7).mean()
    last = df["date"].max()
    base = df.copy()
    base["extrapolated"] = False
    extra = pd.DataFrame([
        {"date": last + pd.Timedelta(days=i), "rain_mm": 0.0, "et0_mm": et0, "extrapolated": True}
        for i in range(1, days + 1)
    ])
    return pd.concat([base, extra], ignore_index=True)


def simulate(df, crop, taw_per_m, sowing, start, dr0, irrigations=None) -> pd.DataFrame:
    """Daily water balance for every day after `start` (the day Dr was known to be dr0)."""
    irrigations = irrigations or {}
    rows, dr = [], dr0
    for r in df[df["date"] > start].itertuples():
        p = day_params((r.date - sowing).days, crop, taw_per_m)
        if p["kc"] is None:  # crop cycle over
            break
        dr += r.et0_mm * p["kc"] - effective_rain_mm(r.rain_mm)
        if r.date in irrigations:
            depth = irrigations[r.date]  # None = full refill
            dr = 0.0 if depth is None else dr - depth
        dr = min(max(dr, 0.0), p["taw"])
        rows.append({"date": r.date, "dr_mm": dr, "raw_mm": p["raw"], "taw_mm": p["taw"],
                     "extrapolated": bool(r.extrapolated)})
    return pd.DataFrame(rows, columns=["date", "dr_mm", "raw_mm", "taw_mm", "extrapolated"])


def _dr_on(sim: pd.DataFrame, day: pd.Timestamp):
    """Depletion on `day` (last simulated day not after it), or None."""
    rows = sim[sim["date"] <= day]
    return float(rows["dr_mm"].iloc[-1]) if len(rows) else None


def decide(weather: pd.DataFrame, crop_name: str, sowing, last_irrigation=None, soil: str | None = None,
           satellite: dict | None = None, today=None, irrigations: dict | None = None,
           crops_path: Path = CROPS_PATH, soil_params: dict | None = None,
           field_area_m2: float | None = None, pump_flow_lps: float | None = None) -> dict:
    """Structured irrigation advice for today. `weather` comes from weather.fetch_weather().

    `soil_params` (from src/soil.py) overrides the texture-class guess with this field's
    own theta_fc / theta_wp / TAW. With `field_area_m2` and `pump_flow_lps` the depth is
    also given as water volume (m3) and hours of pumping.
    """
    crop = load_crop(crop_name, crops_path)
    today = pd.Timestamp(today if today is not None else pd.Timestamp.today()).normalize()
    sowing = pd.Timestamp(sowing).normalize()
    base = {"crop": crop_name, "date": str(today.date())}
    if today < sowing:
        return {**base, "action": "not_sown", "confidence": "n/a",
                "reasons": [f"{crop_name} is not sown yet (sowing date {sowing.date()})."]}

    notes: list[str] = []
    soil_w = soil_water_params(soil, soil_params)
    taw_per_m = soil_w["taw_mm_per_m"]
    if soil_w["assumed"]:
        notes.append(f"soil type assumed: {soil_w['name']}")
    if not crop.get("verified", False):
        notes.append("crop parameters are placeholders, not verified locally")


    w = weather.copy()
    w["date"] = pd.to_datetime(w["date"])
    w["et0_mm"] = w["et0_mm"].ffill().bfill()
    if w["date"].max() < today:
        notes.append("weather data ends before today")

    # Where does the water balance start? At the last irrigation, else at sowing (soil assumed full).
    irrig = {pd.Timestamp(k): v for k, v in (irrigations or {}).items()}
    last_irr = pd.Timestamp(last_irrigation).normalize() if last_irrigation else None
    if last_irr is not None and last_irr > today:
        last_irr = None  # a future date is a placeholder, not a record
    start, dr0, truncated = sowing, 0.0, False
    if last_irr is None:
        notes.append("no irrigation date given: assumed no irrigation since the soil was last refilled by rain")
    elif sowing <= last_irr <= today:
        start = last_irr
    first = w["date"].min()
    uncertain_start = start < first  # weather history is shorter than the period we need
    if uncertain_start:
        start = first - pd.Timedelta(days=1)
        p0 = day_params(max((start - sowing).days, 0), crop, taw_per_m)
        dr0 = 0.5 * p0["raw"]
    ext = extend_weather(w)
    sim = simulate(ext, crop, taw_per_m, sowing, start, dr0, irrig)
    if uncertain_start:
        # Try the driest and the wettest possible starting soil. If heavy rain makes them
        # end up the same today, the unknown start no longer matters.
        ends = [_dr_on(simulate(ext, crop, taw_per_m, sowing, start, d, irrig), today) for d in (0.0, p0["taw"])]
        if None in ends or abs(ends[0] - ends[1]) >= 1.0:
            truncated = True
            notes.append("weather history starts after the last irrigation: starting soil water is a guess")
    das_today = (today - sowing).days
    p_today = day_params(das_today, crop, taw_per_m)
    if p_today["kc"] is None or sim.empty:
        return {**base, "action": "season_over", "confidence": "n/a",
                "reasons": ["Crop cycle is complete or no weather data is available."]}
    past = sim[sim["date"] <= today]
    dr_today = float(past["dr_mm"].iloc[-1]) if len(past) else dr0
    raw_today = float(past["raw_mm"].iloc[-1]) if len(past) else p_today["raw"]
    taw_today = float(past["taw_mm"].iloc[-1]) if len(past) else p_today["taw"]
    root_zone_m = root_depth_m(das_today, crop)
    theta_now = soil_moisture_m3m3(dr_today, root_zone_m, soil_w["theta_fc_m3m3"], soil_w["theta_wp_m3m3"])
    depth = irrigation_depth_mm(dr_today, taw_today)
    if field_area_m2:
        depth["water_m3"] = round(depth["gross_mm"] * field_area_m2 / 1000, 1)
    depth["pump_hours"] = pump_hours(depth["gross_mm"], field_area_m2, pump_flow_lps) if depth["needed"] else None


    # When does depletion reach the stress point?
    days, basis = None, "none"
    if dr_today >= raw_today:
        days, basis = 0, "today"
    else:
        hit = sim[(sim["date"] > today) & (sim["dr_mm"] >= sim["raw_mm"])]
        if len(hit):
            days = int((hit.iloc[0]["date"] - today).days)
            basis = "extrapolated" if hit.iloc[0]["extrapolated"] else "forecast"
    action = "irrigate_now" if days == 0 else "irrigate_soon" if days is not None and days <= SOON_DAYS else "wait"

    rain_past = float(w[w["date"] <= today].tail(7)["rain_mm"].sum())
    rain_next = float(w[w["date"] > today].head(5)["rain_mm"].sum())
    today_w = w[w["date"] == today]
    etc_today = float(today_w["et0_mm"].iloc[0]) * p_today["kc"] if len(today_w) else None
    pct = round(100 * dr_today / raw_today)

    reasons = [f"Root-zone water is down {dr_today:.0f} mm of the {taw_today:.0f} mm the soil can hold "
               f"({100 - water_available_pct(dr_today, taw_today)}% used); "
               f"stress starts at {raw_today:.0f} mm ({pct}% of the way)."]
    if depth["needed"] and action in ("irrigate_now", "irrigate_soon"):
        reasons.append(f"To refill the root zone apply about {depth['gross_mm']:.0f} mm of water in the field "
                       f"({depth['net_mm']:.0f} mm net; surface irrigation is about {depth['efficiency']:.0%} efficient).")
    elif depth["needed"]:
        reasons.append(f"If you irrigate today it would take about {depth['gross_mm']:.0f} mm to refill the root zone.")
    if etc_today is not None:
        reasons.append(f"The {crop_name} ({p_today['stage']}) is using about {etc_today:.1f} mm of water per day.")

    reasons.append(f"Rain: {rain_past:.0f} mm in the last 7 days, {rain_next:.0f} mm forecast for the next 5 days.")
    if days is None:
        reasons.append("The stress point is not reached within the next 3 weeks.")
    elif days > 0:
        extra = " (assuming no rain after the 5-day forecast)" if basis == "extrapolated" else ""
        reasons.append(f"At this rate the stress point is reached in about {days} day(s){extra}.")

    # Satellite cross-check: a fast fall in leaf-water index can flag stress the water balance misses.
    sat_age = None
    if satellite and satellite.get("available"):
        sat_age = satellite["age_days"]
        ch, gap = satellite.get("ndmi_change"), satellite.get("days_between")
        if sat_age > SAT_MAX_AGE_DAYS:
            notes.append(f"newest clear satellite view is {sat_age} days old")
        elif (action == "wait" and ch is not None and gap is not None
              and gap <= SAT_MAX_GAP_DAYS and ch <= NDMI_DROP):
            action = "watch"
            reasons.append(f"Satellite leaf-water index fell by {abs(ch):.2f} in {gap} days. "
                           "Check the soil by hand before deciding.")
    else:
        notes.append("no usable satellite view")

    confidence = "low" if truncated or len(notes) >= 3 else "medium" if notes else "high"
    water_pct = water_available_pct(dr_today, taw_today)
    actions = farmer_actions(action, depth, days, rain_next, water_pct, last_irr is not None, crop)
    return {
        **base, "action": action, "stage": p_today["stage"], "days_after_sowing": int(das_today),
        "depletion_mm": round(dr_today, 1), "allowed_depletion_mm": round(raw_today, 1),
        "root_zone_capacity_mm": round(taw_today, 1), "depletion_pct_of_allowed": int(pct),
        "days_to_irrigation": days, "basis": basis,
        "crop_use_mm_per_day": None if etc_today is None else round(etc_today, 1),
        "rain_last_7d_mm": round(rain_past, 1), "rain_next_5d_mm": round(rain_next, 1),
        "soil": soil_w["name"], "soil_source": soil_w["source"],
        "root_depth_m": round(root_zone_m, 2),
        "soil_moisture_m3m3": round(theta_now, 3),
        "field_capacity_m3m3": soil_w["theta_fc_m3m3"], "wilting_point_m3m3": soil_w["theta_wp_m3m3"],
        "water_available_pct": water_pct,
        "irrigation_depth_mm": depth,
        "satellite_age_days": sat_age, "reasons": reasons, "actions": actions,
        "confidence": confidence, "confidence_notes": notes,
    }



if __name__ == "__main__":
    # usage: python -m src.engine [sowing_date] [last_irrigation_date]   (dates as YYYY-MM-DD)
    from src.weather import fetch_weather

    cfg = json.loads(Path("config/field.json").read_text())
    sowing = sys.argv[1] if len(sys.argv) > 1 else cfg["sowing_date"]
    last_irr = sys.argv[2] if len(sys.argv) > 2 else cfg.get("last_irrigation_date")
    wx = fetch_weather(cfg["latitude"], cfg["longitude"], past_days=90, forecast_days=6, use_cache=False)
    sat = None
    try:
        from src.satellite import load_field_geometry
        from src.timeseries import build_series, satellite_summary

        sat = satellite_summary(build_series(load_field_geometry()))
    except Exception as e:  # noqa: BLE001
        print("satellite unavailable:", e)
    soil_params = None
    try:  # this field's own soil instead of the texture-class guess
        from src.soil import field_soil

        soil_params = field_soil(cfg["latitude"], cfg["longitude"],
                                 bottom_cm=soil_bottom_cm(load_crop(cfg["crop"])))
    except Exception as e:  # noqa: BLE001
        print("field soil unavailable, using the texture-class table:", e)
    print(json.dumps(decide(wx, cfg["crop"], sowing, last_irr, cfg.get("soil"), sat,
                            soil_params=soil_params), indent=2))
