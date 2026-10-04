"""Step 6: does the water balance agree with an independent dataset?

We replay src/engine.py's daily balance over a past window on ERA5-Land rain and ET0, then
compare it with ERA5-Land soil moisture fetched from Open-Meteo's archive.

Absolute levels cannot match, and should not be expected to: ERA5-Land is a ~9 km grid with
its own soil, and its 0-7 cm and 7-28 cm layers are much shallower than a sugarcane root
zone. What has to match is the *change* from one day to the next - does our simple bucket
dry down and wet up on the same days as a state-of-the-art land-surface model? That is what
the pass/fail is based on.

Honest limits: this is a model-vs-model comparison, not ground truth. In-situ probes (ISMN)
or SMAP would be the next level, and both need an account.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src import engine, soil, weather

MIN_DAYS = 20  # fewer overlapping days than this and a correlation means nothing
MIN_ANOMALY_R = 0.5  # project threshold: the two soil-water signals must move together
MIN_EVENT_HIT_RATE = 0.6  # share of big-rain / strong-drying days that must agree in sign
WET_MM = 10.0  # a "wet day" for the event check
DRY_ET0_MM = 4.0  # a "drying day": no rain and at least this much reference evaporation
SPINUP_DAYS = 15  # settle the balance before the window we actually score
VARIANCE_FLOOR = 1e-12  # below this the series is flat and a correlation is meaningless

LAYERS = {"0-7 cm": "sm_0_7_m3m3", "7-28 cm": "sm_7_28_m3m3"}
LAYER_THICKNESS_M = {"0-7 cm": 0.07, "7-28 cm": 0.21}  # to turn theta changes into mm
REPORT_JSON = Path("data/validation.json")
REPORT_PNG = Path("data/validation.png")



def model_series(wx: pd.DataFrame, crop_name: str, sowing, soil_name: str | None = None,
                 soil_params: dict | None = None, start=None, dr0: float = 0.0,
                 irrigations: dict | None = None, crops_path: Path = engine.CROPS_PATH) -> pd.DataFrame:
    """Daily modelled root-zone water content in m3/m3 over a past window.

    `start` is the day the root zone held `dr0` mm of depletion (0 = field capacity).
    """
    crop = engine.load_crop(crop_name, crops_path)
    soil_w = engine.soil_water_params(soil_name, soil_params)
    sowing = pd.Timestamp(sowing)
    wx = wx.copy()
    wx["date"] = pd.to_datetime(wx["date"])
    wx["extrapolated"] = False
    start = pd.Timestamp(start) if start is not None else wx["date"].min() - pd.Timedelta(days=1)
    sim = engine.simulate(wx, crop, soil_w["taw_mm_per_m"], sowing, start, dr0, irrigations)
    if sim.empty:
        return pd.DataFrame(columns=["date", "theta_m3m3", "dr_mm", "taw_mm", "root_depth_m",
                                     "rain_mm", "et0_mm"])
    rows = []
    for r in sim.itertuples():
        root_zone_m = engine.root_depth_m((r.date - sowing).days, crop)
        rows.append({
            "date": r.date, "dr_mm": r.dr_mm, "taw_mm": r.taw_mm, "root_depth_m": root_zone_m,
            "theta_m3m3": engine.soil_moisture_m3m3(
                r.dr_mm, root_zone_m, soil_w["theta_fc_m3m3"], soil_w["theta_wp_m3m3"]),
        })
    model = pd.DataFrame(rows)
    return model.merge(wx[["date", "rain_mm", "et0_mm"]], on="date", how="left")


def compare(model_df: pd.DataFrame, obs_df: pd.DataFrame, obs_col: str = "sm_7_28_m3m3",
            label: str = "7-28 cm", obs_thickness_m: float | None = None) -> dict:
    """How well the modelled root-zone water tracks an observed layer.

    Two families of numbers:
      level_*     absolute differences in m3/m3. A depletion bucket is capped at field
                  capacity, so in a monsoon these are expected to be large and biased.
      drying_*    how fast water is actually lost on rain-free days, in mm/day. This is
                  depth-independent and is what decides when irrigation is due.
    """
    thickness_m = obs_thickness_m or LAYER_THICKNESS_M.get(label, 0.21)
    m = model_df.merge(obs_df[["date", obs_col]], on="date", how="inner").dropna(subset=["theta_m3m3", obs_col])
    n = len(m)
    if n < 3:
        return {"layer": label, "n_days": int(n), "enough_data": False}
    err = m["theta_m3m3"] - m[obs_col]
    depth_m = m["root_depth_m"] if "root_depth_m" in m.columns else pd.Series(1.0, index=m.index)
    m = m.assign(d_model=m["theta_m3m3"].diff(), d_obs=m[obs_col].diff(), root_m=depth_m)
    m = m.dropna(subset=["d_model", "d_obs"])

    def corr(a: pd.Series, b: pd.Series) -> float:
        """Pearson r, or NaN when either series is flat and r would be meaningless."""
        if len(a) < 3:
            return float("nan")
        da, db = a - a.mean(), b - b.mean()
        denom = float(np.sqrt((da ** 2).sum() * (db ** 2).sum()))
        if denom < VARIANCE_FLOOR:
            return float("nan")
        return float((da * db).sum() / denom)

    def hit_rate(days: pd.DataFrame) -> float | None:
        if days.empty:
            return None
        return round(float(((days["d_model"] > 0) == (days["d_obs"] > 0)).mean()), 2)

    wet = m[m["rain_mm"] >= WET_MM]
    dry = m[(m["rain_mm"] == 0) & (m["et0_mm"] >= DRY_ET0_MM)]
    # mm/day of water lost: theta change x the layer's own thickness, so both sides are mm
    mm_model = -m["d_model"] * m["root_m"] * 1000
    mm_obs = -m["d_obs"] * thickness_m * 1000
    drying = m[m["rain_mm"] == 0]
    model_drying_mm = round(float(mm_model[m["rain_mm"] == 0].mean()), 2) if len(drying) else None
    obs_drying_mm = round(float(mm_obs[m["rain_mm"] == 0].mean()), 2) if len(drying) else None
    ratio = (round(model_drying_mm / obs_drying_mm, 2)
             if model_drying_mm is not None and obs_drying_mm else None)
    return {
        "layer": label, "n_days": int(n), "enough_data": n >= MIN_DAYS,
        "level_bias_m3m3": round(float(err.mean()), 4),
        "level_rmse_m3m3": round(float(np.sqrt((err ** 2).mean())), 4),
        "anomaly_r": round(corr(m["d_model"], m["d_obs"]), 3),
        "anomaly_rmse_mm_per_day": round(float(np.sqrt(((mm_model - mm_obs) ** 2).mean())), 3),
        "n_anomaly_days": int(len(m)),
        "wet_days": int(len(wet)), "wet_event_hit_rate": hit_rate(wet),
        "dry_days": int(len(dry)), "dry_event_hit_rate": hit_rate(dry),
        "n_drying_days": int(len(drying)),
        "model_drying_mm_per_day": model_drying_mm, "obs_drying_mm_per_day": obs_drying_mm,
        "drying_ratio": ratio,
    }


def verdict(layers: dict, min_anomaly_r: float = MIN_ANOMALY_R,
            min_event_hit_rate: float = MIN_EVENT_HIT_RATE) -> dict:
    """Project pass/fail on the only two things this dataset can fairly judge.

    Required, per layer:
      1. anomaly_r >= min_anomaly_r        - do the day-to-day changes follow the same pattern?
      2. wet and dry event hit rates >= min_event_hit_rate - do we wet up and dry down on the
         right days?

    Reported but NOT required: the drying rate in mm/day. It is not a like-for-like
    comparison. Our model dries the whole root zone (Kc x ET0, ~5 mm/day for mid-season
    sugarcane), while ERA5-Land's 0-7 cm layer is a thin, drainage-fed surface layer sampled
    as a daily mean. Comparing the two would judge the difference in what they represent,
    not whether the model is any good.

    Two known handicaps on the wet side, both structural rather than numerical: a depletion
    bucket cannot hold water above field capacity, and it has no percolation. In a monsoon
    the observed soil is wetter than our ceiling on most days, so wet_event_hit_rate is the
    weaker of the two event numbers by construction.
    """
    checks = {}
    for label, m in layers.items():
        if not m.get("enough_data"):
            checks[label] = {"passed": False, "why": f"only {m.get('n_days', 0)} overlapping days"}
            continue
        r = m.get("anomaly_r")
        r_ok = bool(r == r and r >= min_anomaly_r)
        rates = {k: m.get(k) for k in ("wet_event_hit_rate", "dry_event_hit_rate")}
        hit_ok = bool(all(v is not None and v >= min_event_hit_rate for v in rates.values()))
        checks[label] = {
            "passed": bool(r_ok and hit_ok),
            "anomaly_r": r, "needs_anomaly_r": min_anomaly_r, "anomaly_r_ok": r_ok,
            **rates, "needs_event_hit_rate": min_event_hit_rate, "event_rates_ok": hit_ok,
            "informational_drying": {
                "model_mm_per_day": m.get("model_drying_mm_per_day"),
                "observed_mm_per_day": m.get("obs_drying_mm_per_day"),
                "ratio": m.get("drying_ratio"),
                "why_not_required": "root zone vs a thin shallow layer: not like-for-like",
            },
            "informational_level_bias_m3m3": m.get("level_bias_m3m3"),
        }
    return {"passed": bool(checks) and all(c["passed"] for c in checks.values()), "checks": checks}



def plot_validation(model_df: pd.DataFrame, obs_df: pd.DataFrame,
                    path: str | Path = REPORT_PNG) -> str:
    """Model against ERA5-Land soil moisture, with rain underneath."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    merged = model_df.merge(obs_df, on="date", how="inner")
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(merged["date"], merged["theta_m3m3"], "o-", label="model root zone (water balance)")
    for label, col in LAYERS.items():
        if col in merged:
            ax.plot(merged["date"], merged[col], "s--", alpha=0.7, label=f"ERA5-Land {label}")
    ax.set_ylabel("soil water (m3/m3)")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Water balance vs ERA5-Land soil moisture")
    ax2.bar(merged["date"], merged["rain_mm"], color="tab:blue")
    ax2.set_ylabel("rain (mm/day)")
    fig.autofmt_xdate()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)



def validate_field(cfg: dict, days: int = 90, today=None, soil_name: str | None = None,
                   soil_params: dict | None = None, spinup_days: int = SPINUP_DAYS,
                   cache_dir: Path = Path("data"), use_soilgrids: bool = True) -> dict:
    """Whole validation for one field config: fetch, replay, compare, report.

    Writes data/validation.json and data/validation.png and returns the report.
    """
    today = pd.Timestamp(today if today is not None else date.today()).normalize()
    end = today
    start = end - pd.Timedelta(days=days)
    fetch_from = start - pd.Timedelta(days=spinup_days)
    lat, lon = cfg["latitude"], cfg["longitude"]

    if soil_params is None and use_soilgrids:
        try:
            bottom_cm = engine.soil_bottom_cm(engine.load_crop(cfg["crop"]))
            soil_params = soil.field_soil(lat, lon, bottom_cm=bottom_cm, cache_dir=cache_dir)
        except Exception as e:  # noqa: BLE001  (offline: fall back to the texture-class table)
            print("field soil unavailable, using the texture-class table:", e)

    wx = weather.fetch_weather_archive(lat, lon, fetch_from.date(), end.date(), cache_dir=cache_dir)
    obs = weather.fetch_soil_moisture(lat, lon, start.date(), end.date(), cache_dir=cache_dir)
    full = model_series(wx, cfg["crop"], cfg["sowing_date"], soil_name=soil_name,
                        soil_params=soil_params, start=fetch_from, dr0=0.0)
    model = full[full["date"] >= start].reset_index(drop=True)

    layers = {label: compare(model, obs, col, label) for label, col in LAYERS.items()}
    report = {
        "field": {"latitude": lat, "longitude": lon, "crop": cfg["crop"], "sowing_date": cfg["sowing_date"]},
        "window_days": int(days), "spinup_days": int(spinup_days),
        "from": str(start.date()), "to": str(end.date()),
        "model_days": int(len(model)), "observed_days": int(len(obs)),
        "soil": soil_params or {"texture_class": soil_name or engine.DEFAULT_SOIL,
                                "source": "texture-class table (FAO-56 typical values)"},
        "layers": layers, "verdict": verdict(layers),
        "caveat": ("Model-vs-model: ERA5-Land soil moisture is a reanalysis on a ~9 km grid, not a probe in "
                   "this field. The verdict uses only the anomaly correlation and the wet/dry event hit "
                   "rates. The drying rate is reported for information, because a root-zone bucket and a "
                   "thin shallow layer do not represent the same thing. Ground truth would need in-situ "
                   "probes (ISMN) or SMAP, and both need an account."),
    }
    report["plot"] = plot_validation(model, obs, Path(cache_dir) / REPORT_PNG.name)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    (Path(cache_dir) / REPORT_JSON.name).write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    cfg = json.loads(Path("config/field.json").read_text())
    print(json.dumps(validate_field(cfg), indent=2))

