"""Step 3D: NDVI/NDMI time series for the field, cached on disk, plus a plot.

Each satellite scene is processed once and stored in data/satellite_series.csv,
so later runs only download new scenes (and the app can work offline on cached data).
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.bands import get_item, read_field_bands
from src.indices import field_indices, scale_offset
from src.satellite import find_scenes, load_field_geometry

CACHE = Path("data/satellite_series.csv")
COLUMNS = ["date", "id", "scene_cloud_pct", "clear_fraction", "ndvi", "ndmi", "usable", "top_scl"]
SCL_NAMES = {0: "no data", 1: "saturated", 2: "dark", 3: "cloud shadow", 4: "vegetation",
             5: "bare soil", 6: "water", 7: "unclassified", 8: "cloud", 9: "cloud",
             10: "thin cirrus", 11: "snow"}


def top_scl_name(scl: np.ndarray, inside: np.ndarray) -> str:
    """Most common scene-classification label inside the field (explains why a scene is unusable)."""
    vals = scl[inside]
    if vals.size == 0:
        return "none"
    return SCL_NAMES.get(int(np.bincount(vals).argmax()), "other")


def compute_row(scene, geom) -> dict:
    item = get_item(scene["id"])
    data, inside = read_field_bands(item, geom)
    meta = {b: scale_offset(item, b) for b in ("B04", "B08", "B11")}
    r = field_indices(data, inside, meta)
    return {
        "date": scene["date"], "id": scene["id"], "scene_cloud_pct": scene["cloud_pct"],
        "clear_fraction": r["clear_fraction"], "ndvi": r["ndvi"], "ndmi": r["ndmi"],
        "usable": r["usable"], "top_scl": top_scl_name(data["SCL"], inside),
    }


def build_series(geom, days: int = 90, max_cloud_pct: float = 90.0,
                 cache_path: Path = CACHE, row_fn=None) -> pd.DataFrame:
    """All scenes in the last `days`, processed once and cached. One bad scene never stops the rest."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(cache_path, parse_dates=["date"]) if cache_path.exists() else pd.DataFrame(columns=COLUMNS)
    known = set(old["id"])
    row_fn = row_fn or (lambda s: compute_row(s, geom))

    new_rows = []
    for _, s in find_scenes(geom, days, max_cloud_pct).iterrows():
        if s["id"] in known:
            continue
        try:
            new_rows.append(row_fn(s))
        except Exception as e:  # noqa: BLE001
            print(f"skipped {s['id']}: {e}")

    frames = [f for f in (old, pd.DataFrame(new_rows, columns=COLUMNS)) if len(f)]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates("id").sort_values("date").reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    return df[df["date"] >= pd.Timestamp(date.today() - timedelta(days=days))].reset_index(drop=True)


def satellite_summary(df: pd.DataFrame, today=None) -> dict:
    """What the decision engine (Step 4) needs: latest usable reading and the change since the one before."""
    today = pd.Timestamp(today or date.today())
    ok = df[df["usable"].astype(bool)].sort_values("date")
    if ok.empty:
        return {"available": False}
    last = ok.iloc[-1]
    prev = ok.iloc[-2] if len(ok) > 1 else None
    return {
        "available": True,
        "date": str(last["date"].date()),
        "age_days": int((today - last["date"]).days),
        "ndvi": round(float(last["ndvi"]), 3),
        "ndmi": round(float(last["ndmi"]), 3),
        "ndmi_change": round(float(last["ndmi"] - prev["ndmi"]), 3) if prev is not None else None,
        "days_between": int((last["date"] - prev["date"]).days) if prev is not None else None,
    }


def plot_series(df: pd.DataFrame, rain: pd.DataFrame | None = None, path: str = "data/series.png") -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok, bad = df[df["usable"].astype(bool)], df[~df["usable"].astype(bool)]
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6), gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(ok["date"], ok["ndvi"], "o-", label="NDVI (greenness)")
    ax.plot(ok["date"], ok["ndmi"], "s-", label="NDMI (leaf water)")
    ax.plot(bad["date"], [0] * len(bad), "x", color="grey", label="no clear view of field")
    ax.set_ylabel("index")
    ax.legend()
    if rain is not None:
        ax2.bar(rain["date"], rain["rain_mm"], color="tab:blue")
    ax2.set_ylabel("rain (mm/day)")
    fig.autofmt_xdate()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    from src.weather import fetch_weather

    cfg = json.loads(Path("config/field.json").read_text())
    df = build_series(load_field_geometry())
    cols = ["date", "scene_cloud_pct", "clear_fraction", "ndvi", "ndmi", "usable", "top_scl"]
    print(df[cols].round({"clear_fraction": 2, "ndvi": 2, "ndmi": 2}).to_string(index=False))
    print("\nSummary:", satellite_summary(df))
    # use_cache=False: the weather cache may only hold 30 days, the plot wants 90
    rain = fetch_weather(cfg["latitude"], cfg["longitude"], past_days=90, forecast_days=1, use_cache=False)
    print("Plot saved to", plot_series(df, rain))