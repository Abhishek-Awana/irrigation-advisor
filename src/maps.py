"""Field maps from satellites: where in and around the field the crop is green, wet, thirsty or hot.

The idea follows IrriWatch-style field maps (NASA Spinoff "Farmers Get Tools from Space":
thermal + optical satellites -> evapotranspiration and stress maps), built only from free,
key-less data on Microsoft Planetary Computer:

  Sentinel-2 L2A (10 m)         NDVI greenness, NDMI leaf water, NDMI change between two dates,
                                and crop water use = Kc(NDVI) x ET0 for every pixel
  Landsat 8/9 C2 L2 (thermal)   land surface temperature, as the difference from nearby crops

Every map covers the field plus a margin, so the field can be compared with its neighbours.
Honest limits: this field (~140 x 75 m) is ~100 Sentinel-2 pixels but smaller than one native
100 m Landsat thermal pixel, so the thermal map compares the field with its surroundings; it
cannot show differences inside the field. Maps inform the farmer; src/engine.py decides.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from affine import Affine
from shapely import wkt

from src.bands import BANDS, get_item, read_area
from src.indices import norm_diff, pixel_indices, scale_offset, to_reflectance
from src.satellite import LANDSAT, find_scenes

CACHE_DIR = Path("data/maps")
S2_BUFFER_M = 300  # neighbouring fields within ~300 m, for comparison
LANDSAT_BUFFER_M = 1000  # thermal pixels are ~100 m: look further out
CROP_NDVI_MIN = 0.4  # pixels at least this green count as "nearby crops"
MIN_CLEAR_FIELD_FRACTION = 0.5
MIN_CHANGE_GAP_DAYS = 10  # compare with a scene at least this much older
CHANGE_LIMIT = 0.15  # colour scale of the NDMI change map, +/-
LANDSAT_SEARCH_DAYS = 60
KELVIN_TO_C = 273.15

# Scaled-NDVI crop coefficient (FAO-56 ch. 9 idea: Kc grows with ground cover; see Allen & Pereira
# 2009). Bare soil NDVI ~0.15 -> Kc_min; full canopy NDVI ~0.85 -> the crop's mid-season Kc.
KC_MIN = 0.15
NDVI_SOIL = 0.15
NDVI_FULL = 0.85

# Landsat QA_PIXEL bits that make a pixel unusable: fill, dilated cloud, cirrus, cloud,
# cloud shadow, snow (bits 0-5) and water (bit 7).
LANDSAT_BAD_BITS = sum(1 << b for b in (0, 1, 2, 3, 4, 5, 7))

# Colours (dataviz reference palette). Magnitudes use one hue light -> dark; signed
# differences use blue <-> red with a neutral grey middle.
SURFACE = "#fcfcfb"
TEXT = "#52514e"
NO_DATA = "#d9d8d4"
NEUTRAL = "#f0efec"
RAMPS = {
    "green": [NEUTRAL, "#a9d6a0", "#008300", "#004a00"],
    "blue": [NEUTRAL, "#9cc3ef", "#2a78d6", "#143f78"],
    "orange": [NEUTRAL, "#f6b08f", "#eb6834", "#8a3412"],
    "blue_red": ["#143f78", "#2a78d6", NEUTRAL, "#e34948", "#7f1d1d"],  # low = blue, high = red
    "red_blue": ["#7f1d1d", "#e34948", NEUTRAL, "#2a78d6", "#143f78"],  # low = red, high = blue
}


# ---------- pure calculations ----------

def kc_from_ndvi(ndvi: np.ndarray, kc_max: float, kc_min: float = KC_MIN,
                 ndvi_soil: float = NDVI_SOIL, ndvi_full: float = NDVI_FULL) -> np.ndarray:
    """Crop coefficient per pixel from greenness: bare soil -> kc_min, full canopy -> kc_max."""
    cover = np.clip((ndvi - ndvi_soil) / (ndvi_full - ndvi_soil), 0.0, 1.0)
    return kc_min + cover * (kc_max - kc_min)


def change(new: np.ndarray, old: np.ndarray, ok_new: np.ndarray, ok_old: np.ndarray) -> np.ndarray | None:
    """new - old where both dates are clear; None when the two grids do not line up."""
    if new.shape != old.shape:
        return None
    return np.where(ok_new & ok_old, new - old, np.nan)


def lst_clear(qa: np.ndarray) -> np.ndarray:
    """Landsat QA_PIXEL -> True where the pixel is clear land."""
    return (qa.astype("uint32") & LANDSAT_BAD_BITS) == 0


def neighbour_compare(values: np.ndarray, valid: np.ndarray, inside: np.ndarray,
                      ndvi: np.ndarray | None = None, crop_ndvi_min: float = CROP_NDVI_MIN) -> dict:
    """Field mean vs the median of nearby crop pixels (outside the field, green enough)."""
    ok = valid & np.isfinite(values)
    field = ok & inside
    around = ok & ~inside
    if ndvi is not None:
        around &= np.nan_to_num(ndvi, nan=-1.0) >= crop_ndvi_min
    f = float(np.mean(values[field])) if field.any() else None
    a = float(np.median(values[around])) if around.any() else None
    return {
        "field": None if f is None else round(f, 3),
        "around": None if a is None else round(a, 3),
        "diff": None if f is None or a is None else round(f - a, 3),
        "n_field": int(field.sum()), "n_around": int(around.sum()),
        "clear_field_fraction": round(float(field.sum() / inside.sum()), 2) if inside.any() else 0.0,
    }


def pick_scenes(series: pd.DataFrame, min_gap_days: int = MIN_CHANGE_GAP_DAYS):
    """(latest usable scene, an earlier usable one at least `min_gap_days` older or None)."""
    ok = series[series["usable"].astype(bool)].copy()
    if ok.empty:
        return None, None
    ok["date"] = pd.to_datetime(ok["date"])
    ok = ok.sort_values("date")
    latest = ok.iloc[-1]
    older = ok[ok["date"] <= latest["date"] - pd.Timedelta(days=min_gap_days)]
    return latest, (older.iloc[-1] if len(older) else None)


# ---------- grid cache (so maps work offline once fetched) ----------

def _save_grid(path: Path, grid: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in grid.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    meta = {"transform": list(grid["transform"])[:6], "crs": str(grid["crs"]), "field": grid["field"].wkt,
            **{k: v for k, v in grid.items() if k not in arrays and k not in ("transform", "crs", "field")}}
    path.with_suffix(".json").write_text(json.dumps(meta))


def _load_grid(path: Path) -> dict | None:
    npz, meta_path = path.with_suffix(".npz"), path.with_suffix(".json")
    if not (npz.exists() and meta_path.exists()):
        return None
    meta = json.loads(meta_path.read_text())
    with np.load(npz) as arrays:
        grid = {k: arrays[k] for k in arrays.files}
    grid.update(meta)
    grid["transform"] = Affine(*meta["transform"])
    grid["field"] = wkt.loads(meta["field"])
    return grid


# ---------- satellite reads ----------

def s2_grid(scene_id: str, geom, buffer_m: float = S2_BUFFER_M, cache_dir: Path = CACHE_DIR) -> dict:
    """Per-pixel NDVI/NDMI around the field for one Sentinel-2 scene (cached on disk)."""
    path = Path(cache_dir) / f"s2_{scene_id}_{int(buffer_m)}"
    cached = _load_grid(path)
    if cached is not None:
        return cached
    item = get_item(scene_id)
    area = read_area(item, geom, BANDS, buffer_m)
    px = pixel_indices(area["data"], {b: scale_offset(item, b) for b in ("B04", "B08", "B11")})
    grid = {**px, "inside": area["inside"], "transform": area["transform"], "crs": area["crs"],
            "field": area["field"], "scene_id": scene_id, "date": str(item.datetime.date())}
    _save_grid(path, grid)
    return grid


def landsat_grid(scene_id: str, geom, buffer_m: float = LANDSAT_BUFFER_M, cache_dir: Path = CACHE_DIR) -> dict:
    """Land surface temperature (deg C) and NDVI around the field for one Landsat scene (cached)."""
    path = Path(cache_dir) / f"landsat_{scene_id}_{int(buffer_m)}"
    cached = _load_grid(path)
    if cached is not None:
        return cached
    item = get_item(scene_id, LANDSAT)
    area = read_area(item, geom, ("lwir11", "red", "nir08", "qa_pixel"), buffer_m)
    d = area["data"]
    lst_c = to_reflectance(d["lwir11"], *scale_offset(item, "lwir11")[:2]) - KELVIN_TO_C
    red = to_reflectance(d["red"], *scale_offset(item, "red")[:2])
    nir = to_reflectance(d["nir08"], *scale_offset(item, "nir08")[:2])
    grid = {"lst_c": lst_c, "ndvi": norm_diff(nir, red), "clear": lst_clear(d["qa_pixel"]) & np.isfinite(lst_c),
            "inside": area["inside"], "transform": area["transform"], "crs": area["crs"],
            "field": area["field"], "scene_id": scene_id, "date": str(item.datetime.date())}
    _save_grid(path, grid)
    return grid


def latest_clear_landsat(geom, days: int = LANDSAT_SEARCH_DAYS, cache_dir: Path = CACHE_DIR,
                         max_tries: int = 4) -> dict | None:
    """Newest Landsat scene in which the field itself is clear of cloud."""
    scenes = find_scenes(geom, days, 80.0, collection=LANDSAT)
    for _, s in scenes.sort_values("date", ascending=False).head(max_tries).iterrows():
        grid = landsat_grid(s["id"], geom, cache_dir=cache_dir)
        inside = grid["inside"]
        if inside.any() and (grid["clear"] & inside).sum() / inside.sum() >= MIN_CLEAR_FIELD_FRACTION:
            return grid
    return None


# ---------- drawing ----------

def colormap(name: str):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, RAMPS[name]).with_extremes(bad=NO_DATA)


def render_map(values: np.ndarray, valid: np.ndarray, grid: dict, title: str, ramp: str,
               vmin: float, vmax: float, unit: str, path: Path, note: str | None = None) -> str:
    """One map: the values around the field, the field outline, a colour bar, a plain caption."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, field = grid["transform"], grid["field"]
    h, w = values.shape
    cx, cy = field.centroid.x, field.centroid.y
    extent = (t.c - cx, t.c + t.a * w - cx, t.f + t.e * h - cy, t.f - cy)  # metres from field centre
    shown = np.ma.masked_where(~(valid & np.isfinite(values)), values)

    fig, ax = plt.subplots(figsize=(6.4, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(shown, cmap=colormap(ramp), vmin=vmin, vmax=vmax, extent=extent, interpolation="nearest")
    for poly in getattr(field, "geoms", [field]):
        xs, ys = poly.exterior.xy
        ax.plot(np.asarray(xs) - cx, np.asarray(ys) - cy, color="#0b0b0b", linewidth=2)
    ax.set_title(title, color="#0b0b0b", fontsize=11, loc="left")
    ax.set_xlabel("metres east of field centre", color=TEXT, fontsize=8)
    ax.set_ylabel("metres north of field centre", color=TEXT, fontsize=8)
    ax.tick_params(colors=TEXT, labelsize=7)
    for side in ax.spines.values():
        side.set_color(NO_DATA)
    bar = fig.colorbar(im, ax=ax, shrink=0.8)
    bar.set_label(unit, color=TEXT, fontsize=8)
    bar.ax.tick_params(colors=TEXT, labelsize=7)
    bar.outline.set_visible(False)
    caption = "Black outline = your field. Grey = cloud, shadow, water or no data."
    fig.text(0.01, 0.01, caption + (f"\n{note}" if note else ""), color=TEXT, fontsize=7, va="bottom")
    fig.tight_layout(rect=(0, 0.06 if note else 0.04, 1, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return str(path)


def overlay_png(values: np.ndarray, valid: np.ndarray, ramp: str, vmin: float, vmax: float) -> bytes:
    """Bare coloured raster (transparent where invalid) to lay over a web map."""
    from io import BytesIO

    import matplotlib.pyplot as plt

    norm = np.clip((np.nan_to_num(values) - vmin) / (vmax - vmin), 0, 1)
    rgba = colormap(ramp)(norm)
    rgba[..., 3] = np.where(valid & np.isfinite(values), 0.8, 0.0)
    buf = BytesIO()
    plt.imsave(buf, rgba, format="png")
    return buf.getvalue()


def bounds_lonlat(grid: dict, shape: tuple[int, int]) -> list[float]:
    """[west, south, east, north] of the grid in degrees, for the web-map overlay."""
    from rasterio.warp import transform_bounds

    t = grid["transform"]
    h, w = shape
    return list(transform_bounds(grid["crs"], "EPSG:4326", t.c, t.f + t.e * h, t.c + t.a * w, t.f))


# ---------- the full set ----------

def _fmt(v, digits=2) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def s2_maps(latest: dict, older: dict | None, et0_mm: float | None, kc_max: float, out_dir: Path) -> dict:
    """NDVI, NDMI, water use and (with an older scene) NDMI change, each with its numbers."""
    out = {}
    inside, ok, day = latest["inside"], latest["clear"], latest["date"]

    def add(key, values, valid, title, ramp, vmin, vmax, unit, cmp, caption, note=None):
        out[key] = {"path": render_map(values, valid, latest, title, ramp, vmin, vmax, unit, out_dir / f"{key}.png", note),
                    "date": day, "stats": cmp, "caption": caption,
                    "overlay": {"values": values, "valid": valid, "ramp": ramp, "vmin": vmin, "vmax": vmax}}

    c = neighbour_compare(latest["ndvi"], ok, inside, latest["ndvi"])
    add("ndvi", latest["ndvi"], ok, f"Greenness (NDVI), {day}", "green", 0.0, 0.9, "NDVI", c,
        f"Greenness on {day}: your field {_fmt(c['field'])}, nearby crops {_fmt(c['around'])}.")

    c = neighbour_compare(latest["ndmi"], ok, inside, latest["ndvi"])
    add("ndmi", latest["ndmi"], ok, f"Leaf water (NDMI), {day}", "blue", -0.1, 0.5, "NDMI", c,
        f"Leaf water on {day}: your field {_fmt(c['field'])}, nearby crops {_fmt(c['around'])}. "
        "Higher = more water in the leaves.")

    if et0_mm is not None:
        etc = kc_from_ndvi(latest["ndvi"], kc_max) * et0_mm
        c = neighbour_compare(etc, ok, inside, latest["ndvi"])
        add("water_use", etc, ok, f"Crop water use today (from {day} greenness)", "orange",
            0.0, max(kc_max * et0_mm, 0.1), "mm per day", c,
            f"Estimated water use today: your field {_fmt(c['field'], 1)} mm/day, nearby crops "
            f"{_fmt(c['around'], 1)} mm/day (Kc from NDVI x today's ET0 {et0_mm:.1f} mm).",
            note="Kc from NDVI is a scaled estimate, not a measurement.")

    if older is not None:
        diff = change(latest["ndmi"], older["ndmi"], ok, older["clear"])
        if diff is not None:
            valid = np.isfinite(diff)
            c = neighbour_compare(diff, valid, inside, latest["ndvi"])
            add("ndmi_change", diff, valid, f"Leaf water change, {older['date']} to {day}", "red_blue",
                -CHANGE_LIMIT, CHANGE_LIMIT, "NDMI change (blue wetter, red drier)", c,
                f"Leaf water change {older['date']} to {day}: your field {_fmt(c['field'], 3)}, "
                f"nearby crops {_fmt(c['around'], 3)}. A field drying faster than its neighbours needs a look.")
    return out


def thermal_map(grid: dict, out_dir: Path) -> dict:
    """Surface temperature minus the median of nearby clear crop pixels."""
    inside, ok = grid["inside"], grid["clear"]
    base = neighbour_compare(grid["lst_c"], ok, inside, grid["ndvi"])
    if base["around"] is None:
        raise ValueError("no clear crop pixels around the field in this Landsat scene")
    anomaly = grid["lst_c"] - base["around"]
    finite = np.abs(anomaly[ok & np.isfinite(anomaly)])
    lim = float(max(3.0, np.percentile(finite, 95))) if finite.size else 3.0
    diff, day = base["diff"], grid["date"]
    word = "warmer" if diff is not None and diff > 0 else "cooler"
    caption = (f"Surface temperature on {day}: your field {_fmt(base['field'], 1)} deg C, nearby crops "
               f"{_fmt(base['around'], 1)} deg C ({abs(diff or 0):.1f} deg C {word}). "
               "A crop that is warmer than its neighbours is often short of water.")
    note = "Landsat thermal is ~100 m per pixel: this compares your field with its neighbours."
    return {"path": render_map(anomaly, ok, grid, f"Surface temperature vs nearby crops, {day}", "blue_red",
                               -lim, lim, "deg C vs nearby crops (red warmer)", out_dir / "thermal.png", note),
            "date": day, "stats": {**base, "lst_field_c": base["field"]}, "caption": caption,
            "overlay": {"values": anomaly, "valid": ok, "ramp": "blue_red", "vmin": -lim, "vmax": lim}}


def build_maps(geom, series: pd.DataFrame, et0_mm: float | None, kc_max: float,
               out_dir: Path = CACHE_DIR, cache_dir: Path = CACHE_DIR, thermal: bool = True) -> dict:
    """All maps for the app. Returns {"maps": {name: {...}}, "grids": {...}, "errors": [...]}.

    Each map is independent: a cloudy Landsat pass never stops the Sentinel-2 maps.
    """
    out_dir, maps, grids, errors = Path(out_dir), {}, {}, []
    latest_row, older_row = pick_scenes(series)
    if latest_row is None:
        errors.append("no clear Sentinel-2 view of the field in the time series")
    else:
        try:
            latest = s2_grid(latest_row["id"], geom, cache_dir=cache_dir)
            older = s2_grid(older_row["id"], geom, cache_dir=cache_dir) if older_row is not None else None
            grids["s2"] = latest
            maps.update(s2_maps(latest, older, et0_mm, kc_max, out_dir))
        except Exception as e:  # noqa: BLE001
            errors.append(f"Sentinel-2 maps failed: {e}")
    if thermal:
        try:
            grid = latest_clear_landsat(geom, cache_dir=cache_dir)
            if grid is None:
                errors.append(f"no cloud-free Landsat view of the field in the last {LANDSAT_SEARCH_DAYS} days")
            else:
                grids["landsat"] = grid
                maps["thermal"] = thermal_map(grid, out_dir)
        except Exception as e:  # noqa: BLE001
            errors.append(f"thermal map failed: {e}")
    return {"maps": maps, "grids": grids, "errors": errors}


if __name__ == "__main__":
    from src.engine import load_crop
    from src.satellite import load_field_geometry
    from src.timeseries import build_series
    from src.weather import fetch_weather

    cfg = json.loads(Path("config/field.json").read_text())
    geom = load_field_geometry()
    wx = fetch_weather(cfg["latitude"], cfg["longitude"])
    today = wx[wx["date"] == pd.Timestamp.today().normalize()]
    et0 = float(today["et0_mm"].iloc[0]) if len(today) else None
    result = build_maps(geom, build_series(geom), et0, load_crop(cfg["crop"])["kc"]["mid"])
    for name, m in result["maps"].items():
        print(f"{name:12} {m['path']}\n             {m['caption']}")
    for e in result["errors"]:
        print("!", e)
