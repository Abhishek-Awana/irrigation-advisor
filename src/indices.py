"""Step 3C: raw band values -> reflectance -> clear-pixel mask -> NDVI and NDMI for the field.

NDVI = (NIR - Red) / (NIR + Red)   greenness / canopy density
NDMI = (NIR - SWIR) / (NIR + SWIR) water content of the leaves (uses B08 and B11)
"""
from __future__ import annotations

import numpy as np

from src.bands import get_item, read_field_bands
from src.satellite import find_scenes, load_field_geometry

GOOD_SCL = (4, 5)  # vegetation, bare soil. Cloud, shadow, cirrus, water, no-data are dropped.
MIN_CLEAR_FRACTION = 0.5  # below this share of clear field pixels the scene is not usable
DEFAULT_SCALE_OFFSET = (1e-4, -0.1)  # standard for scenes processed after Jan 2022


def scale_offset(item, band: str) -> tuple[float, float, str]:
    """(scale, offset, source). Prefer the scene's own metadata, else the standard default."""
    rb = item.assets[band].extra_fields.get("raster:bands")
    if rb and "scale" in rb[0] and "offset" in rb[0]:
        return rb[0]["scale"], rb[0]["offset"], "metadata"
    return (*DEFAULT_SCALE_OFFSET, "default")


def to_reflectance(dn: np.ndarray, scale: float, offset: float) -> np.ndarray:
    """Digital numbers -> reflectance (0 to ~1). DN 0 means no data and becomes NaN."""
    refl = dn.astype("float32") * scale + offset
    refl[dn == 0] = np.nan
    return refl


def norm_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    s = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s > 0, (a - b) / s, np.nan)


def pixel_indices(data: dict, meta: dict) -> dict:
    """Per-pixel NDVI and NDMI, and which pixels are a clear view of land (not cloud/shadow/water)."""
    red = to_reflectance(data["B04"], *meta["B04"][:2])
    nir = to_reflectance(data["B08"], *meta["B08"][:2])
    swir = to_reflectance(data["B11"], *meta["B11"][:2])
    ndvi, ndmi = norm_diff(nir, red), norm_diff(nir, swir)
    clear = np.isin(data["SCL"], GOOD_SCL) & np.isfinite(ndvi) & np.isfinite(ndmi)
    return {"ndvi": ndvi, "ndmi": ndmi, "clear": clear}


def field_indices(data: dict, inside: np.ndarray, meta: dict) -> dict:
    """Mean NDVI/NDMI over clear pixels inside the field.

    data: {"B04","B08","B11","SCL"} arrays from bands.read_field_bands
    meta: {"B04": (scale, offset, ...), ...}
    """
    px = pixel_indices(data, meta)
    ndvi, ndmi = px["ndvi"], px["ndmi"]
    clear = inside & px["clear"]
    n_field, n_clear = int(inside.sum()), int(clear.sum())
    frac = n_clear / n_field if n_field else 0.0
    return {
        "ndvi": float(ndvi[clear].mean()) if n_clear else None,
        "ndmi": float(ndmi[clear].mean()) if n_clear else None,
        "clear_pixels": n_clear,
        "field_pixels": n_field,
        "clear_fraction": round(frac, 2),
        "usable": frac >= MIN_CLEAR_FRACTION,
    }


if __name__ == "__main__":
    geom = load_field_geometry()
    scenes = find_scenes(geom).tail(5)
    print(f"{'date':10} {'scene_cloud%':>12} {'field_clear%':>12} {'NDVI':>6} {'NDMI':>6}  usable")
    meta = {}
    for _, s in scenes.iterrows():
        item = get_item(s["id"])
        data, inside = read_field_bands(item, geom)
        meta = {b: scale_offset(item, b) for b in ("B04", "B08", "B11")}
        r = field_indices(data, inside, meta)
        f = lambda v: f"{v:6.2f}" if v is not None else "   n/a"  # noqa: E731
        print(f"{s['date']!s:10} {s['cloud_pct']:12.1f} {r['clear_fraction'] * 100:12.0f} "
              f"{f(r['ndvi'])} {f(r['ndmi'])}  {r['usable']}")
    print("\nscale/offset (scale, offset, source):", meta)