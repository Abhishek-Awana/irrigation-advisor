"""Step 5: the field's own soil - texture and how much water it can hold.

Three open, key-less pieces:
  ISRIC SoilGrids REST API  -> clay / sand / silt / organic carbon / bulk density at the field point
  USDA texture triangle     -> texture class name (Soil Survey Manual 2017), for humans
  Saxton & Rawls (2006)     -> theta_fc, theta_wp, TAW from the actual texture

theta_fc_m3m3  water held at field capacity (-33 kPa)
theta_wp_m3m3  water held at permanent wilting point (-1500 kPa)
taw_mm_per_m   (theta_fc - theta_wp) * 1000: plant-available water per metre of soil
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
PROPERTIES = ["clay", "sand", "silt", "soc", "bdod"]
DEPTHS = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]
THICKNESS_CM = {"0-5cm": 5, "5-15cm": 10, "15-30cm": 15, "30-60cm": 30, "60-100cm": 40, "100-200cm": 100}
SOC_TO_OM = 1.724  # van Bemmelen factor: organic matter = organic carbon x 1.724
CACHE_DIR = Path("data")
CACHE_MAX_AGE_S = 90 * 24 * 3600  # soil barely changes; refetch a cached profile after 90 days
TIMEOUT_S = 30
SOURCE = "ISRIC SoilGrids 250 m + USDA texture triangle + Saxton & Rawls (2006)"


def _range_cm(label: str) -> tuple[int, int]:
    """'15-30cm' -> (15, 30)."""
    top, bottom = label.replace("cm", "").split("-")
    return int(top), int(bottom)


def fetch_layers(lat: float, lon: float, properties=PROPERTIES, depths=DEPTHS,
                 timeout_s: int = TIMEOUT_S) -> dict:
    """SoilGrids -> {property: {depth_label: value_in_real_units}}.

    The API returns integers scaled by d_factor, so 258 g/kg of clay becomes 25.8 %.
    """
    params = {"lon": lon, "lat": lat, "property": list(properties), "depth": list(depths), "value": "mean"}
    r = requests.get(URL, params=params, timeout=timeout_s)
    r.raise_for_status()
    out: dict = {}
    for layer in r.json()["properties"]["layers"]:
        factor = layer["unit_measure"]["d_factor"]
        out[layer["name"]] = {d["label"]: d["values"]["mean"] / factor for d in layer["depths"]}
    return out


def depth_mean(per_depth: dict, top_cm: int, bottom_cm: int) -> float:
    """Thickness-weighted mean between top_cm and bottom_cm, e.g. 0-30 cm."""
    total_w, acc = 0.0, 0.0
    for label, value in per_depth.items():
        lo, hi = _range_cm(label)
        overlap = min(hi, bottom_cm) - max(lo, top_cm)
        if overlap > 0:
            acc += value * overlap
            total_w += overlap
    return acc / total_w if total_w else float("nan")


def rescale_to_100(sand_pct: float, silt_pct: float, clay_pct: float) -> tuple[float, float, float]:
    """SoilGrids medians rarely add up to exactly 100, and the USDA rules assume they do."""
    total = sand_pct + silt_pct + clay_pct
    if total <= 0:
        return sand_pct, silt_pct, clay_pct
    return tuple(v * 100.0 / total for v in (sand_pct, silt_pct, clay_pct))  # type: ignore[return-value]



def texture_class(sand_pct: float, silt_pct: float, clay_pct: float) -> str:
    """USDA texture class. Rules and their order follow the Soil Survey Manual (2017), ch. 3."""
    sand, silt, clay = rescale_to_100(sand_pct, silt_pct, clay_pct)
    if sand > 85 and silt + 1.5 * clay < 15:
        return "sand"
    if 70 <= sand <= 90 and silt + 1.5 * clay >= 15 and silt + 2 * clay < 30:
        return "loamy_sand"
    if (7 <= clay < 20 and sand > 52 and silt + 2 * clay >= 30) or (clay < 7 and silt < 50 and silt + 2 * clay >= 30):
        return "sandy_loam"
    if 7 <= clay < 27 and 28 <= silt < 50 and sand <= 52:
        return "loam"
    if (silt >= 50 and 12 <= clay < 27) or (50 <= silt < 80 and clay < 12):
        return "silt_loam"
    if clay < 12 and silt >= 80:
        return "silt"
    if 20 <= clay < 35 and sand > 45 and silt < 28:
        return "sandy_clay_loam"
    if 27 <= clay < 40 and 20 < sand <= 45:
        return "clay_loam"
    if 27 <= clay < 40 and sand <= 20:
        return "silty_clay_loam"
    if clay >= 35 and sand > 45:
        return "sandy_clay"
    if clay >= 40 and silt >= 40:
        return "silty_clay"
    if clay >= 40 and sand <= 45 and silt < 40:
        return "clay"
    return "unknown"


def saxton_rawls_2006(sand_pct: float, clay_pct: float, om_pct: float = 0.0) -> dict:
    """Field capacity and wilting point in m3/m3 (Saxton & Rawls 2006, SSSAJ 70:1569-1578)."""
    s, c, om = sand_pct / 100.0, clay_pct / 100.0, om_pct
    wp_t = -0.024 * s + 0.487 * c + 0.006 * om + 0.005 * s * om - 0.013 * c * om + 0.068 * s * c + 0.031
    fc_t = -0.251 * s + 0.195 * c + 0.011 * om + 0.006 * s * om - 0.027 * c * om + 0.452 * s * c + 0.299
    return {
        "theta_wp_m3m3": round(wp_t + 0.14 * wp_t - 0.02, 4),
        "theta_fc_m3m3": round(fc_t + 1.283 * fc_t ** 2 - 0.374 * fc_t - 0.015, 4),
    }


def field_soil(lat: float, lon: float, top_cm: int = 0, bottom_cm: int = 30,
               use_cache: bool = True, cache_dir: Path = CACHE_DIR,
               timeout_s: int = TIMEOUT_S) -> dict:
    """Texture and water-holding capacity of the soil at this field point.

    Defaults to 0-30 cm; callers should pass the crop's full root depth
    (engine.soil_bottom_cm) so TAW describes the whole root zone. Cached to
    data/soil_{lat}_{lon}_{top}_{bottom}.json so the app keeps working offline.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"soil_{lat:.4f}_{lon:.4f}_{top_cm}_{bottom_cm}.json"

    if use_cache and cache.exists() and time.time() - cache.stat().st_mtime < CACHE_MAX_AGE_S:
        return json.loads(cache.read_text())

    layers = fetch_layers(lat, lon, timeout_s=timeout_s)
    mean = {p: depth_mean(v, top_cm, bottom_cm) for p, v in layers.items()}
    sand_pct, silt_pct, clay_pct = rescale_to_100(mean["sand"], mean["silt"], mean["clay"])
    om_pct = mean["soc"] / 10.0 * SOC_TO_OM  # SoilGrids soc is g/kg -> % -> organic matter
    sr = saxton_rawls_2006(sand_pct, clay_pct, om_pct)
    out = {
        "latitude": lat, "longitude": lon, "top_cm": top_cm, "bottom_cm": bottom_cm,
        "sand_pct": round(sand_pct, 1), "silt_pct": round(silt_pct, 1), "clay_pct": round(clay_pct, 1),
        "organic_matter_pct": round(om_pct, 2),
        "bulk_density_kg_dm3": round(mean.get("bdod", float("nan")), 2),
        "texture_class": texture_class(sand_pct, silt_pct, clay_pct),
        **sr,
        "taw_mm_per_m": round((sr["theta_fc_m3m3"] - sr["theta_wp_m3m3"]) * 1000, 1),
        "source": SOURCE,
    }
    cache.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    cfg = json.loads(Path("config/field.json").read_text())
    from src.engine import load_crop, soil_bottom_cm

    soil = field_soil(cfg["latitude"], cfg["longitude"], bottom_cm=soil_bottom_cm(load_crop(cfg["crop"])))
    print(json.dumps(soil, indent=2))
    print(f"\nThis field: {soil['texture_class']}, "
          f"{soil['clay_pct']:.0f}% clay, TAW {soil['taw_mm_per_m']:.0f} mm per metre of soil")

