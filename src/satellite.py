"""Step 3A: find Sentinel-2 L2A scenes over the field (Microsoft Planetary Computer STAC).

No pixels are downloaded here. We only list which satellite passes cover the field.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import planetary_computer
from pystac_client import Client
from shapely.geometry import box, mapping, shape

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
HALF_SIDE_DEG = 0.0005  # ~55 m: fallback square when no polygon is drawn
COLUMNS = ["date", "id", "cloud_pct", "tile"]


def load_field_geometry(
    config_path: str | Path = "config/field.json",
    geojson_path: str | Path = "config/field.geojson",
):
    """Use the drawn polygon if config/field.geojson exists, else a small square at the field point."""
    gj = Path(geojson_path)
    if gj.exists():
        data = json.loads(gj.read_text())
        if data.get("type") == "FeatureCollection":
            data = data["features"][0]["geometry"]
        elif data.get("type") == "Feature":
            data = data["geometry"]
        return shape(data)
    cfg = json.loads(Path(config_path).read_text())
    lat, lon = cfg["latitude"], cfg["longitude"]
    return box(lon - HALF_SIDE_DEG, lat - HALF_SIDE_DEG, lon + HALF_SIDE_DEG, lat + HALF_SIDE_DEG)


def find_scenes(geom, days: int = 90, max_cloud_pct: float = 60.0) -> pd.DataFrame:
    """List Sentinel-2 L2A scenes over `geom` in the last `days`.

    cloud_pct is for the WHOLE scene. Cloud over the field itself is checked in step 3C.
    """
    end = date.today()
    start = end - timedelta(days=days)
    catalog = Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=mapping(geom),
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud_pct}},
    )
    rows = [
        {
            "date": item.datetime.date(),
            "id": item.id,
            "cloud_pct": round(item.properties["eo:cloud_cover"], 1),
            "tile": item.properties.get("s2:mgrs_tile"),
        }
        for item in search.items()
    ]
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    geom = load_field_geometry()
    print("Field bounds (lon/lat):", [round(x, 5) for x in geom.bounds])
    scenes = find_scenes(geom)
    print(scenes.to_string(index=False))
    if len(scenes):
        age = (date.today() - scenes["date"].max()).days
        print(f"\n{len(scenes)} scenes; newest is {age} days old")
    else:
        print("No scenes found - widen days or max_cloud_pct")