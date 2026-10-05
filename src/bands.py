"""Step 3B: read Sentinel-2 band pixels inside the field.

B04 (red) and B08 (near-infrared) are 10 m; B11 (shortwave-infrared) and SCL
(scene classification) are 20 m. Every band is resampled onto one common 10 m
grid around the field so pixels line up. Values are raw digital numbers (DN);
conversion to reflectance and the indices come in step 3C.
"""
from __future__ import annotations

import numpy as np
import planetary_computer
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_geom
from rasterio.windows import from_bounds
from shapely.geometry import mapping, shape

from src.satellite import S2, STAC_URL, find_scenes, load_field_geometry

BANDS = ("B04", "B08", "B11", "SCL")
BUFFER_M = 10  # read a little around the field so tiny polygons still cover >=1 pixel
SCL_NAMES = {
    0: "no data", 1: "saturated", 2: "dark area", 3: "cloud shadow", 4: "vegetation",
    5: "bare soil", 6: "water", 7: "unclassified", 8: "cloud (medium)",
    9: "cloud (high)", 10: "thin cirrus", 11: "snow",
}
GDAL_OPTS = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif"}


def get_item(scene_id: str, collection: str = S2):
    """Fetch one signed STAC item by id."""
    catalog = Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(collections=[collection], ids=[scene_id])
    return next(search.items())


def read_field_bands(item, geom, bands=BANDS):
    """Return ({band: 2D array}, inside_mask) on a common grid.

    inside_mask is True for pixels touched by the field polygon.
    """
    grid = read_area(item, geom, bands, BUFFER_M)
    return grid["data"], grid["inside"]


def read_area(item, geom, bands=BANDS, buffer_m: float = BUFFER_M) -> dict:
    """Bands on one grid covering the field plus `buffer_m` around it (for maps of the neighbourhood).

    Returns {"data": {band: 2D array}, "inside": mask, "transform", "crs", "field": field polygon in crs}.
    Works for any STAC item whose assets are rasters (Sentinel-2, Landsat).
    """
    with rasterio.Env(**GDAL_OPTS):
        with rasterio.open(item.assets[bands[0]].href) as src:
            crs = src.crs
            g = shape(transform_geom("EPSG:4326", crs, mapping(geom)))
            win = from_bounds(*g.buffer(buffer_m).bounds, transform=src.transform)
            win = win.round_offsets().round_lengths()
            transform = src.window_transform(win)
            width, height = int(win.width), int(win.height)

        data = {}
        for name in bands:
            with rasterio.open(item.assets[name].href) as src:
                with WarpedVRT(
                    src, crs=crs, transform=transform, width=width, height=height,
                    resampling=Resampling.nearest,
                ) as vrt:
                    data[name] = vrt.read(1)

    inside = geometry_mask(
        [mapping(g)], out_shape=(height, width), transform=transform, invert=True, all_touched=True
    )
    return {"data": data, "inside": inside, "transform": transform, "crs": crs, "field": g}


if __name__ == "__main__":
    geom = load_field_geometry()
    scenes = find_scenes(geom)
    best = scenes[scenes["cloud_pct"] < 20].sort_values("date").iloc[-1]
    print("Using scene:", best["date"], best["id"], f"(scene cloud {best['cloud_pct']}%)")
    data, inside = read_field_bands(get_item(best["id"]), geom)
    print("Grid shape:", inside.shape, "| field pixels:", int(inside.sum()))
    for name in ("B04", "B08", "B11"):
        print(f"{name}: mean DN in field = {data[name][inside].mean():.0f}")
    classes, counts = np.unique(data["SCL"][inside], return_counts=True)
    for c, n in zip(classes, counts):
        print(f"SCL {c} ({SCL_NAMES.get(int(c), '?')}): {n} px")