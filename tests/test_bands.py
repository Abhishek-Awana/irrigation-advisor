import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_xy
from shapely.geometry import box

from src import bands

LON, LAT = 77.772, 29.373


class Asset:
    def __init__(self, href):
        self.href = href


class Item:
    def __init__(self, assets):
        self.assets = assets


def make_tif(path, res, value, x0, y0, size_m=1000):
    n = size_m // res
    with rasterio.open(
        path, "w", driver="GTiff", height=n, width=n, count=1, dtype="uint16",
        crs="EPSG:32643", transform=from_origin(x0, y0, res, res),
    ) as dst:
        dst.write(np.full((n, n), value, dtype="uint16"), 1)


def test_bands_align_on_common_grid(tmp_path):
    xs, ys = warp_xy("EPSG:4326", "EPSG:32643", [LON], [LAT])
    x0, y0 = (int(xs[0]) // 20) * 20 - 500, (int(ys[0]) // 20) * 20 + 500
    spec = {"B04": (10, 1000), "B08": (10, 4000), "B11": (20, 3000), "SCL": (20, 4)}
    assets = {}
    for name, (res, val) in spec.items():
        p = tmp_path / f"{name}.tif"
        make_tif(p, res, val, x0, y0)
        assets[name] = Asset(str(p))

    geom = box(LON - 0.0003, LAT - 0.0003, LON + 0.0003, LAT + 0.0003)  # ~60 m square
    data, inside = bands.read_field_bands(Item(assets), geom)

    shapes = {a.shape for a in data.values()}
    assert shapes == {inside.shape}  # all bands share one grid
    assert 10 < inside.sum() < 100
    assert (data["B04"][inside] == 1000).all()
    assert (data["B11"][inside] == 3000).all()  # 20 m band resampled to 10 m
    assert (data["SCL"][inside] == 4).all()