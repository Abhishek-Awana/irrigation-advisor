import numpy as np
import pandas as pd
import pytest
from affine import Affine
from shapely.geometry import box

from src import maps


def grid(n=10):
    """A 10 x 10 grid of 10 m pixels; the field is the middle 4 x 4."""
    inside = np.zeros((n, n), bool)
    inside[3:7, 3:7] = True
    return {"inside": inside, "transform": Affine(10, 0, 500_000, 0, -10, 3_250_100),
            "crs": "EPSG:32643", "field": box(500_030, 3_250_030, 500_070, 3_250_070)}


def test_kc_from_ndvi_spans_soil_to_full_canopy():
    kc = maps.kc_from_ndvi(np.array([0.0, 0.15, 0.5, 0.85, 0.95]), kc_max=1.25)
    assert kc[0] == pytest.approx(maps.KC_MIN) and kc[1] == pytest.approx(maps.KC_MIN)
    assert kc[3] == pytest.approx(1.25) and kc[4] == pytest.approx(1.25)
    assert maps.KC_MIN < kc[2] < 1.25


def test_change_needs_both_dates_clear_and_same_grid():
    new, old = np.array([[0.3, 0.3]]), np.array([[0.2, 0.2]])
    d = maps.change(new, old, np.array([[True, True]]), np.array([[True, False]]))
    assert d[0, 0] == pytest.approx(0.1) and np.isnan(d[0, 1])
    assert maps.change(new, np.zeros((2, 2)), new > 0, new > 0) is None


def test_lst_clear_drops_cloud_shadow_and_water():
    qa = np.array([21824, 21824 | (1 << 3), 21824 | (1 << 4), 21824 | (1 << 7), 1])
    # 21824 is Landsat's "clear land" value; then cloud, shadow, water, fill
    assert maps.lst_clear(qa).tolist() == [True, False, False, False, False]


def test_neighbour_compare_uses_only_green_pixels_outside():
    g = grid()
    values = np.full((10, 10), 2.0)
    values[g["inside"]] = 5.0
    ndvi = np.full((10, 10), 0.7)
    ndvi[0, :] = 0.1  # a road: not a crop, must be ignored
    values[0, :] = 100.0
    c = maps.neighbour_compare(values, np.ones((10, 10), bool), g["inside"], ndvi)
    assert c["field"] == 5.0 and c["around"] == 2.0 and c["diff"] == 3.0
    assert c["clear_field_fraction"] == 1.0


def test_pick_scenes_takes_latest_and_one_old_enough():
    s = pd.DataFrame({"date": ["2026-09-01", "2026-09-20", "2026-09-30", "2026-10-02"],
                      "id": list("abcd"), "usable": [True, True, False, True]})
    latest, older = maps.pick_scenes(s, min_gap_days=10)
    assert latest["id"] == "d" and older["id"] == "b"
    assert maps.pick_scenes(s.assign(usable=False)) == (None, None)


def test_grid_cache_round_trip(tmp_path):
    g = {**grid(), "ndvi": np.full((10, 10), 0.6, "float32"), "date": "2026-10-02"}
    maps._save_grid(tmp_path / "x", g)
    back = maps._load_grid(tmp_path / "x")
    assert back["transform"] == g["transform"] and back["field"].equals(g["field"])
    assert back["date"] == "2026-10-02" and np.allclose(back["ndvi"], 0.6)
    assert maps._load_grid(tmp_path / "missing") is None


def test_s2_maps_draws_every_map_and_states_the_numbers(tmp_path):
    g = grid()
    ndvi = np.full((10, 10), 0.6)
    latest = {**g, "ndvi": ndvi, "ndmi": np.full((10, 10), 0.2), "clear": np.ones((10, 10), bool),
              "date": "2026-10-02"}
    older = {**latest, "ndmi": np.full((10, 10), 0.3), "date": "2026-09-20"}
    out = maps.s2_maps(latest, older, et0_mm=4.0, kc_max=1.25, out_dir=tmp_path)
    assert set(out) == {"ndvi", "ndmi", "water_use", "ndmi_change"}
    assert all((tmp_path / f"{k}.png").exists() for k in out)
    assert out["ndmi_change"]["stats"]["field"] == pytest.approx(-0.1)
    assert "0.60" in out["ndvi"]["caption"]


def test_thermal_map_reports_field_vs_neighbours(tmp_path):
    g = grid()
    lst = np.full((10, 10), 30.0)
    lst[g["inside"]] = 33.0
    t = {**g, "lst_c": lst, "ndvi": np.full((10, 10), 0.7), "clear": np.ones((10, 10), bool),
         "date": "2026-09-30"}
    out = maps.thermal_map(t, tmp_path)
    assert out["stats"]["diff"] == pytest.approx(3.0)
    assert "warmer" in out["caption"] and (tmp_path / "thermal.png").exists()


def test_overlay_and_bounds():
    g = grid()
    png = maps.overlay_png(np.full((10, 10), 0.5), np.ones((10, 10), bool), "green", 0, 1)
    assert png[:4] == b"\x89PNG"
    w, s, e, n = maps.bounds_lonlat(g, (10, 10))
    assert w < e and s < n and 70 < w < 80
