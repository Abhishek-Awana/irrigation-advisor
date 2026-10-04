import numpy as np

from src import indices

META = {b: (1e-4, -0.1, "test") for b in ("B04", "B08", "B11")}


def make(n_cloud=0, n=16):
    """n pixels: red DN 2000 (0.1), nir DN 4000 (0.3), swir DN 3000 (0.2); first n_cloud are cloud."""
    scl = np.full(n, 4, dtype="uint8")
    scl[:n_cloud] = 8
    data = {
        "B04": np.full(n, 2000, dtype="uint16"),
        "B08": np.full(n, 4000, dtype="uint16"),
        "B11": np.full(n, 3000, dtype="uint16"),
        "SCL": scl,
    }
    return data, np.ones(n, dtype=bool)


def test_reflectance_applies_offset_and_flags_nodata():
    out = indices.to_reflectance(np.array([0, 2000, 10000], dtype="uint16"), 1e-4, -0.1)
    assert np.isnan(out[0])
    assert np.allclose(out[1:], [0.1, 0.9])


def test_known_index_values():
    data, inside = make()
    r = indices.field_indices(data, inside, META)
    assert abs(r["ndvi"] - 0.5) < 1e-4  # (0.3-0.1)/(0.3+0.1)
    assert abs(r["ndmi"] - 0.2) < 1e-4  # (0.3-0.2)/(0.3+0.2)
    assert r["clear_fraction"] == 1.0 and r["usable"]


def test_cloudy_pixels_are_excluded():
    data, inside = make(n_cloud=4)
    r = indices.field_indices(data, inside, META)
    assert r["clear_pixels"] == 12 and r["field_pixels"] == 16 and r["usable"]


def test_mostly_cloudy_scene_is_unusable():
    data, inside = make(n_cloud=14)
    r = indices.field_indices(data, inside, META)
    assert not r["usable"]