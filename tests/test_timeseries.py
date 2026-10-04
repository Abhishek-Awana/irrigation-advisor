from datetime import date, timedelta

import numpy as np
import pandas as pd

from src import timeseries


def scenes():
    today = date.today()
    return pd.DataFrame({
        "date": [today - timedelta(days=d) for d in (10, 5, 1)],
        "id": ["a", "b", "c"],
        "cloud_pct": [5.0, 30.0, 0.0],
        "tile": ["T"] * 3,
    })


def fake_row(s):
    return {"date": s["date"], "id": s["id"], "scene_cloud_pct": s["cloud_pct"], "clear_fraction": 1.0,
            "ndvi": 0.6, "ndmi": 0.2, "usable": True, "top_scl": "vegetation"}


def patch_search(monkeypatch):
    monkeypatch.setattr(timeseries, "find_scenes", lambda geom, days, max_cloud_pct: scenes())


def test_cache_means_scenes_are_processed_once(tmp_path, monkeypatch):
    patch_search(monkeypatch)
    calls = []
    row_fn = lambda s: (calls.append(s["id"]), fake_row(s))[1]  # noqa: E731
    cache = tmp_path / "series.csv"
    df = timeseries.build_series(None, cache_path=cache, row_fn=row_fn)
    timeseries.build_series(None, cache_path=cache, row_fn=row_fn)
    assert calls == ["a", "b", "c"]  # second run downloaded nothing
    assert list(df["id"]) == ["a", "b", "c"]


def test_one_bad_scene_does_not_stop_the_rest(tmp_path, monkeypatch):
    patch_search(monkeypatch)

    def row_fn(s):
        if s["id"] == "b":
            raise RuntimeError("network")
        return fake_row(s)

    df = timeseries.build_series(None, cache_path=tmp_path / "s.csv", row_fn=row_fn)
    assert list(df["id"]) == ["a", "c"]


def test_summary_uses_latest_usable_scene():
    today = pd.Timestamp.today().normalize()
    df = pd.DataFrame({
        "date": [today - pd.Timedelta(days=d) for d in (10, 3, 2)],
        "ndvi": [0.6, None, 0.7], "ndmi": [0.20, None, 0.15], "usable": [True, False, True],
    })
    s = timeseries.satellite_summary(df)
    assert s["age_days"] == 2 and s["days_between"] == 8
    assert abs(s["ndmi_change"] - (-0.05)) < 1e-9


def test_summary_when_nothing_usable():
    df = pd.DataFrame({"date": [pd.Timestamp.today()], "ndvi": [None], "ndmi": [None], "usable": [False]})
    assert timeseries.satellite_summary(df) == {"available": False}


def test_top_scl_label():
    scl = np.array([4, 4, 8], dtype="uint8")
    assert timeseries.top_scl_name(scl, np.array([True, True, False])) == "vegetation"