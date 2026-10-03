import pandas as pd
import pytest
import requests

from src import weather


def fake_payload():
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=2)
    days = pd.date_range(end=end, periods=10).strftime("%Y-%m-%d").tolist()
    n = len(days)
    return {
        "daily": {
            "time": days,
            "precipitation_sum": [1.0] * (n - 1) + [None],  # last day missing
            "et0_fao_evapotranspiration": [3.0] * n,
            "temperature_2m_max": [30.0] * n,
            "temperature_2m_min": [15.0] * n,
        }
    }


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return fake_payload()


@pytest.fixture
def calls(monkeypatch):
    counter = {"n": 0}

    def fake_get(*args, **kwargs):
        counter["n"] += 1
        return FakeResponse()

    monkeypatch.setattr(weather.requests, "get", fake_get)
    return counter


def test_table_shape_and_flags(tmp_path, calls):
    df = weather.fetch_weather(27.18, 78.0, cache_dir=tmp_path)
    assert list(df.columns) == ["date", "rain_mm", "et0_mm", "tmax_c", "tmin_c", "is_forecast"]
    assert df["rain_mm"].isna().sum() == 0  # missing rain filled with 0
    assert int(df["is_forecast"].sum()) == 2  # today+1 and today+2


def test_cache_avoids_second_download(tmp_path, calls):
    weather.fetch_weather(27.18, 78.0, cache_dir=tmp_path)
    weather.fetch_weather(27.18, 78.0, cache_dir=tmp_path)
    assert calls["n"] == 1


def test_offline_falls_back_to_stale_cache(tmp_path, calls, monkeypatch):
    weather.fetch_weather(27.18, 78.0, cache_dir=tmp_path)

    def boom(*a, **k):
        raise requests.ConnectionError("no internet")

    monkeypatch.setattr(weather.requests, "get", boom)
    with pytest.warns(UserWarning):
        df = weather.fetch_weather(27.18, 78.0, use_cache=False, cache_dir=tmp_path)
    assert len(df) == 10


def test_summarize(tmp_path, calls):
    df = weather.fetch_weather(27.18, 78.0, cache_dir=tmp_path)
    s = weather.summarize(df)
    assert s["rain_last_7d_mm"] == 7.0  # 7 past days x 1 mm
    assert s["rain_next_5d_mm"] == 1.0  # tomorrow 1 mm, day after filled 0
    assert s["et0_last_7d_mm"] == 21.0