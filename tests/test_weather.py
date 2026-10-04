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


# --- Step 6: archive weather and ERA5-Land soil moisture for validation ---------------

class FakeSoilMoistureResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {
            "daily": {
                "time": ["2026-09-01", "2026-09-02", "2026-09-03"],
                "soil_moisture_0_to_7cm_mean": [0.35, 0.30, 0.28],
                "soil_moisture_7_to_28cm_mean": [0.37, 0.36, 0.34],
            }
        }


def test_fetch_soil_moisture_parses_and_caches(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        assert params["daily"] == "soil_moisture_0_to_7cm_mean,soil_moisture_7_to_28cm_mean"
        assert url == weather.ARCHIVE_URL
        return FakeSoilMoistureResponse()

    monkeypatch.setattr(weather.requests, "get", fake_get)
    df = weather.fetch_soil_moisture(29.373, 77.772, "2026-09-01", "2026-09-03", cache_dir=tmp_path)
    assert list(df.columns) == ["date", "sm_0_7_m3m3", "sm_7_28_m3m3"]
    assert len(df) == 3
    assert df["sm_7_28_m3m3"].iloc[0] == 0.37

    again = weather.fetch_soil_moisture(29.373, 77.772, "2026-09-01", "2026-09-03", cache_dir=tmp_path)
    assert len(again) == 3 and calls["n"] == 1  # second call came off disk


def test_fetch_weather_archive_is_all_observed_and_cached(tmp_path, monkeypatch):
    class PastOnly:
        def raise_for_status(self):
            pass

        def json(self):
            return {"daily": {"time": ["2026-09-01", "2026-09-02"], "precipitation_sum": [0.0, 1.0],
                              "et0_fao_evapotranspiration": [4.0, 4.5], "temperature_2m_max": [33.0, 32.0],
                              "temperature_2m_min": [22.0, 21.0]}}

    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        assert url == weather.ARCHIVE_URL
        assert params["start_date"] == "2026-09-01"
        return PastOnly()

    monkeypatch.setattr(weather.requests, "get", fake_get)
    df = weather.fetch_weather_archive(29.373, 77.772, "2026-09-01", "2026-09-02", cache_dir=tmp_path)
    assert list(df.columns) == ["date", "rain_mm", "et0_mm", "tmax_c", "tmin_c", "is_forecast"]
    assert int(df["is_forecast"].sum()) == 0  # the archive is history, never a forecast
    assert df["rain_mm"].tolist() == [0.0, 1.0]

    weather.fetch_weather_archive(29.373, 77.772, "2026-09-01", "2026-09-02", cache_dir=tmp_path)
    assert calls["n"] == 1  # second call came off disk



def test_cache_is_keyed_by_window(tmp_path, calls):
    weather.fetch_weather(27.18, 78.0, past_days=30, cache_dir=tmp_path)
    weather.fetch_weather(27.18, 78.0, past_days=90, cache_dir=tmp_path)
    assert calls["n"] == 2  # a 30-day cache must not answer a 90-day request
