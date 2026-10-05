import pandas as pd
import pytest

from src import engine, irrigation_log as log


def test_empty_log(tmp_path):
    df = log.load_log(tmp_path / "log.csv")
    assert df.empty and list(df.columns) == log.COLUMNS
    assert log.engine_inputs(df) == (None, {})


def test_add_replace_and_remove(tmp_path):
    p = tmp_path / "log.csv"
    log.add_entry("2026-09-01", depth_mm=70, path=p)
    log.add_entry("2026-09-20", hours=5, note="canal", path=p)
    df = log.add_entry("2026-09-01", depth_mm=60, path=p)  # same day: replaced
    assert len(df) == 2 and df.iloc[0]["depth_mm"] == 60
    assert log.load_log(p).iloc[1]["note"] == "canal"
    assert len(log.remove_entry("2026-09-20", path=p)) == 1


def test_depth_from_hours():
    # 1 hour at 10 L/s on 3600 m2 = 36,000 L / 3600 m2 = 10 mm
    assert log.depth_from_hours_mm(1, 10, 3600) == pytest.approx(10)


def test_engine_inputs_turn_records_into_net_depths(tmp_path):
    p = tmp_path / "log.csv"
    log.add_entry("2026-09-01", depth_mm=100, path=p)
    log.add_entry("2026-09-10", hours=1, path=p)
    log.add_entry("2026-09-20", path=p)  # nothing known: full refill
    last, events = log.engine_inputs(log.load_log(p), area_m2=3600, flow_lps=10)
    assert last == pd.Timestamp("2026-09-20")
    assert events[pd.Timestamp("2026-09-01")] == pytest.approx(100 * engine.IRRIGATION_EFFICIENCY)
    assert events[pd.Timestamp("2026-09-10")] == pytest.approx(10 * engine.IRRIGATION_EFFICIENCY)
    assert events[pd.Timestamp("2026-09-20")] is None


def test_hours_without_pump_details_count_as_full_refill(tmp_path):
    p = tmp_path / "log.csv"
    log.add_entry("2026-09-10", hours=3, path=p)
    _, events = log.engine_inputs(log.load_log(p))
    assert events[pd.Timestamp("2026-09-10")] is None
