"""The farmer's irrigation record: the one input satellites and weather cannot give us.

Stored in data/irrigation_log.csv (kept on this machine, not in git). Each row is one
irrigation: the date, and either the depth in mm, or hours of pumping, or neither
(then the root zone is assumed refilled - typical of flood irrigation).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

LOG_PATH = Path("data/irrigation_log.csv")
COLUMNS = ["date", "depth_mm", "hours", "note"]


def load_log(path: Path = LOG_PATH) -> pd.DataFrame:
    """All recorded irrigations, oldest first (empty table if none yet)."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, parse_dates=["date"])
    df["note"] = df["note"].fillna("").astype(str)
    return df.sort_values("date").reset_index(drop=True)


def add_entry(day, depth_mm: float | None = None, hours: float | None = None, note: str = "",
              path: Path = LOG_PATH) -> pd.DataFrame:
    """Record one irrigation. A second entry for the same day replaces the first."""
    df = load_log(path)
    day = pd.Timestamp(day).normalize()
    row = pd.DataFrame([{"date": day, "depth_mm": depth_mm, "hours": hours, "note": note}], columns=COLUMNS)
    df = df[df["date"] != day]
    df = pd.concat([f for f in (df, row) if len(f)], ignore_index=True).sort_values("date").reset_index(drop=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def remove_entry(day, path: Path = LOG_PATH) -> pd.DataFrame:
    """Delete the irrigation recorded on `day` (for typing mistakes)."""
    df = load_log(path)
    df = df[df["date"] != pd.Timestamp(day).normalize()].reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def depth_from_hours_mm(hours: float, flow_lps: float, area_m2: float) -> float:
    """mm of water put on the field by `hours` of pumping (1 litre per m2 = 1 mm)."""
    return hours * 3600 * flow_lps / area_m2


def engine_inputs(df: pd.DataFrame, area_m2: float | None = None, flow_lps: float | None = None):
    """(last_irrigation, {date: net mm or None}) for engine.decide().

    Depth: the recorded mm, else pump hours x flow / area, else None (= full refill).
    The gross depth released is reduced by the same efficiency the engine assumes,
    because only that share reaches the roots.
    """
    from src.engine import IRRIGATION_EFFICIENCY

    if df.empty:
        return None, {}
    events = {}
    for r in df.itertuples():
        gross = None
        if pd.notna(r.depth_mm):
            gross = float(r.depth_mm)
        elif pd.notna(r.hours) and area_m2 and flow_lps:
            gross = depth_from_hours_mm(float(r.hours), flow_lps, area_m2)
        events[pd.Timestamp(r.date)] = None if gross is None else round(gross * IRRIGATION_EFFICIENCY, 1)
    return max(events), events
