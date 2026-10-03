import json
from pathlib import Path

def test_config_loads():
    cfg = json.loads(Path("config/field.json").read_text())
    assert -90 <= cfg["latitude"] <= 90
    assert -180 <= cfg["longitude"] <= 180
    assert cfg["crop"]

def test_imports():
    import streamlit, requests, pandas, numpy, ollama  # noqa