import json

import pytest

from src import soil

# SoilGrids medians for this field's topsoil (0-5 cm), in g/kg before scaling.
FAKE_SOILGRIDS = {
    "properties": {
        "layers": [
            {"name": "clay", "unit_measure": {"d_factor": 10},
             "depths": [{"label": "0-5cm", "values": {"mean": 258}}, {"label": "5-15cm", "values": {"mean": 268}}]},
            {"name": "sand", "unit_measure": {"d_factor": 10},
             "depths": [{"label": "0-5cm", "values": {"mean": 378}}, {"label": "5-15cm", "values": {"mean": 368}}]},
            {"name": "silt", "unit_measure": {"d_factor": 10},
             "depths": [{"label": "0-5cm", "values": {"mean": 365}}, {"label": "5-15cm", "values": {"mean": 365}}]},
            {"name": "soc", "unit_measure": {"d_factor": 10},
             "depths": [{"label": "0-5cm", "values": {"mean": 177}}, {"label": "5-15cm", "values": {"mean": 150}}]},
            {"name": "bdod", "unit_measure": {"d_factor": 100},
             "depths": [{"label": "0-5cm", "values": {"mean": 137}}, {"label": "5-15cm", "values": {"mean": 139}}]},
        ]
    }
}


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture
def fake_api(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        return FakeResponse(FAKE_SOILGRIDS)

    monkeypatch.setattr(soil.requests, "get", fake_get)
    return calls


def test_fetch_layers_applies_d_factor(tmp_path, fake_api):
    layers = soil.fetch_layers(29.3733, 77.7721, depths=["0-5cm"])
    assert layers["clay"]["0-5cm"] == pytest.approx(25.8)   # 258 g/kg / d_factor 10
    assert layers["soc"]["0-5cm"] == pytest.approx(17.7)    # 177 dg/kg / 10 = g/kg
    assert layers["bdod"]["0-5cm"] == pytest.approx(1.37)   # 137 cg/cm3 / 100 = kg/dm3


def test_range_cm_parses_labels():
    assert soil._range_cm("15-30cm") == (15, 30)
    assert soil._range_cm("0-5cm") == (0, 5)


def test_rescale_to_100():
    assert soil.rescale_to_100(37.8, 36.5, 25.8) == pytest.approx((37.8, 36.5, 25.8), abs=0.15)
    s, si, c = soil.rescale_to_100(50.0, 30.0, 30.0)  # sums to 110
    assert s + si + c == pytest.approx(100.0)


def test_depth_mean_is_thickness_weighted():
    per_depth = {"0-5cm": 10.0, "5-15cm": 20.0, "15-30cm": 30.0}
    # (5*10 + 10*20 + 15*30) / 30
    assert soil.depth_mean(per_depth, 0, 30) == pytest.approx((50 + 200 + 450) / 30)
    assert soil.depth_mean(per_depth, 0, 5) == pytest.approx(10.0)
    assert soil.depth_mean(per_depth, 5, 15) == pytest.approx(20.0)


def test_usda_texture_classes_for_canonical_points():
    assert soil.texture_class(92, 5, 3) == "sand"
    assert soil.texture_class(80, 12, 8) == "loamy_sand"
    assert soil.texture_class(65, 25, 10) == "sandy_loam"
    assert soil.texture_class(37.8, 36.5, 25.8) == "loam"  # this field's topsoil
    assert soil.texture_class(20, 65, 15) == "silt_loam"
    assert soil.texture_class(5, 88, 7) == "silt"
    assert soil.texture_class(60, 15, 25) == "sandy_clay_loam"
    assert soil.texture_class(33, 34, 33) == "clay_loam"  # clay stays under 40 %
    assert soil.texture_class(10, 55, 35) == "silty_clay_loam"
    assert soil.texture_class(20, 30, 50) == "clay"
    assert soil.texture_class(5, 55, 40) == "silty_clay"
    assert soil.texture_class(55, 5, 40) == "sandy_clay"


def test_saxton_rawls_returns_plausible_water_contents():
    r = soil.saxton_rawls_2006(37.8, 25.8, 3.05)
    assert r["theta_wp_m3m3"] < r["theta_fc_m3m3"]
    assert 0.10 < r["theta_wp_m3m3"] < 0.25
    assert 0.20 < r["theta_fc_m3m3"] < 0.40


def test_saxton_rawls_sandier_soil_holds_less_water():
    sandy = soil.saxton_rawls_2006(85.0, 5.0, 1.0)
    clayey = soil.saxton_rawls_2006(15.0, 45.0, 1.0)
    assert clayey["theta_fc_m3m3"] > sandy["theta_fc_m3m3"]
    assert clayey["theta_wp_m3m3"] > sandy["theta_wp_m3m3"]


def test_field_soil_reads_api_and_caches(tmp_path, fake_api):
    out = soil.field_soil(29.3733, 77.7721, top_cm=0, bottom_cm=15, cache_dir=tmp_path)
    assert out["texture_class"] == "loam"
    assert out["clay_pct"] == pytest.approx(26.5, abs=0.6)  # weighted 258/268 over 0-15 cm
    assert out["theta_fc_m3m3"] > out["theta_wp_m3m3"]
    assert out["taw_mm_per_m"] == pytest.approx(
        (out["theta_fc_m3m3"] - out["theta_wp_m3m3"]) * 1000, abs=0.05)

    cached = soil.field_soil(29.3733, 77.7721, top_cm=0, bottom_cm=15, cache_dir=tmp_path)
    assert cached == out
    assert fake_api["n"] == 1  # second call came from disk
    assert json.loads((tmp_path / "soil_29.3733_77.7721_0_15.json").read_text())["texture_class"] == "loam"


def test_depth_mean_reaches_below_one_metre():
    per_depth = {"60-100cm": 20.0, "100-200cm": 40.0}
    assert soil.depth_mean(per_depth, 60, 120) == pytest.approx((20 * 40 + 40 * 20) / 60)
