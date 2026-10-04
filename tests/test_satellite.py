import json

from shapely.geometry import Point

from src import satellite


def test_fallback_square_around_field_point(tmp_path):
    cfg = tmp_path / "field.json"
    cfg.write_text(json.dumps({"latitude": 27.0, "longitude": 78.0}))
    geom = satellite.load_field_geometry(cfg, tmp_path / "missing.geojson")
    assert geom.contains(Point(78.0, 27.0))
    assert geom.geom_type == "Polygon"


def test_reads_feature_collection(tmp_path):
    ring = [[78, 27], [78.001, 27], [78.001, 27.001], [78, 27.001], [78, 27]]
    gj = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [ring]}}],
    }
    path = tmp_path / "field.geojson"
    path.write_text(json.dumps(gj))
    geom = satellite.load_field_geometry(tmp_path / "unused.json", path)
    assert geom.geom_type == "Polygon"
    assert abs(geom.area - 1e-6) < 1e-9