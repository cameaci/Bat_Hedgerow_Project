from pathlib import Path

import pytest

gpd = pytest.importorskip("geopandas")
yaml = pytest.importorskip("yaml")
from shapely.geometry import LineString  # noqa: E402

from hedge_features.models import RunOptions  # noqa: E402
from hedge_features.pipeline import run_enrichment  # noqa: E402


def test_end_to_end_geometry_only_profile(tmp_path: Path):
    input_path = tmp_path / "hedges.geojson"
    output_path = tmp_path / "enriched.geojson"
    profile_path = tmp_path / "profile.yaml"

    gdf = gpd.GeoDataFrame(
        {"user_id": [1, 2]},
        geometry=[
            LineString([(0, 0), (10, 0)]),
            LineString([(10, 0), (20, 10)]),
        ],
        crs="EPSG:27700",
    )
    gdf.to_file(input_path, driver="GeoJSON")

    profile = {
        "name": "test_geometry_only",
        "working_crs": "EPSG:27700",
        "buffers_m": [100],
        "output": {"id_column": "hf_uid"},
        "datasets": {},
        "feature_modules": [
            {"type": "geometry_metrics", "enabled": True},
            {"type": "network_metrics", "enabled": False},
        ],
    }
    profile_path.write_text(yaml.safe_dump(profile), encoding="utf-8")

    result = run_enrichment(
        RunOptions(
            input_path=input_path,
            output_path=output_path,
            profile_name="ignored_when_profile_file_passed",
            profile_path=profile_path,
            working_crs="EPSG:27700",
        )
    )

    assert result["rows"] == 2
    assert output_path.exists()
    assert output_path.with_name("METADATA.json").exists()
    assert output_path.with_name("DATA_CATALOGUE.json").exists()
    assert output_path.with_name("FEATURE_HEALTH.json").exists()
    out = gpd.read_file(output_path)
    assert "geom_length_m" in out.columns
    assert "hf_uid" in out.columns
    assert "data_catalogue" in result
    assert "feature_health" in result
