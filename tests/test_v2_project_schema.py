from __future__ import annotations

import pandas as pd
import pytest

from hedge_features.v2 import validate_project_dataset


def test_project_schema_reports_ready_for_complete_geodata():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import LineString

    gdf = gpd.GeoDataFrame(
        {
            "project_id": ["p1", "p1"],
            "scheme_name": ["EGL", "EGL"],
            "section_id": ["S1", "S1"],
            "hedgerow_id": ["h1", "h2"],
            "detector_id": ["d1", "d2"],
            "detector_model": ["SM4", "SM4"],
            "microphone_height_m": [2.0, 2.0],
            "survey_season": ["Spring", "Spring"],
        },
        geometry=[LineString([(0, 0), (10, 0)]), LineString([(0, 10), (10, 10)])],
        crs="EPSG:27700",
    )

    report = validate_project_dataset(gdf)

    assert report["status"] == "ready"
    assert report["geometry_status"] == "geospatial"
    assert report["crs"] == "EPSG:27700"


def test_project_schema_blocks_duplicate_or_missing_id():
    df = pd.DataFrame(
        {
            "project_id": ["p1", "p1"],
            "scheme_name": ["EGL", "EGL"],
            "section_id": ["S1", "S1"],
            "hedgerow_id": ["h1", "h1"],
        }
    )

    report = validate_project_dataset(df)

    assert report["status"] == "blocked"
    assert report["duplicate_hedgerow_ids"] == 1
    assert any("duplicate" in issue for issue in report["issues"])
    assert report["missing_acoustic_effort_columns"]
