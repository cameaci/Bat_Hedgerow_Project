from datetime import datetime

import pytest

gpd = pytest.importorskip("geopandas")
from shapely.geometry import LineString  # noqa: E402

from hedge_features.features.deployment import add_deployment_nights_metrics  # noqa: E402


def test_deployment_nights_metrics_basic():
    gdf = gpd.GeoDataFrame(
        {
            "deployment_start": ["2025-06-01 20:30"],
            "deployment_end": ["2025-06-03 05:30"],
        },
        geometry=[LineString([(-0.2, 51.5), (-0.19, 51.505)])],
        crs="EPSG:4326",
    ).to_crs("EPSG:27700")

    result = add_deployment_nights_metrics(
        gdf,
        start_column="deployment_start",
        end_column="deployment_end",
        timezone_name="Europe/London",
        min_night_overlap_minutes=30,
    )
    out = result.gdf
    assert result.context is not None
    assert out.loc[0, "deploy_status"] == "ok"
    assert int(out.loc[0, "deploy_nights_count"]) == 2
    assert float(out.loc[0, "deploy_total_night_hours"]) > 0


def test_deployment_metrics_missing_mapping_sets_status():
    gdf = gpd.GeoDataFrame(
        {"x": [1]},
        geometry=[LineString([(0, 0), (1, 1)])],
        crs="EPSG:27700",
    )
    result = add_deployment_nights_metrics(gdf, start_column=None, end_column=None)
    assert result.context is None
    assert result.gdf.loc[0, "deploy_status"] == "missing_column_mapping"

