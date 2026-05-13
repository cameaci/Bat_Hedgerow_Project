import pandas as pd
import pytest

from hedge_features.pipeline import _drop_all_null_feature_columns


def test_drop_all_null_feature_columns_keeps_original_columns():
    df = pd.DataFrame(
        {
            "user_col": [None, None],
            "hf_uid": ["a", "b"],
            "geom_length_m": [1.0, 2.0],
            "dist_os_road_m": [None, None],
        }
    )
    out, notes = _drop_all_null_feature_columns(df, original_columns=["user_col", "hf_uid"])
    assert "user_col" in out.columns
    assert "geom_length_m" in out.columns
    assert "dist_os_road_m" not in out.columns
    assert notes


gpd = pytest.importorskip("geopandas")
from shapely.geometry import LineString  # noqa: E402

from hedge_features.pipeline import _qa_line_geometry_output  # noqa: E402


def test_output_geometry_qc_accepts_lines():
    gdf = gpd.GeoDataFrame(
        {"x": [1]},
        geometry=[LineString([(0, 0), (1, 1)])],
        crs="EPSG:27700",
    )
    notes = _qa_line_geometry_output(gdf)
    assert any("line types only" in n for n in notes)

