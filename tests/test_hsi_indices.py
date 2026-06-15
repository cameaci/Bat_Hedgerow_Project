"""Tests for SI proxy resolution and (when GIS libs are present) LiDAR extraction."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from hsi.score import resolve_structural


def test_proxy_resolution_uses_remote_columns():
    """With only remote proxy columns, SIs resolve at proxy/default confidence."""
    df = pd.DataFrame([{
        "hedge_struct_height_p90_5m": 2.5,        # SI1 -> band 3 (>=2 m)
        "hedge_struct_width_proxy_m": 1.6,        # SI2 -> band 3 (>=1.5 m)
        "hedge_struct_gap_fraction_10m": 0.05,    # SI3 -> band 3 (<10%)
        "hedge_struct_tree_standard_pct_10m": 0.5,  # SI5 -> band 4
        "buf100_worldcover_cropland_pct": 0.30,   # SI4 -> band 3
        "dist_os_river_m": 10.0,                  # SI7 -> present (band 2)
    }])
    out = resolve_structural(df)
    assert out.loc[0, "hsi_si1_score"] == 3 and out.loc[0, "hsi_si1_source"] == "proxy"
    assert out.loc[0, "hsi_si2_score"] == 3
    assert out.loc[0, "hsi_si3_score"] == 3
    assert out.loc[0, "hsi_si4_score"] == 3 and out.loc[0, "hsi_si4_source"] == "proxy"
    assert out.loc[0, "hsi_si5_score"] == 4
    assert out.loc[0, "hsi_si6_source"] == "default"
    assert out.loc[0, "hsi_si7_score"] == 2
    assert bool(out.loc[0, "hsi_complete"]) is True


def test_lidar_extraction_end_to_end(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    np = pytest.importorskip("numpy")
    gpd = pytest.importorskip("geopandas")
    from rasterio.transform import from_origin
    from shapely.geometry import LineString

    from hedge_features.features.lidar_structure import add_lidar_hedgerow_structure_features

    size = 50
    transform = from_origin(450000, 206050, 1.0, 1.0)  # 1 m pixels, EPSG:27700
    dtm = np.zeros((size, size), dtype="float32")
    dsm = np.zeros((size, size), dtype="float32")
    dsm[24:27, :] = 7.0  # a 3 m-wide canopy band ~7 m tall across the middle

    profile = {
        "driver": "GTiff", "height": size, "width": size, "count": 1,
        "dtype": "float32", "crs": "EPSG:27700", "transform": transform, "nodata": -9999.0,
    }
    dtm_path = tmp_path / "dtm.tif"
    dsm_path = tmp_path / "dsm.tif"
    with rasterio.open(dtm_path, "w", **profile) as dst:
        dst.write(dtm, 1)
    with rasterio.open(dsm_path, "w", **profile) as dst:
        dst.write(dsm, 1)

    line = LineString([(450010, 206025), (450040, 206025)])
    gdf = gpd.GeoDataFrame({"hf_uid": ["a"]}, geometry=[line], crs="EPSG:27700")

    result = add_lidar_hedgerow_structure_features(
        gdf, dtm_path=str(dtm_path), dsm_path=str(dsm_path)
    )
    row = result.gdf.iloc[0]
    assert row["hedge_struct_status"] == "ok"
    assert row["hedge_struct_height_p90_5m"] is not None and float(row["hedge_struct_height_p90_5m"]) >= 1.5

    scored = resolve_structural(result.gdf)
    assert scored.loc[0, "hsi_si1_source"] == "proxy"
    assert scored.loc[0, "hsi_si1_score"] in (2, 3)
