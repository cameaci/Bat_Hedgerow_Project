"""Tests for LiDAR-derived SI5 tree count and SI7 ditch detection (synthetic rasters)."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")


def _write_raster(path, array, rasterio, transform):
    profile = {
        "driver": "GTiff", "height": array.shape[0], "width": array.shape[1], "count": 1,
        "dtype": "float32", "crs": "EPSG:27700", "transform": transform, "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def test_tree_count_local_maxima(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    np = pytest.importorskip("numpy")
    gpd = pytest.importorskip("geopandas")
    pytest.importorskip("scipy")
    from rasterio.transform import from_origin
    from shapely.geometry import LineString

    from hsi.treecount import add_tree_count

    size = 60
    t = from_origin(450000, 206060, 1.0, 1.0)
    dtm = np.zeros((size, size), dtype="float32")
    dsm = np.zeros((size, size), dtype="float32")
    for cx in (10, 30, 50):  # three crowns ~20 m apart along the corridor
        dsm[29:32, cx - 1:cx + 2] = 8.0
    _write_raster(tmp_path / "dtm.tif", dtm, rasterio, t)
    _write_raster(tmp_path / "dsm.tif", dsm, rasterio, t)

    line = LineString([(450005, 206030), (450055, 206030)])  # 50 m
    gdf = gpd.GeoDataFrame({"hf_uid": ["a"]}, geometry=[line], crs="EPSG:27700")
    out, notes = add_tree_count(
        gdf, dtm_path=str(tmp_path / "dtm.tif"), dsm_path=str(tmp_path / "dsm.tif"),
        tree_height_min_m=3.0, min_separation_m=4.0,
    )
    trees = out.iloc[0]["lidar_trees_per_50m"]
    assert trees is not None and trees >= 2  # ~3 crowns over 50 m


def test_ditch_detection(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    np = pytest.importorskip("numpy")
    gpd = pytest.importorskip("geopandas")
    pytest.importorskip("scipy")
    from rasterio.transform import from_origin
    from shapely.geometry import LineString

    from hsi.ditch import add_ditch

    size = 40
    t = from_origin(450000, 206040, 1.0, 1.0)
    line = LineString([(450005, 206020), (450035, 206020)])
    gdf = gpd.GeoDataFrame({"hf_uid": ["a"]}, geometry=[line], crs="EPSG:27700")

    trench = np.full((size, size), 10.0, dtype="float32")
    trench[19:21, :] = 8.5  # 1.5 m deep ditch along the corridor
    _write_raster(tmp_path / "trench.tif", trench, rasterio, t)
    out, _ = add_ditch(gdf, dtm_path=str(tmp_path / "trench.tif"), corridor_m=8.0,
                       depth_threshold_m=0.25, min_fraction=0.01)
    assert out.iloc[0]["ditch_present"] == 1

    flat = np.full((size, size), 10.0, dtype="float32")
    _write_raster(tmp_path / "flat.tif", flat, rasterio, t)
    out2, _ = add_ditch(gdf, dtm_path=str(tmp_path / "flat.tif"), corridor_m=8.0)
    assert out2.iloc[0]["ditch_present"] == 0
