from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import LineString  # noqa: E402

from hedge_features.features.lidar_structure import add_lidar_hedgerow_structure_features  # noqa: E402


def _write_raster(path: Path, array: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:27700",
        transform=from_origin(0, 20, 1, 1),
        nodata=-9999.0,
    ) as dst:
        dst.write(array.astype("float32"), 1)


def test_lidar_structure_features_compute_canopy_metrics(tmp_path: Path):
    dtm = np.zeros((20, 20), dtype="float32")
    dsm = np.zeros((20, 20), dtype="float32")
    dsm[8:12, 2:18] = 4.0
    dsm[9:11, 6:8] = 7.0

    dtm_path = tmp_path / "dtm.tif"
    dsm_path = tmp_path / "dsm.tif"
    _write_raster(dtm_path, dtm)
    _write_raster(dsm_path, dsm)

    hedges = gpd.GeoDataFrame(
        {"hf_uid": ["h1"]},
        geometry=[LineString([(2, 10), (18, 10)])],
        crs="EPSG:27700",
    )
    result = add_lidar_hedgerow_structure_features(
        hedges,
        dtm_path=str(dtm_path),
        dsm_path=str(dsm_path),
    ).gdf

    assert result.loc[0, "hedge_struct_status"] == "ok"
    assert float(result.loc[0, "hedge_struct_height_mean_5m"]) > 0.0
    assert float(result.loc[0, "hedge_struct_height_p90_5m"]) >= float(result.loc[0, "hedge_struct_height_mean_5m"])
    assert float(result.loc[0, "hedge_struct_canopy_continuity_10m"]) > 0.0
    assert float(result.loc[0, "hedge_struct_width_proxy_m"]) > 0.0
