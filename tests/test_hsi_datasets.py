"""Tests for the data resolver and WCS URL builder (offline only)."""

from __future__ import annotations

import pytest

from hsi.datasets import build_wcs_getcoverage_url


def test_wcs_url_builder():
    url = build_wcs_getcoverage_url("https://env/wcs", "lidar_dtm", (1.0, 2.0, 3.0, 4.0), "E", "N")
    assert "service=WCS" in url and "version=2.0.1" in url
    assert "request=GetCoverage" in url and "coverageId=lidar_dtm" in url
    assert "subset=E(1.00,3.00)" in url and "subset=N(2.00,4.00)" in url


def test_resolver_degrades_offline():
    pytest.importorskip("geopandas")
    import geopandas as gpd
    from shapely.geometry import LineString

    from hsi.datasets import DataResolver

    gdf = gpd.GeoDataFrame(
        {"hf_uid": ["a"]},
        geometry=[LineString([(450000, 206000), (450100, 206000)])],
        crs="EPSG:27700",
    )
    resolver = DataResolver(gdf, allow_live_fetch=False)
    # WCS LiDAR disabled offline, no local tiles -> None, never raises
    assert resolver.get_raster_path("ea_lidar_dtm") is None
    assert resolver.get_raster_path("worldcover") is None
    assert resolver.get_vector("ne_phi") is None
