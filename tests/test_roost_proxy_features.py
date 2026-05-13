import pytest

gpd = pytest.importorskip("geopandas")
from shapely.geometry import LineString, Point  # noqa: E402

from hedge_features.features.proxies_roost import add_roost_proxy_features  # noqa: E402


def test_roost_proxy_features_basic():
    hedges = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[LineString([(0, 0), (100, 0)])],
        crs="EPSG:27700",
    )
    buildings = gpd.GeoDataFrame(
        {"feature_class": ["building"]},
        geometry=[Point(10, 5)],
        crs="EPSG:27700",
    )
    structures = gpd.GeoDataFrame(
        {"feature_class": ["bridge", "tunnel", "cave"]},
        geometry=[Point(150, 0), Point(200, 0), Point(300, 0)],
        crs="EPSG:27700",
    )

    result = add_roost_proxy_features(hedges, buildings_gdf=buildings, structures_gdf=structures)
    out = result.gdf
    assert out.loc[0, "roostpx_status"].startswith("ok_")
    assert out.loc[0, "roostpx_dist_building_m"] is not None
    assert out.loc[0, "roostpx_dist_bridge_m"] is not None
    assert out.loc[0, "roostpx_struct_proxy_score"] is not None

