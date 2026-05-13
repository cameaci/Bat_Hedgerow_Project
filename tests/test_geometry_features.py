import math

import pytest

from hedge_features.features.geometry import orientation_dispersion


gpd = pytest.importorskip("geopandas")
from shapely.geometry import LineString  # noqa: E402

from hedge_features.features.geometry import add_geometry_metrics  # noqa: E402


def test_add_geometry_metrics_basic():
    gdf = gpd.GeoDataFrame(
        {"name": ["a"]},
        geometry=[LineString([(0, 0), (3, 4)])],
        crs="EPSG:27700",
    )
    out = add_geometry_metrics(gdf)
    assert math.isclose(out.loc[0, "geom_length_m"], 5.0, rel_tol=1e-6)
    assert math.isclose(out.loc[0, "geom_endpoint_dist_m"], 5.0, rel_tol=1e-6)
    assert math.isclose(out.loc[0, "geom_sinuosity"], 1.0, rel_tol=1e-6)
    assert 0 <= out.loc[0, "geom_bearing_deg"] < 360


def test_orientation_dispersion_returns_low_for_aligned_angles():
    d = orientation_dispersion([10.0, 12.0, 11.0])
    assert d is not None
    assert d < 0.1

