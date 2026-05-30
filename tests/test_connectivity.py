"""Tests for planar hedgerow-network connectivity."""

from __future__ import annotations

import pytest

gpd = pytest.importorskip("geopandas")
pytest.importorskip("networkx")
from shapely.geometry import LineString

from hsi.connectivity import add_connectivity


def test_planar_connectivity_distinguishes_junction_from_isolated():
    # a-b collinear with c branching at (10,0) form a junction; x is isolated.
    lines = [
        LineString([(0, 0), (10, 0)]),     # a
        LineString([(10, 0), (20, 0)]),    # b
        LineString([(10, 0), (10, 10)]),   # c
        LineString([(100, 100), (110, 100)]),  # x (isolated)
    ]
    gdf = gpd.GeoDataFrame({"hf_uid": ["a", "b", "c", "x"]}, geometry=lines, crs="EPSG:27700")
    out, notes = add_connectivity(gdf)

    deg = dict(zip(out["hf_uid"], out["net_planar_degree"]))
    bc = {k: (v or 0.0) for k, v in zip(out["hf_uid"], out["net_betweenness"])}

    # junction node (10,0) has degree 3
    assert max(deg["a"], deg["b"], deg["c"]) >= 3
    assert deg["x"] == 1
    # connected hedges carry at least as much betweenness as the isolated one
    assert bc["a"] >= bc["x"]
