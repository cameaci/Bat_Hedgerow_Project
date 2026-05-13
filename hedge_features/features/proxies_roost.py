from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .vector import (
    VectorFeatureResult,
    add_vector_distance,
    add_vector_feature_count_in_buffers,
)


@dataclass(slots=True)
class RoostProxyFeatureResult:
    gdf: Any
    notes: list[str]


def add_roost_proxy_features(
    hedges_gdf,
    *,
    buildings_gdf=None,
    structures_gdf=None,
):
    gdf = hedges_gdf.copy()
    notes: list[str] = []
    cols = [
        "roostpx_dist_building_m",
        "roostpx_bldg_density_100m",
        "roostpx_bldg_density_250m",
        "roostpx_dist_bridge_m",
        "roostpx_bridge_count_250m",
        "roostpx_dist_tunnel_m",
        "roostpx_tunnel_count_500m",
        "roostpx_dist_cave_m",
        "roostpx_dist_ancwood_m",
        "roostpx_struct_proxy_score",
        "roostpx_status",
    ]
    for col in cols:
        if col not in gdf.columns:
            gdf[col] = None

    buildings_ok = buildings_gdf is not None and not buildings_gdf.empty
    structures_ok = structures_gdf is not None and not structures_gdf.empty
    if not buildings_ok and not structures_ok:
        gdf["roostpx_status"] = "source_missing"
        return RoostProxyFeatureResult(gdf=gdf, notes=["Roost proxies skipped: OSM buildings/structures datasets unavailable."])

    if buildings_ok:
        res_dist = add_vector_distance(gdf, buildings_gdf, distance_column="roostpx_dist_building_m")
        gdf = res_dist.gdf
        notes.extend(res_dist.notes)
        res_counts = add_vector_feature_count_in_buffers(
            gdf,
            buildings_gdf,
            radii_m=[100, 250],
            count_column_template="roostpx_bldg_count_{radius}m",
            density_column_template="roostpx_bldg_density_{radius}m",
        )
        gdf = res_counts.gdf
        notes.extend(res_counts.notes)
        for tmp in ("roostpx_bldg_count_100m", "roostpx_bldg_count_250m"):
            if tmp in gdf.columns:
                gdf = gdf.drop(columns=[tmp])
    else:
        gdf["roostpx_dist_building_m"] = None
        gdf["roostpx_bldg_density_100m"] = None
        gdf["roostpx_bldg_density_250m"] = None

    if structures_ok:
        bridge_gdf = _filter_structures(structures_gdf, "bridge")
        tunnel_gdf = _filter_structures(structures_gdf, "tunnel")
        cave_gdf = _filter_structures(structures_gdf, "cave")

        gdf = add_vector_distance(gdf, bridge_gdf, distance_column="roostpx_dist_bridge_m").gdf
        gdf = add_vector_distance(gdf, tunnel_gdf, distance_column="roostpx_dist_tunnel_m").gdf
        gdf = add_vector_distance(gdf, cave_gdf, distance_column="roostpx_dist_cave_m").gdf

        gdf = add_vector_feature_count_in_buffers(
            gdf, bridge_gdf, radii_m=[250], count_column_template="roostpx_bridge_count_{radius}m"
        ).gdf
        gdf = add_vector_feature_count_in_buffers(
            gdf, tunnel_gdf, radii_m=[500], count_column_template="roostpx_tunnel_count_{radius}m"
        ).gdf
    else:
        for c in (
            "roostpx_dist_bridge_m",
            "roostpx_bridge_count_250m",
            "roostpx_dist_tunnel_m",
            "roostpx_tunnel_count_500m",
            "roostpx_dist_cave_m",
        ):
            gdf[c] = None

    if "dist_awi_ancwood_m" in gdf.columns:
        gdf["roostpx_dist_ancwood_m"] = gdf["dist_awi_ancwood_m"]

    gdf["roostpx_struct_proxy_score"] = _compute_roost_proxy_score(gdf)
    gdf["roostpx_status"] = _roost_status(gdf, buildings_ok=buildings_ok, structures_ok=structures_ok)
    notes.append("Roost proxy features computed from open OSM building/structure proxies (not exact roost records).")
    return RoostProxyFeatureResult(gdf=gdf, notes=notes)


def _filter_structures(gdf, feature_class: str):
    if gdf is None or gdf.empty or "feature_class" not in gdf.columns:
        return gdf.iloc[0:0] if gdf is not None else None
    return gdf[gdf["feature_class"].astype(str).str.lower() == feature_class.lower()].copy()


def _compute_roost_proxy_score(gdf):
    import pandas as pd

    def series(colname: str):
        if colname in gdf.columns:
            return pd.to_numeric(gdf[colname], errors="coerce")
        return pd.Series([None] * len(gdf), index=gdf.index, dtype="float64")

    def invdist(series, scale):
        s = pd.to_numeric(series, errors="coerce")
        return 1.0 / (1.0 + (s / float(scale)))

    b = invdist(series("roostpx_dist_building_m"), 100.0).fillna(0)
    br = invdist(series("roostpx_dist_bridge_m"), 150.0).fillna(0)
    tn = invdist(series("roostpx_dist_tunnel_m"), 200.0).fillna(0)
    cv = invdist(series("roostpx_dist_cave_m"), 250.0).fillna(0)
    anc = invdist(series("roostpx_dist_ancwood_m"), 500.0).fillna(0)
    bd100 = series("roostpx_bldg_density_100m").fillna(0).clip(lower=0, upper=20) / 20.0
    br250 = series("roostpx_bridge_count_250m").fillna(0).clip(lower=0, upper=5) / 5.0
    tn500 = series("roostpx_tunnel_count_500m").fillna(0).clip(lower=0, upper=5) / 5.0
    score = (
        0.30 * b +
        0.10 * br +
        0.10 * tn +
        0.05 * cv +
        0.10 * anc +
        0.25 * bd100 +
        0.05 * br250 +
        0.05 * tn500
    ).clip(lower=0, upper=1)
    return score


def _roost_status(gdf, *, buildings_ok: bool, structures_ok: bool):
    if not buildings_ok and not structures_ok:
        return "source_missing"
    status = []
    if buildings_ok:
        status.append("bldg")
    if structures_ok:
        status.append("struct")
    return "ok_" + "_".join(status)
