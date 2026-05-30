"""Landscape-context sub-indices (stage B).

Each ``ctx_*`` column is on a 0-1 scale where higher = more suitable for bats. These do
NOT affect the WSP structural category; they refine the final survey-priority ranking.
Every sub-index is the mean of whatever component signals are available, so missing data
degrades the value rather than crashing. All weight-independent (computed once).
"""

from __future__ import annotations

from hedge_features.features.raster import add_raster_zonal_stats_in_buffers
from hedge_features.features.vector import (
    add_vector_distance,
    add_vector_polygon_composition_in_buffers,
)

from . import config
from .datasets import DataResolver
from .score import _first_number, _is_missing


# -- normalisation helpers ----------------------------------------------------------

def _invdist(distance, scale):
    """Closer is better: 1 at distance 0, -> 0 far away."""
    if distance is None or _is_missing(distance):
        return None
    return float(scale) / (float(scale) + max(0.0, float(distance)))


def _fardist(distance, scale):
    """Farther is better (e.g. quiet roads, darkness)."""
    inv = _invdist(distance, scale)
    return None if inv is None else 1.0 - inv


def _scaled(value, cap):
    if value is None or _is_missing(value):
        return None
    v = float(value) / float(cap)
    return 0.0 if v < 0 else 1.0 if v > 1 else v


def _mean_present(values):
    present = [float(v) for v in values if v is not None and not _is_missing(v)]
    if not present:
        return None
    return round(sum(present) / len(present), 4)


def compute_context_subindices(gdf, resolver: DataResolver):
    """Add ctx_* columns to ``gdf``. Returns ``(gdf, notes)``."""
    notes: list[str] = []
    scales = config.CONTEXT_SCALES_M

    # --- gather context layers ------------------------------------------------------
    phi = resolver.get_vector("ne_phi")
    if phi is not None and not phi.empty:
        phi_field = next((f for f in ("MainHabs", "MAINHABS", "mainhabs") if f in phi.columns), None)
        if phi_field:
            try:
                comp = add_vector_polygon_composition_in_buffers(
                    gdf, phi, radii_m=[250],
                    class_field=phi_field,
                    selected_classes={"broadleaved_woodland": {"contains_any": ["Woodland", "Broadleaved"]}},
                    column_template="buf{radius}_phi_{class_name}_pct",
                )
                gdf = comp.gdf
                notes.extend(comp.notes)
            except Exception as exc:  # noqa: BLE001
                notes.append(f"PHI woodland composition failed ({exc}).")

    le = resolver.get_vector("living_england")
    if le is not None and not le.empty:
        le_field = next((f for f in config.LIVING_ENGLAND_FIELD_CANDIDATES if f in le.columns), None)
        if le_field:
            try:
                comp = add_vector_polygon_composition_in_buffers(
                    gdf, le, radii_m=[250],
                    class_field=le_field,
                    selected_classes={"le_broadleaf": {"contains_any": list(config.LIVING_ENGLAND_WOODLAND_TOKENS)}},
                    column_template="buf{radius}_{class_name}_pct",
                )
                gdf = comp.gdf
                notes.extend(comp.notes)
            except Exception as exc:  # noqa: BLE001
                notes.append(f"Living England woodland composition failed ({exc}).")

    awi = resolver.get_vector("ne_awi")
    gdf = _add_distance(gdf, awi, "dist_awi_ancwood_m", ["Polygon", "MultiPolygon"], notes)

    buildings = resolver.get_vector("osm_buildings")
    gdf = _add_distance(gdf, buildings, "dist_building_m", ["Point", "Polygon", "MultiPolygon"], notes)

    structures = resolver.get_vector("osm_structures_roost")
    gdf = _add_distance(gdf, structures, "dist_structure_m", ["Point", "LineString", "Polygon"], notes)

    roads = resolver.get_vector("os_open_roads")
    gdf = _add_distance(gdf, roads, "dist_os_road_m", ["LineString", "MultiLineString"], notes)

    trees = resolver.get_vector("ancient_trees")
    gdf = _add_distance(gdf, trees, "dist_ancient_tree_m", ["Point"], notes)

    # Night-lights raster (optional). Lower radiance = darker = better.
    nl_path = resolver.get_raster_path("nightlights")
    has_nightlights = nl_path is not None
    if has_nightlights:
        try:
            nl = add_raster_zonal_stats_in_buffers(
                gdf, nl_path, radii_m=[250], stats=["mean"],
                column_template="buf{radius}_nightlight_{stat}",
            )
            gdf = nl.gdf
            notes.extend(nl.notes)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Night-lights zonal step failed ({exc}); using distance proxy for darkness.")
            has_nightlights = False
    else:
        notes.append(
            "No night-lights raster — darkness uses a distance-to-built/road proxy (lower confidence). "
            "Drop a VIIRS/Falchi raster into data/nightlights to use measured light."
        )

    # --- build ctx_* per row --------------------------------------------------------
    ctx_cols = {k: [] for k in config.CONTEXT_KEYS}
    for _, row in gdf.iterrows():
        dist_river = _first_number(row, ("dist_os_river_m",))
        dist_awi = _first_number(row, ("dist_awi_ancwood_m",))
        dist_bld = _first_number(row, ("dist_building_m",))
        dist_str = _first_number(row, ("dist_structure_m",))
        dist_road = _first_number(row, ("dist_os_road_m",))
        dist_tree = _first_number(row, ("dist_ancient_tree_m",))
        phi_pct = _first_number(row, ("buf250_phi_broadleaved_woodland_pct",))
        le_pct = _first_number(row, ("buf250_le_broadleaf_pct",))
        net_degree = _first_number(row, ("net_degree_max",))
        net_comp = _first_number(row, ("net_component_size",))
        net_bc = _first_number(row, ("net_betweenness",))
        net_cl = _first_number(row, ("net_closeness",))
        net_pdeg = _first_number(row, ("net_planar_degree",))

        ctx_cols["ctx_water"].append(_mean_present([_invdist(dist_river, scales["water"])]))
        ctx_cols["ctx_woodland"].append(_mean_present([
            _scaled(phi_pct, 0.30),
            _scaled(le_pct, 0.30),
            _invdist(dist_awi, scales["woodland"]),
        ]))
        # Prefer the planar-network metrics (betweenness/closeness already 0-1); fall back to
        # endpoint-snap degree/component size when planar metrics are unavailable.
        ctx_cols["ctx_connectivity"].append(_mean_present([
            net_bc,
            net_cl,
            _scaled(None if net_pdeg is None else net_pdeg - 2.0, 4.0),
            _scaled(None if net_degree is None else net_degree - 1.0, 3.0),
            _scaled(None if net_comp is None else net_comp - 1.0, 10.0),
        ]))
        ctx_cols["ctx_roost"].append(_mean_present([
            _invdist(dist_bld, scales["roost_building"]),
            _invdist(dist_str, scales["roost_structure"]),
            _invdist(dist_tree, scales["roost_building"]),
            _invdist(dist_awi, scales["woodland"]),
        ]))
        if has_nightlights:
            nl_mean = _first_number(row, ("buf250_nightlight_mean",))
            darkness = None if nl_mean is None else 1.0 - _scaled(nl_mean, 20.0)
            ctx_cols["ctx_darkness"].append(_mean_present([darkness]))
        else:
            ctx_cols["ctx_darkness"].append(_mean_present([
                _fardist(dist_bld, scales["darkness_built"]),
                _fardist(dist_road, scales["darkness_built"]),
            ]))
        ctx_cols["ctx_road_severance"].append(_mean_present([_fardist(dist_road, scales["road"])]))

    for col, values in ctx_cols.items():
        gdf[col] = values
    return gdf, notes


def _add_distance(gdf, target, column, geom_kinds, notes):
    try:
        result = add_vector_distance(
            gdf, target, distance_column=column, geometry_kinds=geom_kinds
        )
        notes.extend(result.notes)
        return result.gdf
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Distance to {column} failed ({exc}).")
        gdf = gdf.copy()
        gdf[column] = None
        return gdf
