"""Planar hedgerow-network connectivity.

The v1 connectivity used endpoint snapping only, so hedges that meet mid-span were not
linked. Here we *node* the whole network at every intersection (shapely ``unary_union``),
build a graph of the resulting segments, compute edge betweenness and node closeness, and
aggregate them back to each input hedgerow (by ``hf_uid``). Higher betweenness = the hedge
is a more critical commuting link in the network.
"""

from __future__ import annotations

from hedge_features.deps import require_networkx


def add_connectivity(gdf, *, id_col: str = "hf_uid", snap_tol_m: float = 1.0):
    """Add ``net_betweenness``, ``net_closeness``, ``net_planar_degree``. Returns ``(gdf, notes)``."""
    gdf = gdf.copy()
    for col in ("net_betweenness", "net_closeness", "net_planar_degree"):
        if col not in gdf.columns:
            gdf[col] = None

    try:
        nx = require_networkx()
        from shapely.ops import unary_union
    except Exception as exc:  # noqa: BLE001
        return gdf, [f"Connectivity skipped: dependency unavailable ({exc})."]

    geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
    if len(geoms) < 2:
        return gdf, ["Connectivity skipped: fewer than two hedgerows."]

    segments = _as_lines(unary_union(geoms))
    if not segments:
        return gdf, ["Connectivity skipped: no segments after noding."]

    graph = nx.Graph()
    seg_endpoints: list[tuple] = []
    for idx, seg in enumerate(segments):
        coords = list(seg.coords)
        if len(coords) < 2:
            seg_endpoints.append(None)
            continue
        u = _snap(coords[0], snap_tol_m)
        v = _snap(coords[-1], snap_tol_m)
        graph.add_edge(u, v, length=max(float(seg.length), 1e-6))
        seg_endpoints.append((u, v))

    edge_bc = nx.edge_betweenness_centrality(graph, weight="length") if graph.number_of_edges() else {}
    closeness = nx.closeness_centrality(graph, distance="length") if graph.number_of_nodes() else {}

    parents = _assign_parents(segments, gdf, id_col)

    agg_bc: dict = {}
    agg_cl: dict = {}
    agg_dg: dict = {}
    for idx, ends in enumerate(seg_endpoints):
        if ends is None:
            continue
        pid = parents.get(idx)
        if pid is None:
            continue
        u, v = ends
        bc = edge_bc.get((u, v), edge_bc.get((v, u), 0.0))
        cl = max(closeness.get(u, 0.0), closeness.get(v, 0.0))
        dg = max(int(graph.degree(u)), int(graph.degree(v)))
        agg_bc[pid] = max(agg_bc.get(pid, 0.0), float(bc))
        agg_cl[pid] = max(agg_cl.get(pid, 0.0), float(cl))
        agg_dg[pid] = max(agg_dg.get(pid, 0), dg)

    for i, pid in enumerate(gdf[id_col]):
        if pid in agg_bc:
            gdf.iat[i, gdf.columns.get_loc("net_betweenness")] = agg_bc[pid]
            gdf.iat[i, gdf.columns.get_loc("net_closeness")] = agg_cl[pid]
            gdf.iat[i, gdf.columns.get_loc("net_planar_degree")] = agg_dg[pid]

    return gdf, ["Planar connectivity computed from the noded hedgerow network."]


def _as_lines(geom) -> list:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    if geom.geom_type in ("MultiLineString", "GeometryCollection"):
        return [g for g in geom.geoms if getattr(g, "geom_type", "") == "LineString" and not g.is_empty]
    return []


def _snap(xy, tol: float) -> tuple[int, int]:
    if tol <= 0:
        return (round(xy[0]), round(xy[1]))
    return (round(xy[0] / tol), round(xy[1] / tol))


def _assign_parents(segments, gdf, id_col) -> dict[int, object]:
    """Map each noded segment to the input hedgerow it belongs to (nearest by midpoint)."""
    try:
        sindex = gdf.sindex
    except Exception:  # noqa: BLE001
        sindex = None
    ids = list(gdf[id_col])
    geoms = list(gdf.geometry)
    out: dict[int, object] = {}
    for idx, seg in enumerate(segments):
        mid = seg.interpolate(0.5, normalized=True)
        candidates = range(len(geoms))
        if sindex is not None:
            hits = list(sindex.intersection(seg.buffer(2.0).bounds))
            if hits:
                candidates = hits
        best_id = None
        best_dist = float("inf")
        for j in candidates:
            g = geoms[j]
            if g is None or g.is_empty:
                continue
            d = g.distance(mid)
            if d < best_dist:
                best_dist = d
                best_id = ids[j]
        out[idx] = best_id
    return out
