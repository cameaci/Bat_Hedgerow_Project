from __future__ import annotations

from collections import defaultdict
from math import isnan
from typing import Any

from ..deps import require_networkx
from ..exceptions import OptionalDependencyError


def _endpoint_pair(geom) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        coords = list(geom.coords)
        if len(coords) < 2:
            return None
        return (float(coords[0][0]), float(coords[0][1])), (float(coords[-1][0]), float(coords[-1][1]))
    if geom.geom_type == "MultiLineString":
        parts = [list(part.coords) for part in geom.geoms if len(part.coords) >= 2]
        if not parts:
            return None
        first = parts[0][0]
        last = parts[-1][-1]
        return (float(first[0]), float(first[1])), (float(last[0]), float(last[1]))
    return None


def _snap_xy(xy: tuple[float, float], tolerance_m: float) -> tuple[int, int]:
    if tolerance_m <= 0:
        return (round(xy[0]), round(xy[1]))
    return (round(xy[0] / tolerance_m), round(xy[1] / tolerance_m))


def add_network_metrics(gdf, tolerance_m: float = 1.0):
    try:
        nx = require_networkx()
    except OptionalDependencyError:
        gdf = gdf.copy()
        for col in (
            "net_degree_start",
            "net_degree_end",
            "net_degree_max",
            "net_component_id",
            "net_component_size",
            "net_betweenness",
            "net_closeness",
        ):
            gdf[col] = None
        return gdf, ["networkx is not installed; network metrics set to null."]

    G = nx.MultiGraph()
    edge_rows: list[tuple[Any, Any, int, int]] = []  # u, v, key, row_position
    endpoint_nodes: list[tuple[Any, Any] | None] = []

    for row_pos, geom in enumerate(gdf.geometry):
        pair = _endpoint_pair(geom)
        if pair is None:
            endpoint_nodes.append(None)
            continue
        u = _snap_xy(pair[0], tolerance_m)
        v = _snap_xy(pair[1], tolerance_m)
        key = G.add_edge(u, v, row_pos=row_pos)
        edge_rows.append((u, v, key, row_pos))
        endpoint_nodes.append((u, v))

    gdf = gdf.copy()
    gdf["net_degree_start"] = None
    gdf["net_degree_end"] = None
    gdf["net_degree_max"] = None
    gdf["net_component_id"] = None
    gdf["net_component_size"] = None
    gdf["net_betweenness"] = None
    gdf["net_closeness"] = None

    component_by_node: dict[Any, int] = {}
    component_size_by_node: dict[Any, int] = {}
    for comp_id, component_nodes in enumerate(nx.connected_components(G)):
        size = len(component_nodes)
        for node in component_nodes:
            component_by_node[node] = comp_id
            component_size_by_node[node] = size

    for row_pos, nodes in enumerate(endpoint_nodes):
        if nodes is None:
            continue
        u, v = nodes
        deg_u = int(G.degree(u))
        deg_v = int(G.degree(v))
        gdf.iat[row_pos, gdf.columns.get_loc("net_degree_start")] = deg_u
        gdf.iat[row_pos, gdf.columns.get_loc("net_degree_end")] = deg_v
        gdf.iat[row_pos, gdf.columns.get_loc("net_degree_max")] = max(deg_u, deg_v)
        gdf.iat[row_pos, gdf.columns.get_loc("net_component_id")] = component_by_node.get(u)
        gdf.iat[row_pos, gdf.columns.get_loc("net_component_size")] = component_size_by_node.get(u)

    if G.number_of_edges() == 0:
        return gdf, []

    # Segment centrality via line graph (nodes are original edges).
    try:
        LG = nx.line_graph(G)
        edge_bet = nx.betweenness_centrality(LG, normalized=True)
        edge_close = nx.closeness_centrality(LG) if LG.number_of_nodes() else {}
    except Exception as exc:
        return gdf, [f"Network centrality calculation failed: {exc}"]

    # Map line-graph edge nodes back to row positions.
    for lg_node, value in edge_bet.items():
        row_pos = _line_graph_node_to_rowpos(G, lg_node)
        if row_pos is not None:
            gdf.iat[row_pos, gdf.columns.get_loc("net_betweenness")] = float(value)
    for lg_node, value in edge_close.items():
        row_pos = _line_graph_node_to_rowpos(G, lg_node)
        if row_pos is not None:
            gdf.iat[row_pos, gdf.columns.get_loc("net_closeness")] = float(value)

    return gdf, []


def _line_graph_node_to_rowpos(G, lg_node):
    """Support Graph and MultiGraph line_graph node encodings."""
    try:
        if isinstance(lg_node, tuple) and len(lg_node) == 3:
            u, v, key = lg_node
            return G.edges[u, v, key].get("row_pos")
        if isinstance(lg_node, tuple) and len(lg_node) == 2:
            u, v = lg_node
            data = G.get_edge_data(u, v)
            if not data:
                return None
            # Pick first edge for non-multigraph or ambiguous case.
            if isinstance(data, dict) and "row_pos" in data:
                return data.get("row_pos")
            if isinstance(data, dict):
                for _, edge_data in data.items():
                    if isinstance(edge_data, dict) and "row_pos" in edge_data:
                        return edge_data["row_pos"]
    except Exception:
        return None
    return None

