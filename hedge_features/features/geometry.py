from __future__ import annotations

import math
from typing import Iterable


def _flatten_coords(geom) -> list[tuple[float, float]]:
    if geom is None or geom.is_empty:
        return []
    geom_type = geom.geom_type
    if geom_type == "LineString":
        return [(float(x), float(y)) for x, y, *_ in geom.coords]
    if geom_type == "MultiLineString":
        coords: list[tuple[float, float]] = []
        for part in geom.geoms:
            coords.extend((float(x), float(y)) for x, y, *_ in part.coords)
        return coords
    return []


def _endpoint_distance_and_bearing(geom) -> tuple[float | None, float | None]:
    coords = _flatten_coords(geom)
    if len(coords) < 2:
        return None, None
    (x1, y1), (x2, y2) = coords[0], coords[-1]
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist == 0:
        return 0.0, None
    # Bearing clockwise from North in degrees.
    bearing = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
    return dist, bearing


def add_geometry_metrics(gdf):
    gdf = gdf.copy()
    lengths = gdf.geometry.length.astype(float)
    gdf["geom_length_m"] = lengths

    endpoint_distances: list[float | None] = []
    bearings: list[float | None] = []
    for geom in gdf.geometry:
        dist, bearing = _endpoint_distance_and_bearing(geom)
        endpoint_distances.append(dist)
        bearings.append(bearing)
    gdf["geom_endpoint_dist_m"] = endpoint_distances
    gdf["geom_bearing_deg"] = bearings
    gdf["geom_orientation_deg"] = [
        None if b is None else (b % 180.0) for b in bearings
    ]

    sinuosity: list[float | None] = []
    for length, endpoint_dist in zip(lengths, endpoint_distances):
        if endpoint_dist in (None, 0):
            sinuosity.append(None)
        else:
            sinuosity.append(float(length) / float(endpoint_dist))
    gdf["geom_sinuosity"] = sinuosity

    gdf["geom_centroid_x"] = gdf.geometry.centroid.x
    gdf["geom_centroid_y"] = gdf.geometry.centroid.y
    return gdf


def orientation_dispersion(angles_deg: Iterable[float | None]) -> float | None:
    values = [a for a in angles_deg if a is not None]
    if not values:
        return None
    radians = [math.radians(v * 2.0) for v in values]  # axial data
    c = sum(math.cos(r) for r in radians) / len(radians)
    s = sum(math.sin(r) for r in radians) / len(radians)
    r_bar = math.sqrt(c * c + s * s)
    return 1.0 - r_bar

