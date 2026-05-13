from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class VectorFeatureResult:
    gdf: Any
    notes: list[str]


def _geom_types_match(target_gdf, allowed_types: list[str] | None) -> bool:
    if not allowed_types:
        return True
    allowed = {t.lower() for t in allowed_types}
    found = {str(t).lower() for t in target_gdf.geometry.geom_type.dropna().unique()}
    return bool(found & allowed)


def _subset_by_bbox(target_gdf, geom):
    if target_gdf.empty:
        return target_gdf
    try:
        sindex = target_gdf.sindex
    except Exception:
        sindex = None
    if sindex is None:
        return target_gdf
    minx, miny, maxx, maxy = geom.bounds
    idx = list(sindex.intersection((minx, miny, maxx, maxy)))
    if not idx:
        return target_gdf.iloc[0:0]
    return target_gdf.iloc[idx]


def add_vector_distance(
    hedges_gdf,
    target_gdf,
    *,
    distance_column: str,
    geometry_kinds: list[str] | None = None,
    coverage_flag_column: str | None = None,
):
    gdf = hedges_gdf.copy()
    if target_gdf is None or target_gdf.empty:
        gdf[distance_column] = None
        if coverage_flag_column:
            gdf[coverage_flag_column] = 0
        return VectorFeatureResult(gdf=gdf, notes=[f"No target data for {distance_column}; set to null."])

    if not _geom_types_match(target_gdf, geometry_kinds):
        note = f"Target geometry types do not match expected {geometry_kinds}; distances still computed."
    else:
        note = None
    union_geom = target_gdf.geometry.unary_union
    gdf[distance_column] = gdf.geometry.distance(union_geom)
    if coverage_flag_column:
        # Coarse coverage: use dataset bounds intersection with each segment.
        extent_poly = _bbox_polygon_from_bounds(target_gdf.total_bounds)
        if extent_poly is None:
            gdf[coverage_flag_column] = 1
        else:
            gdf[coverage_flag_column] = gdf.geometry.intersects(extent_poly).astype(int)
    notes = [note] if note else []
    return VectorFeatureResult(gdf=gdf, notes=notes)


def _bbox_polygon_from_bounds(bounds):
    try:
        from shapely.geometry import box
    except Exception:
        return None
    minx, miny, maxx, maxy = bounds
    return box(minx, miny, maxx, maxy)


def add_vector_line_density_in_buffers(
    hedges_gdf,
    target_gdf,
    *,
    radii_m: list[int],
    density_column_template: str,
    output_metric: str = "m_per_ha",
):
    gdf = hedges_gdf.copy()
    if target_gdf is None or target_gdf.empty:
        for radius in radii_m:
            gdf[density_column_template.format(radius=radius)] = None
        return VectorFeatureResult(gdf=gdf, notes=["No target line dataset; density columns set to null."])

    for radius in radii_m:
        buffers = gdf.geometry.buffer(radius)
        values: list[float | None] = []
        for buf in buffers:
            candidates = _subset_by_bbox(target_gdf, buf)
            if candidates.empty:
                values.append(0.0)
                continue
            clipped = candidates.geometry.intersection(buf)
            total_len = float(clipped.length.sum())
            area_m2 = float(buf.area)
            if area_m2 <= 0:
                values.append(None)
                continue
            if output_metric == "m_per_ha":
                values.append(total_len / (area_m2 / 10000.0))
            elif output_metric == "m_per_km2":
                values.append(total_len / (area_m2 / 1_000_000.0))
            else:
                values.append(total_len)
        gdf[density_column_template.format(radius=radius)] = values
    return VectorFeatureResult(gdf=gdf, notes=[])


def add_vector_feature_count_in_buffers(
    hedges_gdf,
    target_gdf,
    *,
    radii_m: list[int],
    count_column_template: str,
    density_column_template: str | None = None,
):
    gdf = hedges_gdf.copy()
    if target_gdf is None or target_gdf.empty:
        for radius in radii_m:
            gdf[count_column_template.format(radius=radius)] = 0
            if density_column_template:
                gdf[density_column_template.format(radius=radius)] = 0.0
        return VectorFeatureResult(gdf=gdf, notes=["No target dataset; count/density columns set to zero."])

    for radius in radii_m:
        count_col = count_column_template.format(radius=radius)
        dens_col = density_column_template.format(radius=radius) if density_column_template else None
        buffers = gdf.geometry.buffer(radius)
        counts: list[int] = []
        densities: list[float | None] = []
        for buf in buffers:
            candidates = _subset_by_bbox(target_gdf, buf)
            if candidates.empty:
                counts.append(0)
                if dens_col:
                    densities.append(0.0)
                continue
            count = int(candidates.geometry.intersects(buf).sum())
            counts.append(count)
            if dens_col:
                area_m2 = float(buf.area)
                densities.append((count / (area_m2 / 10000.0)) if area_m2 > 0 else None)
        gdf[count_col] = counts
        if dens_col:
            gdf[dens_col] = densities
    return VectorFeatureResult(gdf=gdf, notes=[])


def _compile_class_masks(poly_gdf, class_field: str, selected_classes: dict[str, dict[str, Any]]):
    source = poly_gdf[class_field].fillna("").astype(str).str.lower()
    masks: dict[str, Any] = {}
    for class_name, rule in (selected_classes or {}).items():
        contains_any = [s.lower() for s in (rule.get("contains_any") or [])]
        if contains_any:
            mask = source.apply(lambda v: any(token in v for token in contains_any))
        else:
            mask = source == source  # all rows
        masks[class_name] = mask
    return masks


def add_vector_polygon_composition_in_buffers(
    hedges_gdf,
    target_gdf,
    *,
    radii_m: list[int],
    class_field: str,
    selected_classes: dict[str, dict[str, Any]],
    column_template: str,
    coverage_flag_column: str | None = None,
):
    gdf = hedges_gdf.copy()
    if target_gdf is None or target_gdf.empty:
        for radius in radii_m:
            for class_name in (selected_classes or {}):
                gdf[column_template.format(radius=radius, class_name=class_name)] = None
        if coverage_flag_column:
            gdf[coverage_flag_column] = 0
        return VectorFeatureResult(gdf=gdf, notes=["No target polygon dataset; composition columns set to null."])
    if class_field not in target_gdf.columns:
        raise ValueError(f"Class field '{class_field}' not found in target dataset.")

    class_masks = _compile_class_masks(target_gdf, class_field, selected_classes)
    extent_poly = _bbox_polygon_from_bounds(target_gdf.total_bounds)
    if coverage_flag_column and extent_poly is not None:
        gdf[coverage_flag_column] = gdf.geometry.intersects(extent_poly).astype(int)

    for radius in radii_m:
        buffers = gdf.geometry.buffer(radius)
        buffer_areas = buffers.area
        for class_name, mask in class_masks.items():
            values: list[float | None] = []
            subset = target_gdf[mask]
            for buf, area in zip(buffers, buffer_areas):
                if area <= 0:
                    values.append(None)
                    continue
                candidates = _subset_by_bbox(subset, buf)
                if candidates.empty:
                    values.append(0.0)
                    continue
                inter_areas = candidates.geometry.intersection(buf).area
                values.append(float(inter_areas.sum()) / float(area))
            gdf[column_template.format(radius=radius, class_name=class_name)] = values
    return VectorFeatureResult(gdf=gdf, notes=[])
