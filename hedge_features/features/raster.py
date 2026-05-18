from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..deps import require_numpy, require_rasterio
from ..exceptions import OptionalDependencyError


@dataclass(slots=True)
class RasterFeatureResult:
    gdf: Any
    notes: list[str]


def _stat_value(values, stat: str):
    np = require_numpy()
    if values.size == 0:
        return None
    stat = stat.lower()
    if stat == "mean":
        return float(np.mean(values))
    if stat == "median":
        return float(np.median(values))
    if stat == "min":
        return float(np.min(values))
    if stat == "max":
        return float(np.max(values))
    if stat == "p90":
        return float(np.percentile(values, 90))
    raise ValueError(f"Unsupported raster statistic: {stat}")


def _prepare_geoms_for_raster(buffers_geoseries, raster_crs):
    if buffers_geoseries.crs is not None and str(buffers_geoseries.crs) == str(raster_crs):
        return buffers_geoseries
    return buffers_geoseries.to_crs(raster_crs)


def _valid_array(masked_arr, nodata):
    np = require_numpy()
    arr = masked_arr
    if hasattr(arr, "compressed"):
        vals = arr.compressed()
    else:
        vals = np.asarray(arr).reshape(-1)
    if nodata is not None:
        vals = vals[vals != nodata]
    try:
        vals = vals[~np.isnan(vals)]
    except Exception:
        pass
    return vals


def add_raster_zonal_stats_in_buffers(
    hedges_gdf,
    raster_path: str | None,
    *,
    radii_m: list[int],
    stats: list[str],
    column_template: str,
    centroid_sample_column: str | None = None,
):
    gdf = hedges_gdf.copy()
    all_columns = [column_template.format(radius=r, stat=s) for r in radii_m for s in stats]
    if centroid_sample_column:
        all_columns.append(centroid_sample_column)

    if not raster_path:
        for col in all_columns:
            gdf[col] = None
        return RasterFeatureResult(gdf=gdf, notes=["Raster dataset path not provided; raster stats set to null."])

    try:
        rasterio = require_rasterio()
        from rasterio.mask import mask
        from shapely.geometry import mapping
    except OptionalDependencyError:
        for col in all_columns:
            gdf[col] = None
        return RasterFeatureResult(gdf=gdf, notes=["rasterio is not installed; raster stats set to null."])

    notes: list[str] = []
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        for radius in radii_m:
            colnames = [column_template.format(radius=radius, stat=s) for s in stats]
            for c in colnames:
                gdf[c] = None
            buffers = hedges_gdf.geometry.buffer(radius)
            buffers = hedges_gdf.geometry.__class__(buffers, crs=hedges_gdf.crs)
            buffers_raster = _prepare_geoms_for_raster(buffers, src.crs)
            for i, geom in enumerate(buffers_raster):
                try:
                    out, _ = mask(src, [mapping(geom)], crop=True, filled=False)
                    vals = _valid_array(out[0], nodata)
                    for stat, col in zip(stats, colnames):
                        gdf.iat[i, gdf.columns.get_loc(col)] = _stat_value(vals, stat)
                except ValueError:
                    # Often means no overlap with raster extent.
                    continue
                except Exception as exc:
                    if not notes:
                        notes.append(f"Raster zonal stats encountered errors (first: {exc})")
                    continue

        if centroid_sample_column:
            gdf[centroid_sample_column] = None
            centroids = hedges_gdf.geometry.centroid
            centroids = hedges_gdf.geometry.__class__(centroids, crs=hedges_gdf.crs).to_crs(src.crs)
            coords = [(pt.x, pt.y) if pt is not None and not pt.is_empty else None for pt in centroids]
            for i, coord in enumerate(coords):
                if coord is None:
                    continue
                try:
                    sample = next(src.sample([coord]))
                    value = sample[0]
                    if nodata is not None and value == nodata:
                        continue
                    gdf.iat[i, gdf.columns.get_loc(centroid_sample_column)] = float(value)
                except Exception:
                    continue

    return RasterFeatureResult(gdf=gdf, notes=notes)


def _landscape_metric_column(
    *,
    radius: int,
    class_name: str,
    metric_name: str,
    templates: dict[str, str],
) -> str | None:
    template = templates.get(metric_name)
    if not template:
        return None
    return template.format(radius=radius, class_name=class_name, metric=metric_name)


def _landscape_metric_columns(
    *,
    radii_m: list[int],
    class_names: list[str],
    metrics: list[str],
    templates: dict[str, str],
) -> list[str]:
    cols: list[str] = []
    for radius in radii_m:
        for class_name in class_names:
            for metric_name in metrics:
                col = _landscape_metric_column(
                    radius=radius,
                    class_name=class_name,
                    metric_name=metric_name,
                    templates=templates,
                )
                if col:
                    cols.append(col)
    return cols


def _categorical_landscape_metrics(
    masked_arr,
    *,
    nodata,
    class_codes: list[int],
    pixel_width_m: float,
    pixel_height_m: float,
) -> dict[str, float]:
    """Compute lightweight patch/edge metrics for a class inside a masked raster window."""
    np = require_numpy()
    arr = np.asarray(masked_arr)
    if arr.ndim != 2 or arr.size == 0:
        return {
            "edge_density_m_per_ha": 0.0,
            "largest_patch_index": 0.0,
            "core_area_pct": 0.0,
        }

    if hasattr(masked_arr, "mask"):
        valid_mask = ~np.asarray(masked_arr.mask, dtype=bool)
        if valid_mask.shape == ():
            valid_mask = np.full(arr.shape, bool(valid_mask), dtype=bool)
    else:
        valid_mask = np.ones(arr.shape, dtype=bool)
    if nodata is not None:
        valid_mask &= arr != nodata
    try:
        valid_mask &= ~np.isnan(arr)
    except Exception:
        pass

    valid_pixels = int(valid_mask.sum())
    if valid_pixels == 0:
        return {
            "edge_density_m_per_ha": 0.0,
            "largest_patch_index": 0.0,
            "core_area_pct": 0.0,
        }

    class_mask = valid_mask & np.isin(arr.astype(int), np.array(class_codes, dtype=int))
    class_pixels = int(class_mask.sum())
    if class_pixels == 0:
        return {
            "edge_density_m_per_ha": 0.0,
            "largest_patch_index": 0.0,
            "core_area_pct": 0.0,
        }

    edge_length_m = _class_edge_length_m(class_mask, valid_mask, pixel_width_m, pixel_height_m)
    valid_area_ha = (valid_pixels * float(pixel_width_m) * float(pixel_height_m)) / 10000.0
    largest_patch_pixels = _largest_patch_size(class_mask)
    core_pixels = _core_pixels_8_connected(class_mask)
    return {
        "edge_density_m_per_ha": round(float(edge_length_m / valid_area_ha), 6) if valid_area_ha > 0 else 0.0,
        "largest_patch_index": round(float(largest_patch_pixels / valid_pixels), 6),
        "core_area_pct": round(float(core_pixels / valid_pixels), 6),
    }


def _class_edge_length_m(class_mask, valid_mask, pixel_width_m: float, pixel_height_m: float) -> float:
    np = require_numpy()
    padded_class = np.pad(class_mask, 1, mode="constant", constant_values=False)
    padded_valid = np.pad(valid_mask, 1, mode="constant", constant_values=False)
    center = padded_class[1:-1, 1:-1]
    up_edge = center & (~padded_class[:-2, 1:-1] | ~padded_valid[:-2, 1:-1])
    down_edge = center & (~padded_class[2:, 1:-1] | ~padded_valid[2:, 1:-1])
    left_edge = center & (~padded_class[1:-1, :-2] | ~padded_valid[1:-1, :-2])
    right_edge = center & (~padded_class[1:-1, 2:] | ~padded_valid[1:-1, 2:])
    return float((up_edge.sum() + down_edge.sum()) * pixel_width_m + (left_edge.sum() + right_edge.sum()) * pixel_height_m)


def _largest_patch_size(class_mask) -> int:
    np = require_numpy()
    visited = np.zeros(class_mask.shape, dtype=bool)
    largest = 0
    rows, cols = class_mask.shape
    for row in range(rows):
        for col in range(cols):
            if visited[row, col] or not class_mask[row, col]:
                continue
            size = 0
            stack = [(row, col)]
            visited[row, col] = True
            while stack:
                r, c = stack.pop()
                size += 1
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and class_mask[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            largest = max(largest, size)
    return int(largest)


def _core_pixels_8_connected(class_mask) -> int:
    np = require_numpy()
    if min(class_mask.shape) < 3:
        return 0
    core = class_mask.copy()
    for row_shift in (-1, 0, 1):
        for col_shift in (-1, 0, 1):
            if row_shift == 0 and col_shift == 0:
                continue
            shifted = np.zeros(class_mask.shape, dtype=bool)
            src_rows = slice(max(0, -row_shift), class_mask.shape[0] - max(0, row_shift))
            dst_rows = slice(max(0, row_shift), class_mask.shape[0] - max(0, -row_shift))
            src_cols = slice(max(0, -col_shift), class_mask.shape[1] - max(0, col_shift))
            dst_cols = slice(max(0, col_shift), class_mask.shape[1] - max(0, -col_shift))
            shifted[dst_rows, dst_cols] = class_mask[src_rows, src_cols]
            core &= shifted
    return int(core.sum())


def add_raster_categorical_proportions_in_buffers(
    hedges_gdf,
    raster_path: str | None,
    *,
    radii_m: list[int],
    class_map: dict[str, list[int]],
    column_template: str,
    patch_richness_column_template: str | None = None,
    landscape_class_names: list[str] | None = None,
    landscape_metrics: list[str] | None = None,
    landscape_column_templates: dict[str, str] | None = None,
):
    gdf = hedges_gdf.copy()
    all_cols = [
        column_template.format(radius=radius, class_name=class_name)
        for radius in radii_m
        for class_name in class_map
    ]
    if patch_richness_column_template:
        all_cols.extend([patch_richness_column_template.format(radius=radius) for radius in radii_m])
    landscape_class_names = [str(c) for c in (landscape_class_names or []) if str(c) in class_map]
    landscape_metrics = [str(m) for m in (landscape_metrics or [])]
    landscape_column_templates = dict(landscape_column_templates or {})
    landscape_cols = _landscape_metric_columns(
        radii_m=radii_m,
        class_names=landscape_class_names,
        metrics=landscape_metrics,
        templates=landscape_column_templates,
    )
    all_cols.extend(landscape_cols)

    if not raster_path:
        for col in all_cols:
            gdf[col] = None
        return RasterFeatureResult(gdf=gdf, notes=["Raster dataset path not provided; categorical proportions set to null."])

    try:
        rasterio = require_rasterio()
        np = require_numpy()
        from rasterio.mask import mask
        from shapely.geometry import mapping
    except OptionalDependencyError:
        for col in all_cols:
            gdf[col] = None
        return RasterFeatureResult(gdf=gdf, notes=["rasterio is not installed; categorical raster metrics set to null."])

    notes: list[str] = []
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        for radius in radii_m:
            buffers = hedges_gdf.geometry.buffer(radius)
            buffers = hedges_gdf.geometry.__class__(buffers, crs=hedges_gdf.crs)
            buffers_raster = _prepare_geoms_for_raster(buffers, src.crs)
            radius_cols = {
                class_name: column_template.format(radius=radius, class_name=class_name)
                for class_name in class_map
            }
            for col in radius_cols.values():
                gdf[col] = None
            richness_col = (
                patch_richness_column_template.format(radius=radius)
                if patch_richness_column_template
                else None
            )
            if richness_col:
                gdf[richness_col] = None
            radius_landscape_cols = _landscape_metric_columns(
                radii_m=[radius],
                class_names=landscape_class_names,
                metrics=landscape_metrics,
                templates=landscape_column_templates,
            )
            for col in radius_landscape_cols:
                gdf[col] = None

            for i, geom in enumerate(buffers_raster):
                try:
                    out, out_transform = mask(src, [mapping(geom)], crop=True, filled=False)
                    band = out[0]
                    vals = _valid_array(band, nodata)
                    if vals.size == 0:
                        for col in radius_cols.values():
                            gdf.iat[i, gdf.columns.get_loc(col)] = 0.0
                        if richness_col:
                            gdf.iat[i, gdf.columns.get_loc(richness_col)] = 0
                        for col in radius_landscape_cols:
                            gdf.iat[i, gdf.columns.get_loc(col)] = 0.0
                        continue

                    vals_int = vals.astype(int)
                    total = vals_int.size
                    for class_name, codes in class_map.items():
                        pct = float(np.isin(vals_int, np.array(codes, dtype=int)).sum()) / float(total)
                        col = radius_cols[class_name]
                        gdf.iat[i, gdf.columns.get_loc(col)] = pct
                    if richness_col:
                        gdf.iat[i, gdf.columns.get_loc(richness_col)] = int(np.unique(vals_int).size)
                    if landscape_class_names and landscape_metrics:
                        pixel_width_m = abs(float(out_transform.a)) or 1.0
                        pixel_height_m = abs(float(out_transform.e)) or pixel_width_m
                        for class_name in landscape_class_names:
                            metrics = _categorical_landscape_metrics(
                                band,
                                nodata=nodata,
                                class_codes=class_map[class_name],
                                pixel_width_m=pixel_width_m,
                                pixel_height_m=pixel_height_m,
                            )
                            for metric_name in landscape_metrics:
                                col = _landscape_metric_column(
                                    radius=radius,
                                    class_name=class_name,
                                    metric_name=metric_name,
                                    templates=landscape_column_templates,
                                )
                                if col:
                                    gdf.iat[i, gdf.columns.get_loc(col)] = metrics.get(metric_name)
                except ValueError:
                    for col in radius_cols.values():
                        gdf.iat[i, gdf.columns.get_loc(col)] = None
                    if richness_col:
                        gdf.iat[i, gdf.columns.get_loc(richness_col)] = None
                    for col in radius_landscape_cols:
                        gdf.iat[i, gdf.columns.get_loc(col)] = None
                except Exception as exc:
                    if not notes:
                        notes.append(f"Raster categorical extraction encountered errors (first: {exc})")
                    continue
    return RasterFeatureResult(gdf=gdf, notes=notes)

