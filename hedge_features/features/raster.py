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


def add_raster_categorical_proportions_in_buffers(
    hedges_gdf,
    raster_path: str | None,
    *,
    radii_m: list[int],
    class_map: dict[str, list[int]],
    column_template: str,
    patch_richness_column_template: str | None = None,
):
    gdf = hedges_gdf.copy()
    all_cols = [
        column_template.format(radius=radius, class_name=class_name)
        for radius in radii_m
        for class_name in class_map
    ]
    if patch_richness_column_template:
        all_cols.extend([patch_richness_column_template.format(radius=radius) for radius in radii_m])

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

            for i, geom in enumerate(buffers_raster):
                try:
                    out, _ = mask(src, [mapping(geom)], crop=True, filled=False)
                    vals = _valid_array(out[0], nodata)
                    if vals.size == 0:
                        for col in radius_cols.values():
                            gdf.iat[i, gdf.columns.get_loc(col)] = 0.0
                        if richness_col:
                            gdf.iat[i, gdf.columns.get_loc(richness_col)] = 0
                        continue

                    vals_int = vals.astype(int)
                    total = vals_int.size
                    for class_name, codes in class_map.items():
                        pct = float(np.isin(vals_int, np.array(codes, dtype=int)).sum()) / float(total)
                        col = radius_cols[class_name]
                        gdf.iat[i, gdf.columns.get_loc(col)] = pct
                    if richness_col:
                        gdf.iat[i, gdf.columns.get_loc(richness_col)] = int(np.unique(vals_int).size)
                except ValueError:
                    for col in radius_cols.values():
                        gdf.iat[i, gdf.columns.get_loc(col)] = None
                    if richness_col:
                        gdf.iat[i, gdf.columns.get_loc(richness_col)] = None
                except Exception as exc:
                    if not notes:
                        notes.append(f"Raster categorical extraction encountered errors (first: {exc})")
                    continue
    return RasterFeatureResult(gdf=gdf, notes=notes)

