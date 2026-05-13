from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..deps import require_numpy, require_rasterio
from ..exceptions import OptionalDependencyError


@dataclass(slots=True)
class LidarStructureFeatureResult:
    gdf: Any
    notes: list[str]


def add_lidar_hedgerow_structure_features(
    hedges_gdf,
    *,
    dtm_path: str | None,
    dsm_path: str | None,
    height_buffer_m: float = 5.0,
    continuity_buffer_m: float = 10.0,
) -> LidarStructureFeatureResult:
    gdf = hedges_gdf.copy()
    notes: list[str] = []
    output_columns = (
        "hedge_struct_height_mean_5m",
        "hedge_struct_height_p90_5m",
        "hedge_struct_canopy_continuity_10m",
        "hedge_struct_gap_fraction_10m",
        "hedge_struct_tree_standard_pct_10m",
        "hedge_struct_width_proxy_m",
        "hedge_struct_status",
    )
    for column in output_columns:
        if column not in gdf.columns:
            gdf[column] = None

    if not dtm_path or not dsm_path:
        gdf["hedge_struct_status"] = "source_missing"
        notes.append("LiDAR hedgerow structure features skipped: DTM or DSM path was not provided.")
        return LidarStructureFeatureResult(gdf=gdf, notes=notes)

    try:
        rasterio = require_rasterio()
        np = require_numpy()
        from rasterio.mask import mask
        from rasterio.warp import reproject, Resampling
        from shapely.geometry import mapping
    except OptionalDependencyError:
        gdf["hedge_struct_status"] = "dependency_missing"
        notes.append("LiDAR hedgerow structure features skipped: rasterio/numpy are unavailable.")
        return LidarStructureFeatureResult(gdf=gdf, notes=notes)

    with rasterio.open(dtm_path) as dtm_src, rasterio.open(dsm_path) as dsm_src:
        if str(dtm_src.crs) != str(dsm_src.crs):
            raise ValueError("LiDAR DTM and DSM must share the same CRS.")
        height_buffers = hedges_gdf.geometry.buffer(float(height_buffer_m))
        continuity_buffers = hedges_gdf.geometry.buffer(float(continuity_buffer_m))
        height_buffers = hedges_gdf.geometry.__class__(height_buffers, crs=hedges_gdf.crs).to_crs(dtm_src.crs)
        continuity_buffers = hedges_gdf.geometry.__class__(continuity_buffers, crs=hedges_gdf.crs).to_crs(dtm_src.crs)
        hedges_raster = hedges_gdf.to_crs(dtm_src.crs)

        for row_pos, (geom5, geom10, hedge_geom) in enumerate(zip(height_buffers, continuity_buffers, hedges_raster.geometry)):
            if hedge_geom is None or hedge_geom.is_empty:
                gdf.iat[row_pos, gdf.columns.get_loc("hedge_struct_status")] = "missing_geometry"
                continue
            try:
                chm_5m = _masked_canopy_height(
                    dsm_src=dsm_src,
                    dtm_src=dtm_src,
                    geom=geom5,
                    mask_fn=mask,
                    reproject_fn=reproject,
                    resampling=Resampling,
                    np=np,
                )
                chm_10m = _masked_canopy_height(
                    dsm_src=dsm_src,
                    dtm_src=dtm_src,
                    geom=geom10,
                    mask_fn=mask,
                    reproject_fn=reproject,
                    resampling=Resampling,
                    np=np,
                )
            except ValueError:
                gdf.iat[row_pos, gdf.columns.get_loc("hedge_struct_status")] = "outside_coverage"
                continue
            except Exception as exc:
                if not notes:
                    notes.append(f"LiDAR hedgerow structure features encountered errors (first: {exc})")
                gdf.iat[row_pos, gdf.columns.get_loc("hedge_struct_status")] = "error"
                continue

            if chm_5m.size == 0 or chm_10m.size == 0:
                gdf.iat[row_pos, gdf.columns.get_loc("hedge_struct_status")] = "outside_coverage"
                continue

            height_mean = float(chm_5m.mean()) if chm_5m.size else None
            height_p90 = float(np.percentile(chm_5m, 90)) if chm_5m.size else None
            continuity = float((chm_10m >= 1.5).mean()) if chm_10m.size else None
            gap_fraction = float((chm_10m <= 0.5).mean()) if chm_10m.size else None
            tree_standard_pct = float((chm_10m >= 6.0).mean()) if chm_10m.size else None
            width_proxy = _width_proxy_m(
                hedge_geom=hedge_geom,
                canopy_height_values=chm_10m,
                pixel_area_m2=abs(float(dtm_src.transform.a) * float(dtm_src.transform.e)),
            )

            gdf.iat[row_pos, gdf.columns.get_loc("hedge_struct_height_mean_5m")] = height_mean
            gdf.iat[row_pos, gdf.columns.get_loc("hedge_struct_height_p90_5m")] = height_p90
            gdf.iat[row_pos, gdf.columns.get_loc("hedge_struct_canopy_continuity_10m")] = continuity
            gdf.iat[row_pos, gdf.columns.get_loc("hedge_struct_gap_fraction_10m")] = gap_fraction
            gdf.iat[row_pos, gdf.columns.get_loc("hedge_struct_tree_standard_pct_10m")] = tree_standard_pct
            gdf.iat[row_pos, gdf.columns.get_loc("hedge_struct_width_proxy_m")] = width_proxy
            gdf.iat[row_pos, gdf.columns.get_loc("hedge_struct_status")] = "ok"

    notes.append(
        "LiDAR hedgerow structure features computed from DSM-DTM canopy height proxies within 5 m and 10 m hedge corridors."
    )
    return LidarStructureFeatureResult(gdf=gdf, notes=notes)


def _masked_canopy_height(*, dsm_src, dtm_src, geom, mask_fn, reproject_fn, resampling, np):
    from shapely.geometry import mapping

    dsm_arr, dsm_transform = mask_fn(dsm_src, [mapping(geom)], crop=True, filled=False)
    dtm_arr, dtm_transform = mask_fn(dtm_src, [mapping(geom)], crop=True, filled=False)

    dsm_vals = _filled_float_array(dsm_arr[0], nodata=dsm_src.nodata, np=np)
    dtm_vals = _filled_float_array(dtm_arr[0], nodata=dtm_src.nodata, np=np)
    if dsm_vals.shape != dtm_vals.shape or dsm_transform != dtm_transform:
        aligned = np.full(dsm_vals.shape, np.nan, dtype="float64")
        reproject_fn(
            source=dtm_vals,
            destination=aligned,
            src_transform=dtm_transform,
            src_crs=dtm_src.crs,
            dst_transform=dsm_transform,
            dst_crs=dsm_src.crs,
            resampling=resampling.bilinear,
        )
        dtm_vals = aligned

    canopy = dsm_vals - dtm_vals
    canopy = canopy[np.isfinite(canopy)]
    canopy = canopy[canopy >= 0.0]
    return canopy


def _filled_float_array(masked_arr, *, nodata, np):
    if hasattr(masked_arr, "filled"):
        arr = masked_arr.filled(np.nan).astype("float64")
    else:
        arr = np.asarray(masked_arr, dtype="float64")
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr


def _width_proxy_m(*, hedge_geom, canopy_height_values, pixel_area_m2: float) -> float | None:
    if hedge_geom is None or hedge_geom.is_empty:
        return None
    line_length = float(hedge_geom.length)
    if line_length <= 0.0 or pixel_area_m2 <= 0.0:
        return None
    canopy_area = float((canopy_height_values >= 1.5).sum()) * float(abs(pixel_area_m2))
    return canopy_area / line_length
