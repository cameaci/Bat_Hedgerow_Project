"""SI5 — estimate trees per 50 m from a LiDAR canopy-height model (local maxima).

More faithful to the WSP SI5 definition ("trees per 50 m") than a canopy-cover fraction:
detects individual tree crowns as local maxima of the 2-D canopy height model (CHM =
DSM - DTM) within the hedgerow corridor, then normalises the count to per-50 m.
"""

from __future__ import annotations

from hedge_features.deps import require_numpy, require_rasterio


def add_tree_count(
    gdf,
    *,
    dtm_path: str | None,
    dsm_path: str | None,
    corridor_m: float = 10.0,
    tree_height_min_m: float = 3.0,
    min_separation_m: float = 3.0,
):
    """Add a ``trees_per_50m`` column. Returns ``(gdf, notes)``; degrades to null if no LiDAR."""
    gdf = gdf.copy()
    if "lidar_trees_per_50m" not in gdf.columns:
        gdf["lidar_trees_per_50m"] = None
    if not dtm_path or not dsm_path:
        return gdf, ["Tree count (SI5) skipped: LiDAR DTM/DSM not available."]

    try:
        rasterio = require_rasterio()
        np = require_numpy()
        from rasterio.mask import mask
        from rasterio.warp import Resampling, reproject
        from scipy.ndimage import label, maximum_filter
        from shapely.geometry import mapping
    except Exception as exc:  # noqa: BLE001
        return gdf, [f"Tree count skipped: dependency unavailable ({exc})."]

    notes: list[str] = []
    with rasterio.open(dtm_path) as dtm_src, rasterio.open(dsm_path) as dsm_src:
        buffers = gdf.geometry.buffer(float(corridor_m))
        buffers = gdf.geometry.__class__(buffers, crs=gdf.crs).to_crs(dtm_src.crs)
        hedges_r = gdf.to_crs(dtm_src.crs)
        pixel_m = abs(float(dtm_src.transform.a)) or 1.0
        win = max(1, int(round(float(min_separation_m) / pixel_m)))

        for i, (buf, geom) in enumerate(zip(buffers, hedges_r.geometry)):
            if geom is None or geom.is_empty:
                continue
            length_m = float(geom.length)
            if length_m <= 0:
                continue
            try:
                chm = _corridor_chm_2d(dsm_src, dtm_src, buf, mask, reproject, Resampling, np, mapping)
            except Exception:  # noqa: BLE001
                continue
            if chm is None or chm.size == 0:
                continue
            chm = np.where(np.isfinite(chm), chm, 0.0)
            peaks = (chm == maximum_filter(chm, size=2 * win + 1)) & (chm >= float(tree_height_min_m))
            crown_count = 0 if not peaks.any() else int(label(peaks)[1])
            gdf.iat[i, gdf.columns.get_loc("lidar_trees_per_50m")] = float(crown_count) / (length_m / 50.0)

    notes.append("SI5 trees-per-50m estimated from LiDAR canopy-height local maxima.")
    return gdf, notes


def _corridor_chm_2d(dsm_src, dtm_src, geom, mask, reproject, Resampling, np, mapping):
    """Return a 2-D canopy-height array (DSM-DTM) masked to ``geom`` (NaN outside)."""
    dsm_arr, dsm_t = mask(dsm_src, [mapping(geom)], crop=True, filled=False)
    dtm_arr, dtm_t = mask(dtm_src, [mapping(geom)], crop=True, filled=False)
    dsm = _filled(dsm_arr[0], dsm_src.nodata, np)
    dtm = _filled(dtm_arr[0], dtm_src.nodata, np)
    if dsm.shape != dtm.shape or dsm_t != dtm_t:
        aligned = np.full(dsm.shape, np.nan, dtype="float64")
        reproject(
            source=dtm, destination=aligned,
            src_transform=dtm_t, src_crs=dtm_src.crs,
            dst_transform=dsm_t, dst_crs=dsm_src.crs,
            resampling=Resampling.bilinear,
        )
        dtm = aligned
    chm = dsm - dtm
    chm[chm < 0] = 0.0
    return chm


def _filled(masked_arr, nodata, np):
    arr = masked_arr.filled(np.nan).astype("float64") if hasattr(masked_arr, "filled") else np.asarray(masked_arr, dtype="float64")
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr
