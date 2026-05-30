"""SI7 — detect a wet/drainage ditch beside a hedgerow from the LiDAR DTM.

Mapped watercourse layers miss most field ditches. A 1 m DTM reveals them as narrow
linear depressions: a pixel sitting locally lower than its surroundings (negative relative
elevation). We flag a ditch where a meaningful fraction of the hedgerow corridor is
depressed. This is a Medium-confidence structural signal; the mapped-watercourse proximity
remains a Low-confidence fallback in the scorer.
"""

from __future__ import annotations

from hedge_features.deps import require_numpy, require_rasterio


def add_ditch(
    gdf,
    *,
    dtm_path: str | None,
    corridor_m: float = 8.0,
    window_m: float = 7.0,
    depth_threshold_m: float = 0.25,
    min_fraction: float = 0.02,
):
    """Add a ``ditch_present`` (1/0) column from DTM micro-topography. Returns ``(gdf, notes)``."""
    gdf = gdf.copy()
    if "ditch_present" not in gdf.columns:
        gdf["ditch_present"] = None
    if not dtm_path:
        return gdf, ["Ditch detection (SI7) skipped: LiDAR DTM not available."]

    try:
        rasterio = require_rasterio()
        np = require_numpy()
        from rasterio.mask import mask
        from scipy.ndimage import uniform_filter
        from shapely.geometry import mapping
    except Exception as exc:  # noqa: BLE001
        return gdf, [f"Ditch detection skipped: dependency unavailable ({exc})."]

    with rasterio.open(dtm_path) as dtm_src:
        buffers = gdf.geometry.buffer(float(corridor_m))
        buffers = gdf.geometry.__class__(buffers, crs=gdf.crs).to_crs(dtm_src.crs)
        pixel_m = abs(float(dtm_src.transform.a)) or 1.0
        win = max(3, int(round(float(window_m) / pixel_m)))

        for i, geom in enumerate(buffers):
            if geom is None or geom.is_empty:
                continue
            try:
                arr, _ = mask(dtm_src, [mapping(geom)], crop=True, filled=False)
            except Exception:  # noqa: BLE001
                continue
            dtm = arr[0]
            valid = ~np.ma.getmaskarray(dtm) if hasattr(dtm, "mask") else np.ones(dtm.shape, bool)
            dtm = np.asarray(dtm, dtype="float64")
            if dtm_src.nodata is not None:
                valid &= dtm != dtm_src.nodata
            valid &= np.isfinite(dtm)
            if valid.sum() < win * win:
                gdf.iat[i, gdf.columns.get_loc("ditch_present")] = 0
                continue
            filled = np.where(valid, dtm, np.nanmean(dtm[valid]))
            local_mean = uniform_filter(filled, size=win, mode="nearest")
            rel = filled - local_mean  # negative where locally low (ditch-like)
            ditch_mask = valid & (rel < -float(depth_threshold_m))
            frac = float(ditch_mask.sum()) / float(valid.sum())
            gdf.iat[i, gdf.columns.get_loc("ditch_present")] = 1 if frac >= float(min_fraction) else 0

    return gdf, ["SI7 wet-ditch presence inferred from LiDAR DTM micro-topography."]
