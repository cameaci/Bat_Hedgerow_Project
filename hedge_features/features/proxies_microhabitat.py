from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..deps import require_numpy, require_rasterio
from ..exceptions import OptionalDependencyError
from .raster import add_raster_categorical_proportions_in_buffers


@dataclass(slots=True)
class MicrohabitatProxyFeatureResult:
    gdf: Any
    notes: list[str]


def add_microhabitat_proxy_features(
    hedges_gdf,
    *,
    worldcover_path: str | None = None,
    copdem_path: str | None = None,
    include_roost_proxy_alias: bool = True,
):
    gdf = hedges_gdf.copy()
    notes: list[str] = []
    for c in (
        "mhb_corridor10_tree_pct",
        "mhb_corridor10_built_pct",
        "mhb_corridor10_water_pct",
        "mhb_corridor10_wetland_pct",
        "mhb_dem_slope_mean_50m",
        "mhb_dem_shelter_idx_100m",
        "mhb_awi_within100m_flag",
        "mhb_phi_wetland_within100m_flag",
        "mhb_water_dist_m",
        "mhb_roost_proxy_score",
        "mhb_status",
    ):
        if c not in gdf.columns:
            gdf[c] = None

    status_flags: list[str] = []

    if worldcover_path:
        wc_result = add_raster_categorical_proportions_in_buffers(
            gdf,
            worldcover_path,
            radii_m=[10],
            class_map={"tree": [10], "built": [50], "water": [80], "wetland": [90]},
            column_template="mhb_corridor10_{class_name}_pct",
            patch_richness_column_template=None,
        )
        gdf = wc_result.gdf
        notes.extend(wc_result.notes)
        status_flags.append("worldcover")
    else:
        notes.append("Microhabitat proxies: WorldCover unavailable, corridor land-cover metrics not computed.")

    if copdem_path:
        gdf, dem_notes = _add_dem_local_slope_and_shelter(gdf, copdem_path)
        notes.extend(dem_notes)
        status_flags.append("copdem")
    else:
        notes.append("Microhabitat proxies: Copernicus DEM unavailable, terrain shelter/slope proxies not computed.")

    # Reuse existing derived columns where available.
    if "dist_awi_ancwood_m" in gdf.columns:
        try:
            gdf["mhb_awi_within100m_flag"] = (gdf["dist_awi_ancwood_m"].astype(float) <= 100.0).astype(int)
        except Exception:
            pass
    if "buf100_phi_wetland_pct" in gdf.columns:
        try:
            gdf["mhb_phi_wetland_within100m_flag"] = (gdf["buf100_phi_wetland_pct"].astype(float) > 0).astype(int)
        except Exception:
            pass
    if "dist_os_river_m" in gdf.columns:
        gdf["mhb_water_dist_m"] = gdf["dist_os_river_m"]
    if include_roost_proxy_alias and "roostpx_struct_proxy_score" in gdf.columns:
        gdf["mhb_roost_proxy_score"] = gdf["roostpx_struct_proxy_score"]

    gdf["mhb_status"] = "ok" if status_flags else "source_missing"
    if status_flags:
        notes.append(
            "Microhabitat proxy features computed (corridor WorldCover + DEM local terrain proxies + reused proximity/context metrics)."
        )
    return MicrohabitatProxyFeatureResult(gdf=gdf, notes=notes)


def _add_dem_local_slope_and_shelter(gdf, copdem_path: str):
    notes: list[str] = []
    try:
        rasterio = require_rasterio()
        np = require_numpy()
        from rasterio.mask import mask
        from shapely.geometry import mapping
    except OptionalDependencyError:
        gdf["mhb_dem_slope_mean_50m"] = None
        gdf["mhb_dem_shelter_idx_100m"] = None
        return gdf, ["rasterio/numpy missing; DEM microhabitat proxies set to null."]

    with rasterio.open(copdem_path) as src:
        gdf["mhb_dem_slope_mean_50m"] = None
        gdf["mhb_dem_shelter_idx_100m"] = None
        # Prepare buffers in raster CRS.
        buf50 = gdf.geometry.buffer(50)
        buf100 = gdf.geometry.buffer(100)
        buf50 = gdf.geometry.__class__(buf50, crs=gdf.crs).to_crs(src.crs)
        buf100 = gdf.geometry.__class__(buf100, crs=gdf.crs).to_crs(src.crs)

        for i, (geom50, geom100) in enumerate(zip(buf50, buf100)):
            slope_mean = None
            shelter_idx = None
            try:
                arr50, tf50 = mask(src, [mapping(geom50)], crop=True, filled=False)
                slope_mean = _mean_slope_degrees(arr50[0], tf50, nodata=src.nodata)
            except Exception:
                slope_mean = None
            try:
                arr100, _ = mask(src, [mapping(geom100)], crop=True, filled=False)
                shelter_idx = _shelter_index_from_relief(arr100[0], nodata=src.nodata)
            except Exception:
                shelter_idx = None

            gdf.iat[i, gdf.columns.get_loc("mhb_dem_slope_mean_50m")] = slope_mean
            gdf.iat[i, gdf.columns.get_loc("mhb_dem_shelter_idx_100m")] = shelter_idx

    notes.append(
        "mhb_dem_shelter_idx_100m uses a local-relief proxy: 1 / (1 + relief_10_90 / 10), where relief_10_90 is DEM p90-p10 within 100 m."
    )
    return gdf, notes


def _mean_slope_degrees(masked_arr, transform, nodata=None):
    np = require_numpy()
    if hasattr(masked_arr, "filled"):
        arr = masked_arr.filled(np.nan).astype(float)
    else:
        arr = np.asarray(masked_arr, dtype=float)
    if nodata is not None:
        arr[arr == nodata] = np.nan
    if np.isnan(arr).all():
        return None
    # Fill sparse NaNs with local median fallback for simple gradient estimation.
    finite = arr[~np.isnan(arr)]
    if finite.size == 0:
        return None
    arr = np.where(np.isnan(arr), float(np.nanmedian(arr)), arr)
    xres = abs(float(transform.a)) or 1.0
    yres = abs(float(transform.e)) or 1.0
    gy, gx = np.gradient(arr, yres, xres)
    slope = np.degrees(np.arctan(np.sqrt(gx * gx + gy * gy)))
    slope = slope[np.isfinite(slope)]
    if slope.size == 0:
        return None
    return float(np.mean(slope))


def _shelter_index_from_relief(masked_arr, nodata=None):
    np = require_numpy()
    if hasattr(masked_arr, "compressed"):
        vals = masked_arr.compressed().astype(float)
    else:
        vals = np.asarray(masked_arr, dtype=float).reshape(-1)
    if nodata is not None:
        vals = vals[vals != nodata]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    p10 = float(np.percentile(vals, 10))
    p90 = float(np.percentile(vals, 90))
    relief = max(0.0, p90 - p10)
    return float(1.0 / (1.0 + (relief / 10.0)))
