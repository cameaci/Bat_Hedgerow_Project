"""Compute the remote/GIS proxy columns that feed the seven WSP suitability indices.

This module only *produces feature columns*; the band mapping and confidence live in
:mod:`hsi.score`. Every step degrades gracefully: a missing data source leaves its
columns null, and the scorer then falls back (or applies the precautionary default).

  SI1 Height, SI2 Width, SI3 Gappiness, SI5 Trees  <- EA LiDAR canopy-height model
  SI4 Arable margin                                <- CROME arable adjacency / WorldCover cropland
  SI5 Trees (fallback)                             <- WorldCover tree cover
  SI7 Wet ditch                                    <- watercourse proximity / WorldCover water+wetland
  SI6 Woody species diversity                      <- precautionary default (handled in scorer)
"""

from __future__ import annotations

from hedge_features.features.lidar_structure import add_lidar_hedgerow_structure_features
from hedge_features.features.raster import add_raster_categorical_proportions_in_buffers
from hedge_features.features.vector import (
    add_vector_distance,
    add_vector_line_density_in_buffers,
    add_vector_polygon_composition_in_buffers,
)

from . import config
from .datasets import DataResolver


def compute_si_proxies(gdf, resolver: DataResolver):
    """Add SI proxy feature columns to ``gdf``. Returns ``(gdf, notes)``."""
    notes: list[str] = []

    # --- LiDAR canopy structure (SI1/SI2/SI3 and SI5 fallback) ----------------------
    dtm_path = resolver.get_raster_path("ea_lidar_dtm")
    dsm_path = resolver.get_raster_path("ea_lidar_dsm")
    try:
        lidar = add_lidar_hedgerow_structure_features(
            gdf, dtm_path=dtm_path, dsm_path=dsm_path, height_buffer_m=5.0, continuity_buffer_m=10.0
        )
        gdf = lidar.gdf
        notes.extend(lidar.notes)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"LiDAR structure step failed ({exc}); SI1/SI2/SI3/SI5 will use fallbacks.")
    if not dtm_path or not dsm_path:
        notes.append(
            "EA LiDAR DTM/DSM not provided — SI1 (height), SI2 (width) and SI3 (gappiness) "
            "cannot be derived and the WSP category will be flagged Incomplete. Drop 1 m tiles "
            "into data/lidar/{dtm,dsm} to enable them."
        )

    # --- WorldCover land cover (SI4 cropland, SI5 tree fallback, SI7 water/wetland) ---
    wc_path = resolver.get_raster_path("worldcover")
    try:
        wc = add_raster_categorical_proportions_in_buffers(
            gdf,
            wc_path,
            radii_m=[100, 250],
            class_map=config.WORLDCOVER_CLASS_MAP,
            column_template="buf{radius}_worldcover_{class_name}_pct",
        )
        gdf = wc.gdf
        notes.extend(wc.notes)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"WorldCover step failed ({exc}).")

    # --- Watercourse proximity / density (SI7 + context water) ----------------------
    rivers = resolver.get_vector("os_open_rivers")
    try:
        dist = add_vector_distance(
            gdf, rivers, distance_column="dist_os_river_m",
            geometry_kinds=["LineString", "MultiLineString"],
        )
        gdf = dist.gdf
        notes.extend(dist.notes)
        dens = add_vector_line_density_in_buffers(
            gdf, rivers, radii_m=[100, 250],
            density_column_template="buf{radius}_os_river_density_m_per_ha",
        )
        gdf = dens.gdf
        notes.extend(dens.notes)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Watercourse proximity step failed ({exc}).")

    # --- CROME arable adjacency (optional SI4 refinement) ---------------------------
    crome = resolver.get_vector("crome")
    if crome is not None and not crome.empty:
        class_field = next((f for f in config.CROME_CLASS_FIELD_CANDIDATES if f in crome.columns), None)
        if class_field is None:
            notes.append("CROME provided but no recognised class field; SI4 will use WorldCover cropland.")
        else:
            try:
                comp = add_vector_polygon_composition_in_buffers(
                    gdf, crome, radii_m=[50],
                    class_field=class_field,
                    selected_classes={"arable": {"contains_any": list(config.CROME_ARABLE_TOKENS)}},
                    column_template="crome_{class_name}_pct",
                )
                gdf = comp.gdf
                notes.extend(comp.notes)
            except Exception as exc:  # noqa: BLE001
                notes.append(f"CROME arable step failed ({exc}); SI4 will use WorldCover cropland.")

    return gdf, notes
