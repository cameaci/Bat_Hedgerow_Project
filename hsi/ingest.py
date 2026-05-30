"""Input ingestion — thin wrapper over the proven ``hedge_features.io`` plumbing."""

from __future__ import annotations

from pathlib import Path

from hedge_features.io import (
    prepare_working_gdf,
    read_input_geodata,
    validate_hedgerow_geometry_types,
)

from . import config


def load_hedgerows(input_path: str | Path, *, input_crs: str | None = None):
    """Read a hedgerow layer and return a working GeoDataFrame in the analysis CRS.

    Accepts zipped shapefile, GeoPackage, GeoJSON or a bare ``.shp``. Validates that
    geometries are lineal, repairs invalid ones, assigns a stable ``hf_uid`` and
    reprojects to EPSG:27700.

    Returns ``(gdf, notes)``.
    """
    gdf, notes = read_input_geodata(input_path, input_crs=input_crs)
    validate_hedgerow_geometry_types(gdf)
    gdf, source_crs, repaired = prepare_working_gdf(
        gdf, working_crs=config.WORKING_CRS, id_column="hf_uid"
    )
    if repaired:
        notes.append(f"Repaired {repaired} invalid geometr{'y' if repaired == 1 else 'ies'}.")
    notes.append(f"Input CRS detected as {source_crs}; reprojected to {config.WORKING_CRS}.")
    return gdf, notes
