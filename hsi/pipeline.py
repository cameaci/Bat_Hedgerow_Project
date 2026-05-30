"""End-to-end orchestration: hedgerow layer -> scored, ranked GeoDataFrame.

The heavy GIS work (``run_features``) is separated from the cheap, weight-dependent
scoring (``hsi.score.apply_scoring``) so a UI can cache the features once and re-rank
instantly when the user moves a weight slider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hedge_features.features.geometry import add_geometry_metrics
from hedge_features.features.network import add_network_metrics

from . import config
from .context import compute_context_subindices
from .datasets import DataResolver
from .indices import compute_si_proxies
from .ingest import load_hedgerows
from .score import apply_scoring, resolve_structural


# Datasets we attempt to use, for provenance metadata.
USED_DATASETS = [
    "ea_lidar_dtm", "ea_lidar_dsm", "worldcover", "os_open_rivers", "os_open_roads",
    "ne_phi", "ne_awi", "osm_buildings", "osm_structures_roost", "crome",
    "nightlights", "ancient_trees",
]


@dataclass(slots=True)
class FeaturesResult:
    """Output of the heavy GIS stage; feed ``gdf`` to ``apply_scoring`` to rank."""

    gdf: Any
    notes: list[str] = field(default_factory=list)
    dataset_status: dict[str, str] = field(default_factory=dict)
    dataset_metadata: list[dict[str, Any]] = field(default_factory=list)


def run_features(
    source,
    *,
    input_crs: str | None = None,
    data_dir: str | Path | None = None,
    allow_live_fetch: bool = True,
    si6_default_band: int = config.SI6_DEFAULT_BAND,
) -> FeaturesResult:
    """Ingest, extract all GIS proxies/context and resolve the weight-independent SIs.

    ``source`` may be a file path/str or an already-loaded GeoDataFrame.
    """
    notes: list[str] = []
    if isinstance(source, (str, Path)):
        gdf, ingest_notes = load_hedgerows(source, input_crs=input_crs)
        notes.extend(ingest_notes)
    else:
        gdf = source.copy()
        if gdf.crs is not None and str(gdf.crs) != str(config.WORKING_CRS):
            gdf = gdf.to_crs(config.WORKING_CRS)

    gdf = add_geometry_metrics(gdf)
    gdf, net_notes = add_network_metrics(gdf, tolerance_m=1.0)
    notes.extend(net_notes)

    resolver = DataResolver(gdf, data_dir=data_dir, allow_live_fetch=allow_live_fetch)

    gdf, si_notes = compute_si_proxies(gdf, resolver)
    notes.extend(si_notes)
    gdf, ctx_notes = compute_context_subindices(gdf, resolver)
    notes.extend(ctx_notes)

    gdf = resolve_structural(gdf, si6_default_band=si6_default_band)

    notes.extend(resolver.notes)
    return FeaturesResult(
        gdf=gdf,
        notes=notes,
        dataset_status=resolver.status,
        dataset_metadata=resolver.metadata_records(USED_DATASETS),
    )


def run_hsi(
    source,
    *,
    input_crs: str | None = None,
    data_dir: str | Path | None = None,
    allow_live_fetch: bool = True,
    settings: config.ScoreSettings | None = None,
) -> FeaturesResult:
    """Convenience single-shot: features + scoring with the given (or default) weights."""
    settings = settings or config.ScoreSettings()
    result = run_features(
        source,
        input_crs=input_crs,
        data_dir=data_dir,
        allow_live_fetch=allow_live_fetch,
        si6_default_band=settings.si6_default_band,
    )
    result.gdf = apply_scoring(result.gdf, settings=settings)
    return result
