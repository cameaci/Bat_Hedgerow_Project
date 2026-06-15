"""Outputs: GeoPackage / Shapefile / CSV writers and a defensible method statement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hedge_features.io import write_geodata

from . import config


def build_run_metadata(
    *,
    settings: config.ScoreSettings,
    dataset_status: dict[str, str],
    dataset_metadata: list[dict[str, Any]],
    feature_count: int,
    notes: list[str],
) -> dict[str, Any]:
    return {
        "tool": "england-bat-hedgerow-hsi",
        "method": "WSP/HyNet 7-index hedgerow suitability (SI1-SI7) + landscape context",
        "guidance_regime_version": settings.guidance_regime_version,
        "working_crs": config.WORKING_CRS,
        "feature_count": int(feature_count),
        "settings": {
            "si_weights": settings.si_weights,
            "context_weights": settings.context_weights,
            "alpha_structure_vs_context": settings.alpha,
            "si6_default_band": settings.si6_default_band,
        },
        "dataset_status": dataset_status,
        "datasets": dataset_metadata,
        "notes": notes,
    }


def write_outputs(
    gdf,
    output_path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    write_csv: bool = True,
    write_shapefile_zip: bool = True,
) -> dict[str, str]:
    """Write the scored layer as GeoPackage (+ optional CSV and zipped shapefile)."""
    return write_geodata(
        gdf,
        output_path,
        metadata=metadata,
        export_crs=config.EXPORT_CRS,
        write_csv=write_csv,
        write_shapefile_zip=write_shapefile_zip,
    )


def _category_counts(gdf) -> dict[str, int]:
    counts: dict[str, int] = {}
    if "hsi_wsp_category" in gdf.columns:
        for value in gdf["hsi_wsp_category"]:
            counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def build_method_statement(
    gdf,
    *,
    settings: config.ScoreSettings,
    dataset_status: dict[str, str],
) -> str:
    """Produce a Markdown method statement suitable for a client deliverable."""
    n = len(gdf)
    counts = _category_counts(gdf)
    complete = int(gdf["hsi_complete"].sum()) if "hsi_complete" in gdf.columns else 0

    lines: list[str] = []
    lines.append("# Hedgerow Suitability Index — Method Statement")
    lines.append("")
    lines.append(
        "A hedgerow suitability scoring approach was applied to provide a structured "
        "indication of relative bat habitat suitability and to support survey "
        "prioritisation. The assessment uses the seven suitability indices (SI1-SI7) of "
        "the WSP/HyNet hedgerow assessment: height, width, gappiness, arable field "
        "margin, trees present, woody species diversity and wet ditch presence. Each "
        "hedgerow is assigned a suitability category of Poor, Good or Excellent based on "
        "the unweighted arithmetic mean of its SI scores (1.00-1.69 Poor, 1.70-2.39 "
        "Good, >=2.40 Excellent)."
    )
    lines.append("")
    lines.append(
        "Because field access was not available for all hedgerows, indices were derived "
        "remotely where possible: height, width, gappiness and tree presence from EA "
        "National LiDAR Programme canopy-height models; arable margin from CROME / ESA "
        "WorldCover; wet ditch from watercourse proximity. Woody species diversity is not "
        "remotely verifiable and is given a precautionary default with low confidence. "
        "A separate landscape-context layer (woodland, water and roost proximity, "
        "darkness, connectivity and road quietness) refines the final survey-priority "
        "ranking but does not change the structural WSP category."
    )
    lines.append("")
    lines.append("## Scoring configuration")
    lines.append("")
    lines.append(f"- Structure vs context blend (alpha): {settings.alpha}")
    lines.append(f"- SI weights: {settings.si_weights}")
    lines.append(f"- Context weights: {settings.context_weights}")
    lines.append(f"- Guidance regime: {settings.guidance_regime_version}")
    lines.append("")
    lines.append("## Results summary")
    lines.append("")
    lines.append(f"- Hedgerows assessed: {n}")
    lines.append(f"- With a complete (all-SI) WSP category: {complete}")
    for category, count in counts.items():
        lines.append(f"- {category}: {count}")
    lines.append("")
    lines.append("## Data sources used")
    lines.append("")
    for name, status in sorted(dataset_status.items()):
        lines.append(f"- {name}: {status}")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- The index is a decision-support tool and does not confirm bat presence or "
        "absence. Higher suitability indicates a higher priority for survey, not a "
        "guarantee of activity."
    )
    lines.append(
        "- Where field access was not available, variables were derived from desk-based "
        "or proxy data and carry lower confidence (see the confidence columns). Results "
        "should be interpreted alongside survey findings and professional judgement."
    )
    lines.append(
        "- Where suitable survey data are available, calculated categories should be "
        "compared with observed bat activity (static detectors / crossing points) to "
        "refine the scoring approach."
    )
    return "\n".join(lines) + "\n"
