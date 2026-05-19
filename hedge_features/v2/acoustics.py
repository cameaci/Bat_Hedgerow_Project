from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


STATIC_ACOUSTIC_SUMMARY_VERSION = "static_acoustic_summary_v2"


@dataclass(frozen=True, slots=True)
class StaticAcousticSummarySettings:
    hedgerow_id_column: str = "hedgerow_id"
    species_column: str = "species"
    datetime_column: str = "datetime"
    activity_column: str | None = None
    season_column: str | None = "survey_season"
    detector_id_column: str | None = "detector_id"
    detector_model_column: str | None = "detector_model"
    microphone_height_column: str | None = "microphone_height_m"


def summarise_static_acoustics(detections_df, *, settings: StaticAcousticSummarySettings | None = None):
    """Summarise static detector rows into hedgerow-season-species activity metrics."""
    import pandas as pd

    settings = settings or StaticAcousticSummarySettings()
    if settings.hedgerow_id_column not in detections_df.columns:
        raise ValueError(f"Hedgerow id column '{settings.hedgerow_id_column}' was not found.")
    if settings.species_column not in detections_df.columns:
        raise ValueError(f"Species column '{settings.species_column}' was not found.")

    df = detections_df.copy()
    if settings.datetime_column in df.columns:
        dt = pd.to_datetime(df[settings.datetime_column], errors="coerce", utc=True)
        df["_acoustic_night"] = dt.dt.date.astype("string")
        df["_survey_season"] = (
            df[settings.season_column].astype("string")
            if settings.season_column and settings.season_column in df.columns
            else dt.dt.month.map(_season_from_month).astype("string")
        )
    else:
        df["_acoustic_night"] = pd.NA
        df["_survey_season"] = (
            df[settings.season_column].astype("string")
            if settings.season_column and settings.season_column in df.columns
            else "unknown"
        )

    if settings.activity_column and settings.activity_column in df.columns:
        df["_passes"] = pd.to_numeric(df[settings.activity_column], errors="coerce").fillna(0.0)
    else:
        df["_passes"] = 1.0

    group_cols = [settings.hedgerow_id_column, "_survey_season", settings.species_column]
    grouped = df.groupby(group_cols, dropna=False, sort=True)
    rows: list[dict[str, Any]] = []
    for (hedge_id, season, species), group in grouped:
        nights = int(group["_acoustic_night"].dropna().nunique())
        total = float(group["_passes"].sum())
        rows.append(
            {
                settings.hedgerow_id_column: str(hedge_id),
                "survey_season": str(season),
                "acoustic_species": str(species),
                "acoustic_total_passes": total,
                "acoustic_nights": nights,
                "acoustic_passes_per_night": round(total / nights, 6) if nights else None,
                "acoustic_activity_index": round(total / nights, 6) if nights else total,
                "acoustic_baiv_ready": int(nights > 0),
                "acoustic_species_richness_in_hedge_season": int(group[settings.species_column].astype("string").nunique()),
            }
        )
    summary_df = pd.DataFrame(rows)
    metadata_fields = [
        col
        for col in (settings.detector_id_column, settings.detector_model_column, settings.microphone_height_column)
        if col
    ]
    missing_metadata = [col for col in metadata_fields if col not in detections_df.columns]
    run_summary = {
        "method_version": STATIC_ACOUSTIC_SUMMARY_VERSION,
        "settings": asdict(settings),
        "input_detection_rows": int(len(detections_df)),
        "summary_rows": int(len(summary_df)),
        "hedgerow_count": int(summary_df[settings.hedgerow_id_column].nunique()) if not summary_df.empty else 0,
        "missing_effort_metadata_fields": missing_metadata,
        "qa_notes": (
            ["Detector effort metadata is incomplete; cross-location comparability should be reviewed."]
            if missing_metadata
            else []
        ),
    }
    return summary_df, run_summary


def _season_from_month(month) -> str:
    try:
        m = int(month)
    except Exception:
        return "unknown"
    if m in {3, 4, 5}:
        return "Spring"
    if m in {6, 7, 8}:
        return "Summer"
    if m in {9, 10, 11}:
        return "Autumn"
    return "Winter"
