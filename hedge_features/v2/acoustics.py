from __future__ import annotations

import io
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


STATIC_ACOUSTIC_SUMMARY_VERSION = "static_acoustic_summary_v2"

CANONICAL_ACOUSTIC_COLUMNS = (
    "hedgerow_id",
    "survey_night",
    "survey_season",
    "acoustic_species",
    "acoustic_passes",
    "detector_id",
    "detector_model",
    "microphone_height_m",
    "qa_status",
)

REQUIRED_ACOUSTIC_COLUMNS = ("hedgerow_id", "acoustic_species")

ACOUSTIC_COLUMN_ALIASES = {
    "hedgerow_id": (
        "hedgerow_id",
        "hedge_id",
        "hedgeid",
        "hf_uid",
        "source_hf_uid",
        "corridor_id",
        "hedgerow",
    ),
    "survey_night": (
        "survey_night",
        "night",
        "date",
        "survey_date",
        "recording_date",
        "datetime",
        "timestamp",
        "date_time",
        "start_time",
        "recording_start",
    ),
    "survey_season": ("survey_season", "season", "survey_period", "period"),
    "acoustic_species": (
        "species",
        "taxon",
        "class",
        "predicted_class",
        "bat_species",
        "species_group",
        "label",
    ),
    "acoustic_passes": (
        "passes",
        "bat_passes",
        "calls",
        "call_count",
        "detections",
        "count",
        "activity",
        "number_of_passes",
    ),
    "detector_id": ("detector_id", "static_id", "location_id", "recorder_id", "station_id", "detector"),
    "detector_model": ("detector_model", "recorder_model", "model", "detector_type", "recorder_type"),
    "microphone_height_m": (
        "microphone_height_m",
        "mic_height_m",
        "microphone_height",
        "mic_height",
        "microphone_height_metres",
    ),
    "qa_status": ("qa_status", "manual_qa_status", "verified", "verification_status", "qa", "confidence"),
}


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
    qa_status_column: str | None = "qa_status"


def parse_acoustic_survey_table(
    table,
    *,
    column_mapping: dict[str, str] | None = None,
    settings: StaticAcousticSummarySettings | None = None,
):
    """Normalise a static detector result table into the V2 acoustic schema.

    The input is expected to be tabular survey output, not audio. It may be a
    pandas DataFrame, a CSV/TSV/markdown-like text table, or a CSV/XLSX path.
    """
    import pandas as pd

    settings = settings or StaticAcousticSummarySettings()
    raw = _coerce_table(table)
    mapping = _build_mapping(raw, settings=settings, column_mapping=column_mapping)
    missing_required = [col for col in REQUIRED_ACOUSTIC_COLUMNS if mapping.get(col) is None]
    if missing_required:
        raise ValueError(
            "Acoustic survey table is missing required column(s): "
            + ", ".join(missing_required)
            + ". Supply a column_mapping override if the source uses non-standard names."
        )

    out = raw.copy()
    out["hedgerow_id"] = raw[mapping["hedgerow_id"]].astype("string")
    out["acoustic_species"] = raw[mapping["acoustic_species"]].astype("string")

    if mapping.get("survey_night"):
        dt = pd.to_datetime(raw[mapping["survey_night"]], errors="coerce", utc=True)
        out["survey_night"] = dt.dt.date.astype("string")
        survey_month = dt.dt.month
    else:
        out["survey_night"] = pd.NA
        survey_month = None

    if mapping.get("survey_season"):
        out["survey_season"] = raw[mapping["survey_season"]].astype("string").fillna("unknown")
    elif survey_month is not None:
        out["survey_season"] = survey_month.map(_season_from_month).astype("string")
    else:
        out["survey_season"] = "unknown"

    if mapping.get("acoustic_passes"):
        out["acoustic_passes"] = pd.to_numeric(raw[mapping["acoustic_passes"]], errors="coerce").fillna(0.0)
    else:
        out["acoustic_passes"] = 1.0

    for canonical in ("detector_id", "detector_model", "qa_status"):
        if mapping.get(canonical):
            out[canonical] = raw[mapping[canonical]].astype("string")
        else:
            out[canonical] = pd.NA
    if mapping.get("microphone_height_m"):
        out["microphone_height_m"] = pd.to_numeric(raw[mapping["microphone_height_m"]], errors="coerce")
    else:
        out["microphone_height_m"] = pd.NA

    audit = {
        "method_version": STATIC_ACOUSTIC_SUMMARY_VERSION,
        "input_rows": int(len(raw)),
        "column_mapping": mapping,
        "missing_required_columns": missing_required,
        "missing_effort_metadata_fields": [
            col for col in ("detector_id", "detector_model", "microphone_height_m") if mapping.get(col) is None
        ],
        "warnings": _mapping_warnings(mapping),
    }
    return out, audit


def summarise_static_acoustics(detections_df, *, settings: StaticAcousticSummarySettings | None = None):
    """Summarise static detector rows into defensible hedgerow-season-species metrics."""
    import pandas as pd

    settings = settings or StaticAcousticSummarySettings()
    df, audit = parse_acoustic_survey_table(detections_df, settings=settings)

    group_cols = ["hedgerow_id", "survey_season", "acoustic_species"]
    grouped = df.groupby(group_cols, dropna=False, sort=True)
    rows: list[dict[str, Any]] = []
    for (hedge_id, season, species), group in grouped:
        nights = int(group["survey_night"].dropna().nunique())
        total = float(group["acoustic_passes"].sum())
        flags = _qa_comparability_flags(group)
        rows.append(
            {
                "summary_level": "hedgerow_season_species",
                "hedgerow_id": str(hedge_id),
                "survey_season": str(season),
                "acoustic_species": str(species),
                "acoustic_total_passes": total,
                "acoustic_nights": nights,
                "acoustic_passes_per_night": round(total / nights, 6) if nights else None,
                "acoustic_activity_index": round(total / nights, 6) if nights else total,
                "acoustic_baiv_ready": int(nights > 0),
                "detector_effort_nights": nights,
                "detector_effort_complete": int(not flags and nights > 0),
                "acoustic_qa_comparability_flag": ";".join(flags) if flags else "ok",
            }
        )
    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        season_totals = (
            df.groupby(["hedgerow_id", "survey_season"], dropna=False)
            .agg(
                acoustic_hedgerow_season_total_passes=("acoustic_passes", "sum"),
                acoustic_species_richness_in_hedge_season=("acoustic_species", "nunique"),
                acoustic_hedgerow_season_nights=("survey_night", lambda x: int(x.dropna().nunique())),
            )
            .reset_index()
        )
        season_totals["acoustic_hedgerow_season_passes_per_night"] = season_totals.apply(
            lambda row: round(float(row["acoustic_hedgerow_season_total_passes"]) / row["acoustic_hedgerow_season_nights"], 6)
            if int(row["acoustic_hedgerow_season_nights"]) > 0
            else None,
            axis=1,
        )
        summary_df = summary_df.merge(season_totals, how="left", on=["hedgerow_id", "survey_season"])

    effort_completeness = {
        "rows_with_survey_night": int(df["survey_night"].notna().sum()),
        "rows_with_detector_id": int(df["detector_id"].notna().sum()),
        "rows_with_detector_model": int(df["detector_model"].notna().sum()),
        "rows_with_microphone_height_m": int(df["microphone_height_m"].notna().sum()),
        "rows_with_qa_status": int(df["qa_status"].notna().sum()),
    }
    missing_metadata = audit["missing_effort_metadata_fields"]
    qa_notes = []
    if missing_metadata:
        qa_notes.append("Detector effort metadata is incomplete; cross-location comparability should be reviewed.")
    if effort_completeness["rows_with_survey_night"] < len(df):
        qa_notes.append("One or more rows lack a parseable survey night; passes per night may be limited.")

    run_summary = {
        "method_version": STATIC_ACOUSTIC_SUMMARY_VERSION,
        "settings": asdict(settings),
        "input_detection_rows": int(len(df)),
        "summary_rows": int(len(summary_df)),
        "hedgerow_count": int(summary_df["hedgerow_id"].nunique()) if not summary_df.empty else 0,
        "column_audit": audit,
        "missing_effort_metadata_fields": missing_metadata,
        "effort_completeness": effort_completeness,
        "qa_notes": qa_notes,
    }
    return summary_df, run_summary


def _coerce_table(table):
    import pandas as pd

    if hasattr(table, "columns"):
        return table.copy()
    if isinstance(table, (str, bytes)):
        text = table.decode("utf-8") if isinstance(table, bytes) else table
        candidate_path = Path(text)
        if "\n" not in text and candidate_path.exists():
            return _read_table_path(candidate_path)
        markdown = _parse_markdown_table(text)
        if markdown is not None:
            return markdown
        return pd.read_csv(io.StringIO(text), sep=None, engine="python")
    if isinstance(table, Path):
        return _read_table_path(table)
    raise TypeError("Acoustic survey table must be a DataFrame, text table, or CSV/XLSX path.")


def _read_table_path(path: Path):
    import pandas as pd

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _parse_markdown_table(text: str):
    import pandas as pd

    lines = [line.strip() for line in text.splitlines() if "|" in line]
    if len(lines) < 2:
        return None
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells or all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells):
            continue
        rows.append(cells)
    if len(rows) < 2:
        return None
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        return None
    return pd.DataFrame(rows[1:], columns=rows[0])


def _build_mapping(raw, *, settings: StaticAcousticSummarySettings, column_mapping: dict[str, str] | None) -> dict[str, str | None]:
    mapping = {canonical: None for canonical in CANONICAL_ACOUSTIC_COLUMNS}
    explicit = {
        "hedgerow_id": settings.hedgerow_id_column,
        "acoustic_species": settings.species_column,
        "survey_night": settings.datetime_column,
        "acoustic_passes": settings.activity_column,
        "survey_season": settings.season_column,
        "detector_id": settings.detector_id_column,
        "detector_model": settings.detector_model_column,
        "microphone_height_m": settings.microphone_height_column,
        "qa_status": settings.qa_status_column,
    }
    for canonical, source in explicit.items():
        if source and source in raw.columns:
            mapping[canonical] = source
    for canonical in CANONICAL_ACOUSTIC_COLUMNS:
        if mapping[canonical] is None:
            mapping[canonical] = _detect_column(raw.columns, ACOUSTIC_COLUMN_ALIASES.get(canonical, (canonical,)))
    if column_mapping:
        for canonical, source in column_mapping.items():
            if canonical not in mapping:
                continue
            mapping[canonical] = source if source in raw.columns else None
    return mapping


def _detect_column(columns, aliases: tuple[str, ...]) -> str | None:
    normalized = {_normalize_column_name(col): col for col in columns}
    for alias in aliases:
        key = _normalize_column_name(alias)
        if key in normalized:
            return normalized[key]
    return None


def _normalize_column_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _mapping_warnings(mapping: dict[str, str | None]) -> list[str]:
    warnings = []
    if mapping.get("survey_night") is None:
        warnings.append("No survey night/date column was mapped; pass-per-night metrics will be limited.")
    if mapping.get("acoustic_passes") is None:
        warnings.append("No pass/count column was mapped; each row will be treated as one pass.")
    if mapping.get("qa_status") is None:
        warnings.append("No QA status column was mapped; manual verification comparability cannot be checked.")
    return warnings


def _qa_comparability_flags(group) -> list[str]:
    flags = []
    if group["detector_id"].isna().any():
        flags.append("missing_detector_id")
    if group["detector_model"].isna().any():
        flags.append("missing_detector_model")
    if group["microphone_height_m"].isna().any():
        flags.append("missing_microphone_height_m")
    if group["detector_model"].dropna().astype("string").nunique() > 1:
        flags.append("mixed_detector_model")
    if group["microphone_height_m"].dropna().nunique() > 1:
        flags.append("mixed_microphone_height_m")
    qa = group["qa_status"].dropna().astype("string").str.lower()
    if not qa.empty and not qa.str.contains("pass|verified|accepted|manual|true|1", regex=True).any():
        flags.append("qa_status_requires_review")
    return flags


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
