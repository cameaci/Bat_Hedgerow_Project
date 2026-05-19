from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


PROJECT_SCHEMA_VERSION = "bat_hedgerow_project_schema_v2"


@dataclass(frozen=True, slots=True)
class ProjectSchemaSettings:
    hedgerow_id_column: str = "hedgerow_id"
    fallback_id_columns: tuple[str, ...] = ("hf_uid", "source_hf_uid")
    required_project_columns: tuple[str, ...] = ("project_id", "scheme_name", "section_id")
    acoustic_effort_columns: tuple[str, ...] = ("detector_id", "detector_model", "microphone_height_m", "survey_season")
    allow_missing_crs: bool = False


@dataclass(frozen=True, slots=True)
class ProjectReadinessReport:
    schema_version: str
    status: str
    row_count: int
    hedgerow_id_column: str | None
    duplicate_hedgerow_ids: int
    missing_project_columns: list[str] = field(default_factory=list)
    missing_acoustic_effort_columns: list[str] = field(default_factory=list)
    geometry_status: str = "not_geospatial"
    crs: str | None = None
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_project_dataset(df, *, settings: ProjectSchemaSettings | None = None) -> dict[str, Any]:
    """Validate project-level identifiers, geospatial readiness, and acoustic metadata coverage."""
    settings = settings or ProjectSchemaSettings()
    id_column = _resolve_id_column(df, settings=settings)
    issues: list[str] = []
    warnings: list[str] = []
    duplicate_count = 0
    if id_column is None:
        issues.append("No hedgerow id column was found.")
    else:
        duplicate_count = int(df[id_column].astype("string").duplicated().sum())
        if duplicate_count:
            issues.append(f"{duplicate_count} duplicate hedgerow id row(s) found in '{id_column}'.")

    missing_project = [col for col in settings.required_project_columns if col not in df.columns]
    if missing_project:
        warnings.append("Standard project identifiers are incomplete.")

    missing_acoustic = [col for col in settings.acoustic_effort_columns if col not in df.columns]
    if missing_acoustic:
        warnings.append("Acoustic effort metadata is incomplete; post-survey comparability will be limited.")

    geometry_status, crs = _geometry_status(df)
    if geometry_status in {"missing_geometry", "invalid_geometry"}:
        issues.append(f"Geometry status is {geometry_status}.")
    if crs is None and geometry_status == "geospatial" and not settings.allow_missing_crs:
        issues.append("Geospatial dataset has no CRS.")

    if issues:
        status = "blocked"
    elif warnings:
        status = "review_required"
    else:
        status = "ready"
    return ProjectReadinessReport(
        schema_version=PROJECT_SCHEMA_VERSION,
        status=status,
        row_count=int(len(df)),
        hedgerow_id_column=id_column,
        duplicate_hedgerow_ids=duplicate_count,
        missing_project_columns=missing_project,
        missing_acoustic_effort_columns=missing_acoustic,
        geometry_status=geometry_status,
        crs=crs,
        issues=issues,
        warnings=warnings,
    ).to_dict()


def _resolve_id_column(df, *, settings: ProjectSchemaSettings) -> str | None:
    if settings.hedgerow_id_column in df.columns:
        return settings.hedgerow_id_column
    for col in settings.fallback_id_columns:
        if col in df.columns:
            return col
    return None


def _geometry_status(df) -> tuple[str, str | None]:
    geometry = getattr(df, "geometry", None)
    if geometry is None:
        return "not_geospatial", None
    if len(df) == 0:
        return "geospatial", str(getattr(df, "crs", None)) if getattr(df, "crs", None) is not None else None
    try:
        missing = bool(geometry.isna().any() or geometry.is_empty.any())
        invalid = bool((~geometry.is_valid).any())
    except Exception:
        return "missing_geometry", str(getattr(df, "crs", None)) if getattr(df, "crs", None) is not None else None
    if missing:
        return "missing_geometry", str(getattr(df, "crs", None)) if getattr(df, "crs", None) is not None else None
    if invalid:
        return "invalid_geometry", str(getattr(df, "crs", None)) if getattr(df, "crs", None) is not None else None
    return "geospatial", str(getattr(df, "crs", None)) if getattr(df, "crs", None) is not None else None
