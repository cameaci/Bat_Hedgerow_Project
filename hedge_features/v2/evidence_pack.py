from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..utils import dataframe_fingerprint, ensure_parent_dir


V2_EVIDENCE_PACK_VERSION = "bat_hedgerow_intelligence_pack_v2"


@dataclass(frozen=True, slots=True)
class V2EvidencePackSettings:
    project_name: str = "Unnamed scheme"
    analyst: str = ""
    deterministic_output: bool = True
    assumptions: tuple[str, ...] = (
        "Remote BHSA proxy scores are decision-support outputs and do not replace ecologist judgement.",
        "SI6 woody species diversity and SI7 wet ditch require field verification unless field data are supplied.",
        "No acoustic species classifier is implemented in this tool.",
    )
    method_version: str = V2_EVIDENCE_PACK_VERSION


def build_v2_evidence_pack(
    *,
    bhsa_gdf=None,
    readiness_report: dict[str, Any] | None = None,
    acoustic_summary=None,
    calibration_summary: dict[str, Any] | None = None,
    planner_summary: dict[str, Any] | None = None,
    settings: V2EvidencePackSettings | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build an inquiry-ready V2 manifest and optional report artefacts."""
    settings = settings or V2EvidencePackSettings()
    manifest = {
        "evidence_pack_version": settings.method_version,
        "created_at_utc": None if settings.deterministic_output else datetime.now(timezone.utc).isoformat(),
        "project_name": settings.project_name,
        "analyst": settings.analyst,
        "assumptions": list(settings.assumptions),
        "input_fingerprint": dataframe_fingerprint(bhsa_gdf) if bhsa_gdf is not None else None,
        "readiness": readiness_report or {},
        "bhsa": _bhsa_summary(bhsa_gdf),
        "acoustic": _table_summary(acoustic_summary),
        "calibration": calibration_summary or {},
        "planner": planner_summary or {},
    }
    method_statement = _render_method_statement(manifest, settings=settings)
    out = {"manifest": manifest, "method_statement_md": method_statement}
    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        manifest_path = path / "v2_run_manifest.json"
        method_path = path / "v2_method_statement.md"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        method_path.write_text(method_statement, encoding="utf-8")
        out["files"] = {"manifest": str(manifest_path), "method_statement": str(method_path)}
        if bhsa_gdf is not None:
            csv_path = path / "v2_bhsa_decision_table.csv"
            ensure_parent_dir(csv_path)
            bhsa_gdf.to_csv(csv_path, index=False)
            out["files"]["bhsa_decision_table"] = str(csv_path)
    return out


def _bhsa_summary(df) -> dict[str, Any]:
    if df is None:
        return {"status": "not_run"}
    summary = {"row_count": int(len(df))}
    if "bhsa_class" in df.columns:
        summary["class_counts"] = {str(k): int(v) for k, v in df["bhsa_class"].astype("string").value_counts().items()}
    if "field_verification_required" in df.columns:
        summary["field_verification_required_count"] = int(df["field_verification_required"].astype(bool).sum())
    if "bhsa_confidence_level" in df.columns:
        summary["confidence_counts"] = {
            str(k): int(v) for k, v in df["bhsa_confidence_level"].astype("string").value_counts().items()
        }
    return summary


def _table_summary(df) -> dict[str, Any]:
    if df is None:
        return {"status": "not_run"}
    return {"status": "available", "row_count": int(len(df))}


def _render_method_statement(manifest: dict[str, Any], *, settings: V2EvidencePackSettings) -> str:
    bhsa = manifest.get("bhsa", {})
    return "\n".join(
        [
            f"# Bat Hedgerow Intelligence Platform V2 Method Statement",
            "",
            f"Project: {settings.project_name}",
            f"Evidence pack version: {settings.method_version}",
            "",
            "## Decision Basis",
            "The platform applies the BHSA scoring structure to field and/or remote GIS proxy evidence, reports confidence limitations, and preserves ecologist review assumptions.",
            "",
            "## BHSA Summary",
            f"- Hedgerows assessed: {bhsa.get('row_count', 0)}",
            f"- Field verification required: {bhsa.get('field_verification_required_count', 0)}",
            f"- Class counts: {bhsa.get('class_counts', {})}",
            "",
            "## Assumptions",
            *[f"- {item}" for item in settings.assumptions],
            "",
        ]
    )
