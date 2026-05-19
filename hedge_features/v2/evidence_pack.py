from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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
    validation_diagnostics: dict[str, Any] | None = None,
    calibration_summary: dict[str, Any] | None = None,
    planner_summary: dict[str, Any] | None = None,
    detector_deployment=None,
    reviewer_override_log=None,
    settings: V2EvidencePackSettings | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build regulator-ready V2 evidence artefacts from app/service decisions."""
    settings = settings or V2EvidencePackSettings()
    tables = {
        "bhsa_decision_table": _decision_table(bhsa_gdf),
        "field_verification_table": _field_verification_table(bhsa_gdf),
        "acoustic_validation_summary": _validation_summary_table(validation_diagnostics),
        "detector_deployment_rationale": _detector_deployment_table(detector_deployment, planner_summary=planner_summary),
        "reviewer_override_log": _override_log_table(reviewer_override_log),
    }
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
        "validation": validation_diagnostics or {"status": "not_run"},
        "calibration": calibration_summary or {"status": "not_run"},
        "planner": planner_summary or {"status": "not_run"},
        "artefacts": {
            "method_statement": "v2_method_statement.md",
            "run_manifest": "v2_run_manifest.json",
            "bhsa_decision_table": "v2_bhsa_decision_table.csv",
            "field_verification_table": "v2_field_verification_table.csv",
            "acoustic_validation_summary_csv": "v2_acoustic_validation_summary.csv",
            "acoustic_validation_summary_json": "v2_acoustic_validation_summary.json",
            "detector_deployment_rationale": "v2_detector_deployment_rationale.csv",
        },
    }
    method_statement = _render_method_statement(manifest, settings=settings)
    out = {"manifest": manifest, "method_statement_md": method_statement, "tables": tables}
    if output_dir is not None:
        out["files"] = _write_evidence_pack(Path(output_dir), manifest, method_statement, tables, validation_diagnostics)
    return out


def _write_evidence_pack(
    path: Path,
    manifest: dict[str, Any],
    method_statement: str,
    tables: dict[str, Any],
    validation_diagnostics: dict[str, Any] | None,
) -> dict[str, str]:
    path.mkdir(parents=True, exist_ok=True)
    files = {
        "manifest": path / "v2_run_manifest.json",
        "method_statement": path / "v2_method_statement.md",
        "bhsa_decision_table": path / "v2_bhsa_decision_table.csv",
        "field_verification_table": path / "v2_field_verification_table.csv",
        "acoustic_validation_summary_csv": path / "v2_acoustic_validation_summary.csv",
        "acoustic_validation_summary_json": path / "v2_acoustic_validation_summary.json",
        "detector_deployment_rationale": path / "v2_detector_deployment_rationale.csv",
    }
    files["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    files["method_statement"].write_text(method_statement, encoding="utf-8")
    for key, table in (
        ("bhsa_decision_table", tables["bhsa_decision_table"]),
        ("field_verification_table", tables["field_verification_table"]),
        ("acoustic_validation_summary_csv", tables["acoustic_validation_summary"]),
        ("detector_deployment_rationale", tables["detector_deployment_rationale"]),
    ):
        ensure_parent_dir(files[key])
        table.to_csv(files[key], index=False)
    files["acoustic_validation_summary_json"].write_text(
        json.dumps(validation_diagnostics or {"status": "not_run"}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {key: str(value) for key, value in files.items()}


def _decision_table(df):
    import pandas as pd

    columns = [
        "project_id",
        "scheme_name",
        "section_id",
        "hedgerow_id",
        "hf_uid",
        "bhsa_score",
        "bhsa_class",
        "bhsa_survey_requirement",
        "bhsa_confidence_level",
        "field_verification_required",
        "bhsa_missing_reasons",
        "bhsa_notes",
        "bhsa_major_road_downgraded",
        "bhsa_adjustment_applied",
    ]
    si_columns = []
    for idx in range(1, 8):
        si_columns.extend([f"bhsa_si{idx}_score", f"bhsa_si{idx}_source", f"bhsa_si{idx}_confidence"])
    columns.extend(si_columns)
    if df is None:
        return pd.DataFrame(columns=columns)
    present = [col for col in columns if col in df.columns]
    return pd.DataFrame(df[present]).copy()


def _field_verification_table(df):
    import pandas as pd

    columns = [
        "hedgerow_id",
        "hf_uid",
        "section_id",
        "bhsa_class",
        "bhsa_confidence_level",
        "field_verification_required",
        "verification_reason",
        "bhsa_missing_reasons",
        "bhsa_notes",
        "bhsa_si6_source",
        "bhsa_si6_confidence",
        "bhsa_si7_source",
        "bhsa_si7_confidence",
    ]
    if df is None or "field_verification_required" not in df.columns:
        return pd.DataFrame(columns=columns)
    review = df.loc[df["field_verification_required"].astype(bool)].copy()
    if review.empty:
        return pd.DataFrame(columns=columns)
    review["verification_reason"] = review.apply(_verification_reason, axis=1)
    present = [col for col in columns if col in review.columns]
    return pd.DataFrame(review[present]).copy()


def _validation_summary_table(validation_diagnostics: dict[str, Any] | None):
    import pandas as pd

    columns = [
        "status",
        "sample_size",
        "true_positive",
        "false_positive_high_score_no_evidence",
        "true_negative",
        "false_negative_low_score_positive_evidence",
        "sensitivity",
        "specificity",
        "precision",
        "accuracy",
        "auc",
        "caveats",
    ]
    if not validation_diagnostics:
        return pd.DataFrame([{"status": "not_run"}], columns=columns)
    matrix = validation_diagnostics.get("confusion_matrix", {})
    metrics = validation_diagnostics.get("metrics", {})
    row = {
        "status": validation_diagnostics.get("status", "unknown"),
        "sample_size": validation_diagnostics.get("sample_size", 0),
        "true_positive": matrix.get("true_positive"),
        "false_positive_high_score_no_evidence": matrix.get("false_positive_high_score_no_evidence"),
        "true_negative": matrix.get("true_negative"),
        "false_negative_low_score_positive_evidence": matrix.get("false_negative_low_score_positive_evidence"),
        "sensitivity": metrics.get("sensitivity"),
        "specificity": metrics.get("specificity"),
        "precision": metrics.get("precision"),
        "accuracy": metrics.get("accuracy"),
        "auc": metrics.get("auc"),
        "caveats": " | ".join(validation_diagnostics.get("caveats", [])),
    }
    return pd.DataFrame([row], columns=columns)


def _detector_deployment_table(detector_deployment, *, planner_summary: dict[str, Any] | None):
    import pandas as pd

    columns = [
        "hedgerow_id",
        "hf_uid",
        "section_id",
        "bhsa_score",
        "priority_score",
        "selected",
        "detector_rationale",
    ]
    if detector_deployment is not None and hasattr(detector_deployment, "columns"):
        present = [col for col in columns if col in detector_deployment.columns]
        table = pd.DataFrame(detector_deployment[present]).copy() if present else pd.DataFrame(detector_deployment).copy()
        if "selected" not in table.columns:
            table["selected"] = True
        if "detector_rationale" not in table.columns:
            table["detector_rationale"] = "Selected by detector deployment optimiser for survey design review."
        return table
    if not planner_summary:
        return pd.DataFrame(columns=columns)
    row = {
        "detector_rationale": json.dumps(planner_summary, sort_keys=True),
        "selected": None,
    }
    return pd.DataFrame([row])


def _override_log_table(reviewer_override_log):
    import pandas as pd

    columns = ["hedgerow_id", "reviewer", "override_type", "original_value", "override_value", "reason"]
    if reviewer_override_log is None:
        return pd.DataFrame(columns=columns)
    if hasattr(reviewer_override_log, "columns"):
        present = [col for col in columns if col in reviewer_override_log.columns]
        return pd.DataFrame(reviewer_override_log[present]).copy()
    return pd.DataFrame(reviewer_override_log, columns=columns)


def _verification_reason(row) -> str:
    reasons = []
    missing = str(row.get("bhsa_missing_reasons", "") or "").strip("|")
    if missing:
        reasons.append(f"Missing BHSA evidence: {missing}")
    if row.get("bhsa_si6_source") != "field":
        reasons.append("SI6 woody species diversity cannot be remotely verified.")
    if row.get("bhsa_si7_source") != "field":
        reasons.append("SI7 wet ditch cannot be remotely verified.")
    notes = str(row.get("bhsa_notes", "") or "").replace("|", "; ")
    if notes:
        reasons.append(notes)
    return " ".join(reasons) if reasons else "Confidence-limited BHSA output requires ecologist review."


def _bhsa_summary(df) -> dict[str, Any]:
    if df is None:
        return {"status": "not_run"}
    summary = {"status": "available", "row_count": int(len(df))}
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
    validation = manifest.get("validation", {})
    calibration = manifest.get("calibration", {})
    planner = manifest.get("planner", {})
    return "\n".join(
        [
            "# Bat Hedgerow Intelligence Platform V2 Method Statement",
            "",
            f"Project: {settings.project_name}",
            f"Analyst / reviewer: {settings.analyst or 'Not recorded'}",
            f"Evidence pack version: {settings.method_version}",
            "",
            "## Decision Basis",
            "The assessment applies the BHSA SI1-SI7 scoring structure to supplied field evidence and/or desk-based GIS proxy evidence. Outputs are framed as survey-effort decisions, confidence limitations, and field-verification requirements for bat specialist review.",
            "",
            "## Remote BHSA Method",
            "Field BHSA values are used where supplied. GIS proxy values are used only as remote decision-support evidence. SI6 woody species diversity and SI7 wet ditch are flagged for field verification unless field evidence is present.",
            f"Hedgerows assessed: {bhsa.get('row_count', 0)}",
            f"BHSA class counts: {bhsa.get('class_counts', {})}",
            f"Field verification required: {bhsa.get('field_verification_required_count', 0)}",
            "",
            "## Acoustic Evidence",
            "The platform accepts structured static detector result tables containing hedgerow, night, season, species or species-group, pass count, detector metadata, and QA status. It does not classify audio files.",
            f"Acoustic summary status: {manifest.get('acoustic', {}).get('status', 'not_run')}",
            "",
            "## Validation Diagnostics",
            f"Validation status: {validation.get('status', 'not_run')}",
            f"Validation sample size: {validation.get('sample_size', 0)}",
            f"Validation metrics: {validation.get('metrics', {})}",
            "",
            "## Calibration Position",
            f"Calibration status: {calibration.get('status', 'not_run')}",
            f"Do not use calibrated model: {calibration.get('do_not_use_calibrated_model', 'not_applicable')}",
            "Equal-prior BHSA weighting remains the default unless the calibration output has adequate sample size, class balance, and technical review.",
            "",
            "## Detector Deployment Rationale",
            "Detector deployment outputs should be reviewed as survey design recommendations. Selected locations, non-selected high-priority locations, access constraints, and reviewer overrides should be retained in the evidence pack.",
            f"Planner status: {planner.get('status', 'available' if planner else 'not_run')}",
            "",
            "## Assumptions And Limitations",
            *[f"- {item}" for item in settings.assumptions],
            "- Absence of acoustic evidence in a supplied table is not proof of absence unless survey effort and QA are demonstrably comparable.",
            "- Professional judgement overrides should be recorded with the reason and reviewer.",
            "",
        ]
    )
