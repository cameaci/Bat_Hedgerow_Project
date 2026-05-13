from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..deps import require_geopandas
from ..exceptions import InputValidationError
from ..utils import ensure_parent_dir, sha1_text, stable_json_dumps


EVIDENCE_PACK_VERSION = "planning_evidence_pack_v1"


def build_planning_run_summary(*, candidates_gdf, selected_gdf, settings, evidence_summary=None) -> dict[str, object]:
    payload = stable_json_dumps(
        {
            "settings": asdict(settings),
            "candidate_ids": candidates_gdf["candidate_id"].astype(str).tolist(),
            "selected_ids": selected_gdf["candidate_id"].astype(str).tolist(),
        }
    )
    summary = {
        "planning_run_id": f"plan_{sha1_text(payload, length=12)}",
        "analysis_timestamp_utc": None if settings.deterministic_output else datetime.now(timezone.utc).isoformat(),
        "settings": asdict(settings),
        "framework_versions": _framework_versions(settings),
        "guidance_regime_version": getattr(settings, "guidance_regime_version", None),
        "planning_target_scenario": getattr(settings, "target_scenario", "all_bats"),
        "candidate_count": int(len(candidates_gdf)),
        "eligible_count": int((candidates_gdf["eligible_for_selection"].astype(int) == 1).sum()),
        "selected_count": int(len(selected_gdf)),
        "selected_candidate_ids": selected_gdf["candidate_id"].astype(str).tolist(),
        "counts_by_status": {
            str(k): int(v)
            for k, v in candidates_gdf["planning_status"].astype("string").fillna("NA").value_counts().items()
        },
    }
    if evidence_summary is not None:
        summary["evidence"] = dict(evidence_summary)
    if "eco_primary_guild" in candidates_gdf.columns:
        summary["candidate_primary_guild_counts"] = {
            str(k): int(v)
            for k, v in candidates_gdf["eco_primary_guild"].astype("string").fillna("NA").value_counts().items()
        }
    if "evidence_confidence_level" in candidates_gdf.columns:
        summary["candidate_evidence_confidence_counts"] = {
            str(k): int(v)
            for k, v in candidates_gdf["evidence_confidence_level"].astype("string").fillna("NA").value_counts().items()
        }
    if "data_quality_state" in candidates_gdf.columns:
        summary["candidate_data_quality_state_counts"] = {
            str(k): int(v)
            for k, v in candidates_gdf["data_quality_state"].astype("string").fillna("NA").value_counts().items()
        }
    if "planning_target_domain_status" in candidates_gdf.columns:
        summary["candidate_target_domain_counts"] = {
            str(k): int(v)
            for k, v in candidates_gdf["planning_target_domain_status"].astype("string").fillna("NA").value_counts().items()
        }
    if "planning_priority_score" in candidates_gdf.columns:
        scores = candidates_gdf["planning_priority_score"].astype("float64")
        summary["planning_priority_score_stats"] = {
            "min": float(scores.min()) if len(scores) else None,
            "mean": float(scores.mean()) if len(scores) else None,
            "max": float(scores.max()) if len(scores) else None,
        }
    if "optimizer_strategy" in selected_gdf.columns and len(selected_gdf) > 0:
        summary["optimizer"] = {
            "strategy": str(selected_gdf["optimizer_strategy"].astype("string").dropna().iloc[0]),
            "selected_route_units": int(selected_gdf["optimization_route_unit"].astype("string").nunique()),
            "selected_corridors": int(selected_gdf["optimization_corridor_unit"].astype("string").nunique()),
            "selected_high_risk_corridors": int((selected_gdf["optimization_high_risk_flag"].astype(int) == 1).sum()),
            "selected_primary_guild_counts": {
                str(k): int(v)
                for k, v in selected_gdf["optimization_primary_guild"].astype("string").fillna("NA").value_counts().items()
            },
            "mean_marginal_gain": float(selected_gdf["optimizer_marginal_gain"].astype("float64").mean()),
            "mean_route_gain": float(selected_gdf["optimizer_gain_route_coverage"].astype("float64").mean()),
            "mean_habitat_gain": float(selected_gdf["optimizer_gain_habitat_representation"].astype("float64").mean()),
            "mean_high_risk_gain": float(selected_gdf["optimizer_gain_high_risk_coverage"].astype("float64").mean()),
            "mean_uncertainty_gain": float(selected_gdf["optimizer_gain_uncertainty_reduction"].astype("float64").mean()),
            "mean_redundancy_penalty": float(selected_gdf["optimizer_penalty_redundancy"].astype("float64").mean()),
        }
    return summary


def write_planning_outputs(
    run_result,
    output_path: str | Path,
    *,
    source_name: str = "planning_source",
    source_metadata: dict[str, Any] | None = None,
    review_summary: dict[str, Any] | None = None,
) -> dict[str, str]:
    return write_planning_evidence_pack(
        run_result,
        output_path,
        source_name=source_name,
        source_metadata=source_metadata,
        review_summary=review_summary,
    )


def write_planning_evidence_pack(
    run_result,
    output_path: str | Path,
    *,
    source_name: str = "planning_source",
    source_metadata: dict[str, Any] | None = None,
    reviewed_candidates_gdf=None,
    selected_gdf=None,
    review_summary: dict[str, Any] | None = None,
) -> dict[str, str]:
    require_geopandas()
    source_gdf = run_result.screened_gdf
    if source_gdf is None:
        raise InputValidationError("Planning evidence pack requires a screened/source hedgerow GeoDataFrame.")

    candidates_gdf = reviewed_candidates_gdf if reviewed_candidates_gdf is not None else run_result.candidates_gdf
    annotated_candidates = annotate_candidate_evidence(
        candidates_gdf,
        framework_versions=run_result.run_summary.get("framework_versions", {}),
    )
    if selected_gdf is not None:
        selected_ids = selected_gdf["candidate_id"].astype(str).tolist()
        selected_export_gdf = annotated_candidates[
            annotated_candidates["candidate_id"].astype(str).isin(selected_ids)
        ].copy()
    else:
        selected_flag_col = _selected_flag_column(annotated_candidates)
        selected_export_gdf = annotated_candidates[annotated_candidates[selected_flag_col].astype(int) == 1].copy()
    screened_export_gdf = build_screened_hedges_export(source_gdf=source_gdf, candidates_gdf=annotated_candidates)

    path = Path(output_path)
    ensure_parent_dir(path)
    if path.suffix.lower() != ".gpkg":
        raise InputValidationError("Planning evidence pack requires a .gpkg output path for the screened output.")
    screened_path = path
    candidates_path = path.with_name(f"{path.stem}_candidates.gpkg")
    selected_path = path.with_name(f"{path.stem}_selected.gpkg")
    manifest_path = path.with_name(f"{path.stem}_run_manifest.json")
    report_path = path.with_name(f"{path.stem}_evidence_report.md")
    data_catalogue_path = path.with_name(f"{path.stem}_data_catalogue.json")
    feature_health_path = path.with_name(f"{path.stem}_feature_health.json")
    method_statement_path = path.with_name(f"{path.stem}_method_statement.md")

    for existing in (screened_path, candidates_path, selected_path):
        if existing.exists():
            existing.unlink()

    _write_gpkg_layer(screened_export_gdf, screened_path, layer_name="screened_hedges", mode="w")
    _write_gpkg_layer(annotated_candidates, candidates_path, layer_name="candidate_points", mode="w")
    _write_gpkg_layer(selected_export_gdf, selected_path, layer_name="selected_detectors", mode="w")

    outputs = {
        "screened_gpkg": str(screened_path),
        "candidate_gpkg": str(candidates_path),
        "chosen_detector_set": str(selected_path),
    }
    manifest = build_planning_run_manifest(
        run_result=run_result,
        output_files=outputs,
        source_name=source_name,
        source_metadata=source_metadata,
        review_summary=review_summary,
    )
    report = build_planning_evidence_report(
        run_result=run_result,
        candidates_gdf=annotated_candidates,
        selected_gdf=selected_export_gdf,
        source_name=source_name,
        source_metadata=source_metadata,
        review_summary=review_summary,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_path.write_text(render_planning_evidence_report_markdown(report), encoding="utf-8")
    data_catalogue_path.write_text(
        json.dumps((source_metadata or {}).get("data_catalogue", {"status": "not_provided"}), indent=2),
        encoding="utf-8",
    )
    feature_health_path.write_text(
        json.dumps((source_metadata or {}).get("feature_health", {"status": "not_provided"}), indent=2),
        encoding="utf-8",
    )
    method_statement_path.write_text(_render_method_statement(report), encoding="utf-8")

    outputs["run_manifest"] = str(manifest_path)
    outputs["evidence_report"] = str(report_path)
    outputs["data_catalogue"] = str(data_catalogue_path)
    outputs["feature_health"] = str(feature_health_path)
    outputs["method_statement"] = str(method_statement_path)
    return outputs


def build_planning_run_manifest(
    *,
    run_result,
    output_files: dict[str, str],
    source_name: str,
    source_metadata: dict[str, Any] | None = None,
    review_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "manifest_version": EVIDENCE_PACK_VERSION,
        "planning_run_id": str(run_result.run_summary.get("planning_run_id", "")),
        "analysis_timestamp_utc": run_result.run_summary.get("analysis_timestamp_utc"),
        "framework_versions": dict(run_result.run_summary.get("framework_versions", {})),
        "guidance_regime_version": run_result.run_summary.get("guidance_regime_version"),
        "planning_target_scenario": run_result.run_summary.get("planning_target_scenario"),
        "source": {
            "source_name": str(source_name),
            "feature_count": int(len(run_result.screened_gdf)) if run_result.screened_gdf is not None else None,
            "working_crs": str(run_result.screened_gdf.crs) if run_result.screened_gdf is not None else None,
        },
        "dataset_provenance": source_metadata if source_metadata is not None else {"status": "not_provided"},
        "data_catalogue": (source_metadata or {}).get("data_catalogue", {"status": "not_provided"}),
        "feature_health": (source_metadata or {}).get("feature_health", {"status": "not_provided"}),
        "summary": dict(run_result.run_summary),
        "review": review_summary,
        "outputs": dict(output_files),
    }


def build_planning_evidence_report(
    *,
    run_result,
    candidates_gdf,
    selected_gdf,
    source_name: str,
    source_metadata: dict[str, Any] | None = None,
    review_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_flag_col = _selected_flag_column(candidates_gdf)
    ranked_selected = candidates_gdf[candidates_gdf[selected_flag_col].astype(int) == 1].copy()
    if not ranked_selected.empty:
        rank_col = _selected_rank_column(candidates_gdf)
        ranked_selected = ranked_selected.sort_values(rank_col, kind="mergesort")

    not_selected = candidates_gdf[candidates_gdf[selected_flag_col].astype(int) == 0].copy()
    if "planning_priority_score" in not_selected.columns and not not_selected.empty:
        not_selected = not_selected.sort_values("planning_priority_score", ascending=False, kind="mergesort")

    return {
        "report_version": EVIDENCE_PACK_VERSION,
        "planning_run_id": str(run_result.run_summary.get("planning_run_id", "")),
        "source_name": str(source_name),
        "framework_versions": dict(run_result.run_summary.get("framework_versions", {})),
        "guidance_regime_version": run_result.run_summary.get("guidance_regime_version"),
        "planning_target_scenario": run_result.run_summary.get("planning_target_scenario"),
        "dataset_provenance": source_metadata if source_metadata is not None else {"status": "not_provided"},
        "data_catalogue": (source_metadata or {}).get("data_catalogue", {"status": "not_provided"}),
        "feature_health": (source_metadata or {}).get("feature_health", {"status": "not_provided"}),
        "summary": {
            "screened_hedgerows": int(len(run_result.screened_gdf)) if run_result.screened_gdf is not None else None,
            "candidate_count": int(len(candidates_gdf)),
            "selected_count": int(len(selected_gdf)),
        },
        "review": review_summary,
        "selected_candidates": [
            _candidate_record_for_report(row, include_selected_reason=True)
            for _, row in ranked_selected.iterrows()
        ],
        "not_selected_candidates": [
            _candidate_record_for_report(row, include_selected_reason=False)
            for _, row in not_selected.head(25).iterrows()
        ],
        "not_selected_candidate_count_total": int(len(not_selected)),
    }


def render_planning_evidence_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Static Detector Planning Evidence Report",
        "",
        f"- Planning run ID: `{report.get('planning_run_id', '')}`",
        f"- Source: `{report.get('source_name', '')}`",
        f"- Evidence engine: `{report.get('framework_versions', {}).get('evidence_engine_version', 'n/a')}`",
        f"- Optimizer: `{report.get('framework_versions', {}).get('optimizer_version', 'n/a')}`",
        f"- Guidance regime: `{report.get('guidance_regime_version', 'n/a')}`",
        f"- Target scenario: `{report.get('planning_target_scenario', 'all_bats')}`",
        "",
        "## Summary",
        "",
        f"- Screened hedgerows: {report.get('summary', {}).get('screened_hedgerows')}",
        f"- Candidate locations: {report.get('summary', {}).get('candidate_count')}",
        f"- Chosen detector locations: {report.get('summary', {}).get('selected_count')}",
        "",
        "## Data Health",
        "",
        "```json",
        json.dumps((report.get("feature_health") or {}).get("summary", {"status": "not_provided"}), indent=2),
        "```",
        "",
        "## Dataset Provenance",
        "",
        "```json",
        json.dumps(report.get("dataset_provenance", {"status": "not_provided"}), indent=2),
        "```",
        "",
        "## Why Selected",
        "",
    ]
    selected_candidates = report.get("selected_candidates", [])
    if not selected_candidates:
        lines.extend(["No candidates were selected.", ""])
    for item in selected_candidates:
        lines.extend(
            [
                f"### {item['candidate_id']}",
                "",
                f"- Hedge: `{item['source_hf_uid']}`",
                f"- Decision state: `{item['decision_state']}`",
                f"- Domain status: `{item['domain_status']}`",
                f"- Why selected: {item['why_selected']}",
                f"- Missing data: {item['missing_data']}",
                f"- Confidence: {item['confidence']}",
                "",
            ]
        )

    lines.extend(["## Why Not Selected", ""])
    not_selected = report.get("not_selected_candidates", [])
    if not not_selected:
        lines.extend(["No non-selected candidates are included in this report sample.", ""])
    for item in not_selected:
        lines.extend(
            [
                f"### {item['candidate_id']}",
                "",
                f"- Hedge: `{item['source_hf_uid']}`",
                f"- Decision state: `{item['decision_state']}`",
                f"- Domain status: `{item['domain_status']}`",
                f"- Why not selected: {item['why_not_selected']}",
                f"- Missing data: {item['missing_data']}",
                f"- Confidence: {item['confidence']}",
                "",
            ]
        )
    total_non_selected = int(report.get("not_selected_candidate_count_total", 0))
    if total_non_selected > len(not_selected):
        lines.append(
            f"Only the top {len(not_selected)} non-selected candidates by planning priority score are listed here. "
            "The full candidate evidence set is in the candidate GeoPackage."
        )
        lines.append("")
    return "\n".join(lines)


def build_screened_hedges_export(*, source_gdf, candidates_gdf):
    gdf = source_gdf.copy()
    selected_flag_col = _selected_flag_column(candidates_gdf)
    records: list[dict[str, Any]] = []
    for source_hf_uid, group in candidates_gdf.groupby("source_hf_uid", dropna=False):
        ordered = group.copy()
        if "planning_priority_score" in ordered.columns:
            ordered = ordered.sort_values("planning_priority_score", ascending=False, kind="mergesort")
        top_row = ordered.iloc[0]
        selected_rows = group[group[selected_flag_col].astype(int) == 1].copy()
        decision_state = "selected" if not selected_rows.empty else (
            "ineligible" if int(group["eligible_for_selection"].astype(int).sum()) == 0 else "not_selected"
        )
        selected_candidate_ids = selected_rows["candidate_id"].astype(str).tolist()
        records.append(
            {
                "source_hf_uid": str(source_hf_uid),
                "bank_candidate_count": int(len(group)),
                "bank_eligible_candidate_count": int(group["eligible_for_selection"].astype(int).sum()),
                "bank_selected_candidate_count": int(len(selected_rows)),
                "bank_selected_candidate_ids": "|".join(selected_candidate_ids),
                "bank_decision_state": decision_state,
                "bank_max_planning_priority_score": _safe_float(top_row.get("planning_priority_score", top_row.get("candidate_score"))),
                "bank_missing_data": str(top_row.get("bank_missing_data", "")),
                "bank_confidence": str(top_row.get("bank_confidence", "")),
                "bank_domain_status": str(top_row.get("bank_domain_status", "")),
                "bank_target_scenario": str(top_row.get("bank_target_scenario", "")),
                "bank_why_selected": str(selected_rows.iloc[0].get("bank_why_selected", "")) if not selected_rows.empty else "",
                "bank_why_not_selected": str(top_row.get("bank_why_not_selected", "")) if selected_rows.empty else "",
            }
        )
    if not records:
        return gdf

    import pandas as pd

    summary_df = pd.DataFrame(records)
    return gdf.merge(summary_df, how="left", left_on="hf_uid", right_on="source_hf_uid")


def annotate_candidate_evidence(candidates_gdf, *, framework_versions: dict[str, Any]) -> Any:
    gdf = candidates_gdf.copy()
    selected_flag_col = _selected_flag_column(gdf)
    rank_col = _selected_rank_column(gdf)
    gdf["bank_decision_state"] = ""
    gdf["bank_why_selected"] = ""
    gdf["bank_why_not_selected"] = ""
    gdf["bank_missing_data"] = ""
    gdf["bank_confidence"] = ""
    gdf["bank_domain_status"] = ""
    gdf["bank_target_scenario"] = gdf.get("planning_target_scenario", "all_bats")
    gdf["bank_framework_version"] = _format_framework_versions(framework_versions)

    for idx, row in gdf.iterrows():
        is_selected = int(row.get(selected_flag_col, 0) or 0) == 1
        decision_state = _candidate_decision_state(row, selected_flag_col=selected_flag_col)
        gdf.at[idx, "bank_decision_state"] = decision_state
        gdf.at[idx, "bank_missing_data"] = _candidate_missing_data_summary(row)
        gdf.at[idx, "bank_confidence"] = _candidate_confidence_summary(row)
        gdf.at[idx, "bank_domain_status"] = str(
            row.get("planning_target_domain_status", row.get("evidence_domain_status", ""))
        )
        if is_selected:
            gdf.at[idx, "bank_why_selected"] = _candidate_why_selected(row, rank_col=rank_col)
        else:
            gdf.at[idx, "bank_why_not_selected"] = _candidate_why_not_selected(row)
    return gdf


def _candidate_record_for_report(row, *, include_selected_reason: bool) -> dict[str, Any]:
    return {
        "candidate_id": str(row.get("candidate_id", "")),
        "source_hf_uid": str(row.get("source_hf_uid", "")),
        "decision_state": str(row.get("bank_decision_state", "")),
        "domain_status": str(row.get("bank_domain_status", "")),
        "why_selected": str(row.get("bank_why_selected", "")) if include_selected_reason else "",
        "why_not_selected": str(row.get("bank_why_not_selected", "")) if not include_selected_reason else "",
        "missing_data": str(row.get("bank_missing_data", "")),
        "confidence": str(row.get("bank_confidence", "")),
    }


def _candidate_decision_state(row, *, selected_flag_col: str) -> str:
    if int(row.get(selected_flag_col, 0) or 0) == 1:
        return str(row.get("final_selection_status", "selected") or "selected")
    if str(row.get("final_selection_status", "")).strip() == "manual_removed":
        return "manual_removed"
    if int(row.get("eligible_for_selection", 0) or 0) != 1:
        return "ineligible"
    return "eligible_not_selected"


def _candidate_why_selected(row, *, rank_col: str) -> str:
    status = str(row.get("final_selection_status", "")).strip()
    rationale = str(row.get("review_override_rationale", "")).strip()
    if status == "manual_added":
        return f"Added during expert review. Rationale: {rationale or 'No rationale recorded.'}"

    contributions = [
        ("route coverage", _safe_float(row.get("optimizer_gain_route_coverage"))),
        ("habitat representation", _safe_float(row.get("optimizer_gain_habitat_representation"))),
        ("high-risk corridor coverage", _safe_float(row.get("optimizer_gain_high_risk_coverage"))),
        ("uncertainty reduction", _safe_float(row.get("optimizer_gain_uncertainty_reduction"))),
        ("base priority", _safe_float(row.get("optimizer_gain_base_score"))),
    ]
    contributions = [item for item in contributions if item[1] is not None and item[1] > 0]
    contributions.sort(key=lambda item: (-item[1], item[0]))
    top_parts = ", ".join(f"{label} ({value:.2f})" for label, value in contributions[:3]) or "overall planning priority"
    rank = _safe_int(row.get(rank_col))
    phase = str(row.get("selection_phase", "")).strip() or "planner selection"
    priority = _safe_float(row.get("planning_priority_score", row.get("candidate_score")))
    priority_text = f" Planning priority score: {priority:.3f}." if priority is not None else ""
    return f"Selected at rank {rank} during {phase}. Main gains: {top_parts}.{priority_text}"


def _candidate_why_not_selected(row) -> str:
    status = str(row.get("final_selection_status", "")).strip()
    rationale = str(row.get("review_override_rationale", "")).strip()
    if status == "manual_removed":
        return f"Removed during expert review. Rationale: {rationale or 'No rationale recorded.'}"
    if int(row.get("eligible_for_selection", 0) or 0) != 1:
        reasons = _split_codes(row.get("constraint_reason_codes"))
        if reasons:
            return "Not eligible for selection because: " + ", ".join(reasons) + "."
        return "Not eligible for selection because one or more planning constraints were not met."

    blockers: list[str] = []
    redundancy = _safe_float(row.get("optimizer_penalty_redundancy"))
    if redundancy is not None and redundancy > 0.15:
        blockers.append(f"redundancy penalty {redundancy:.2f}")
    route_gain = _safe_float(row.get("optimizer_gain_route_coverage"))
    if route_gain is not None and route_gain <= 0.05:
        blockers.append("limited additional route coverage")
    high_risk_gain = _safe_float(row.get("optimizer_gain_high_risk_coverage"))
    if high_risk_gain is not None and high_risk_gain <= 0.05:
        blockers.append("limited high-risk corridor gain")
    uncertainty_gain = _safe_float(row.get("optimizer_gain_uncertainty_reduction"))
    if uncertainty_gain is not None and uncertainty_gain <= 0.05:
        blockers.append("limited uncertainty reduction")
    if not blockers:
        blockers.append("lower marginal gain than the chosen detector set")
    return "Eligible but not selected within the detector budget due to " + ", ".join(blockers) + "."


def _candidate_missing_data_summary(row) -> str:
    missing_count = _safe_int(row.get("evidence_missing_feature_count", row.get("missing_required_feature_count")))
    coverage = _safe_float(row.get("evidence_feature_coverage_pct", row.get("gis_feature_coverage_pct")))
    reason_codes = _split_codes(row.get("evidence_reason_codes"))
    data_quality_state = str(row.get("data_quality_state", "")).strip()
    parts: list[str] = []
    if missing_count is not None:
        parts.append(f"{missing_count} missing evidence features")
    if coverage is not None:
        parts.append(f"coverage {coverage:.2%}")
    if data_quality_state:
        parts.append(f"data quality {data_quality_state}")
    if reason_codes:
        parts.append("reason codes: " + ", ".join(reason_codes[:4]))
    return "; ".join(parts) if parts else "No missing-data flags recorded in the planner evidence layer."


def _candidate_confidence_summary(row) -> str:
    level = str(row.get("evidence_confidence_level", row.get("confidence_level", "Unknown"))).strip() or "Unknown"
    score = _safe_float(row.get("evidence_confidence_score"))
    if score is None:
        return level
    return f"{level} ({score:.2f})"


def _framework_versions(settings) -> dict[str, Any]:
    return {
        "evidence_engine_version": settings.evidence_engine_version if getattr(settings, "use_evidence_engine", False) else None,
        "optimizer_version": getattr(settings, "optimizer_version", None),
        "guidance_regime_version": getattr(settings, "guidance_regime_version", None),
        "planning_target_scenario": getattr(settings, "target_scenario", "all_bats"),
    }


def _format_framework_versions(framework_versions: dict[str, Any]) -> str:
    evidence_version = framework_versions.get("evidence_engine_version") or "none"
    optimizer_version = framework_versions.get("optimizer_version") or "unknown"
    guidance_version = framework_versions.get("guidance_regime_version") or "n/a"
    target_scenario = framework_versions.get("planning_target_scenario") or "all_bats"
    return f"evidence={evidence_version}; optimizer={optimizer_version}; guidance={guidance_version}; target={target_scenario}"


def _selected_flag_column(gdf) -> str:
    return "final_selected_flag" if "final_selected_flag" in gdf.columns else "selected_flag"


def _selected_rank_column(gdf) -> str:
    return "final_selection_rank" if "final_selection_rank" in gdf.columns else "selection_rank"


def _write_gpkg_layer(gdf, out_path: Path, *, layer_name: str, mode: str) -> None:
    try:
        gdf.to_file(out_path, layer=layer_name, driver="GPKG", mode=mode, engine="fiona")
    except TypeError:
        gdf.to_file(out_path, layer=layer_name, driver="GPKG", mode=mode)


def _split_codes(raw_value) -> list[str]:
    out: list[str] = []
    if raw_value is None:
        return out
    for token in str(raw_value).split("|"):
        token = token.strip()
        if token:
            out.append(token)
    return out


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
        if result != result:
            return None
        return result
    except Exception:
        return None


def _safe_int(value) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _render_method_statement(report: dict[str, Any]) -> str:
    data_catalogue_summary = (report.get("data_catalogue") or {}).get("summary", {})
    feature_health_summary = (report.get("feature_health") or {}).get("summary", {})
    return "\n".join(
        [
            "# Method Statement",
            "",
            f"- Planning run ID: `{report.get('planning_run_id', '')}`",
            f"- Guidance regime: `{report.get('guidance_regime_version', 'n/a')}`",
            f"- Target scenario: `{report.get('planning_target_scenario', 'all_bats')}`",
            f"- Evidence engine: `{report.get('framework_versions', {}).get('evidence_engine_version', 'n/a')}`",
            f"- Optimizer: `{report.get('framework_versions', {}).get('optimizer_version', 'n/a')}`",
            "",
            "This evidence pack supports detector placement planning. It does not by itself demonstrate legal absence, impact significance, or species certainty.",
            "",
            "## Data Catalogue Summary",
            "",
            "```json",
            json.dumps(data_catalogue_summary or {"status": "not_provided"}, indent=2),
            "```",
            "",
            "## Feature Health Summary",
            "",
            "```json",
            json.dumps(feature_health_summary or {"status": "not_provided"}, indent=2),
            "```",
            "",
        ]
    )
