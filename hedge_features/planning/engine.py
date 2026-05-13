from __future__ import annotations

from dataclasses import replace

from .candidates import build_candidate_points
from .constraints import apply_planning_constraints
from .evidence import add_planning_evidence_scores
from .optimizer import select_detector_locations
from .reporting import build_planning_run_summary
from .schema import PlanningRunResult


def plan_static_detectors(
    hedges_gdf,
    *,
    settings,
    include_area_gdf=None,
    exclude_area_gdf=None,
    hedge_id_column: str = "hf_uid",
) -> PlanningRunResult:
    working_gdf = hedges_gdf.copy()
    evidence_summary = None
    if settings.use_evidence_engine:
        evidence_result = add_planning_evidence_scores(working_gdf, settings=settings)
        working_gdf = evidence_result.gdf
        evidence_summary = evidence_result.summary
    effective_score_column = settings.score_column or (
        settings.evidence_score_column if settings.use_evidence_engine else None
    )

    if effective_score_column and effective_score_column not in working_gdf.columns:
        raise ValueError(f"Score column '{effective_score_column}' was not found in the hedgerow input.")
    if settings.access_flag_column and settings.access_flag_column not in hedges_gdf.columns:
        raise ValueError(f"Access flag column '{settings.access_flag_column}' was not found in the hedgerow input.")
    if settings.section_column and settings.section_column not in hedges_gdf.columns:
        raise ValueError(f"Section column '{settings.section_column}' was not found in the hedgerow input.")
    candidate_settings = replace(settings, score_column=effective_score_column)

    candidates = build_candidate_points(
        working_gdf,
        settings=candidate_settings,
        hedge_id_column=hedge_id_column,
    )
    candidates = apply_planning_constraints(
        candidates,
        settings=candidate_settings,
        include_area_gdf=include_area_gdf,
        exclude_area_gdf=exclude_area_gdf,
    )
    selected, candidates = select_detector_locations(candidates, settings=candidate_settings)
    summary = build_planning_run_summary(
        candidates_gdf=candidates,
        selected_gdf=selected,
        settings=candidate_settings,
        evidence_summary=evidence_summary,
    )
    return PlanningRunResult(
        candidates_gdf=candidates,
        selected_gdf=selected,
        run_summary=summary,
        screened_gdf=working_gdf,
    )
