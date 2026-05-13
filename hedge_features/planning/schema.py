from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PlanningSettings:
    detector_budget: int
    optimizer_version: str = "greedy_coverage_v1"
    candidate_spacing_m: float = 100.0
    endpoint_offset_m: float = 20.0
    min_detector_spacing_m: float = 150.0
    score_column: str | None = None
    use_evidence_engine: bool = True
    evidence_score_column: str = "planning_priority_score"
    evidence_engine_version: str = "bankable_species_v1"
    target_scenario: str = "all_bats"
    guidance_regime_version: str = "bct4_ne2025_england_v1"
    min_score: float | None = None
    access_flag_column: str | None = None
    section_column: str | None = None
    section_minimum_counts: dict[str, int] = field(default_factory=dict)
    objective_weight_base_score: float = 0.30
    objective_weight_habitat_representation: float = 0.22
    objective_weight_route_coverage: float = 0.22
    objective_weight_high_risk_coverage: float = 0.16
    objective_weight_uncertainty_reduction: float = 0.10
    objective_weight_redundancy_penalty: float = 0.18
    high_risk_quantile: float = 0.80
    soft_spacing_multiplier: float = 3.0
    reject_overlit_candidates: bool = True
    reject_low_confidence_candidates: bool = False
    lighting_risk_threshold: float = 0.70
    deterministic_output: bool = True


@dataclass(slots=True)
class PlanningRunResult:
    candidates_gdf: Any
    selected_gdf: Any
    run_summary: dict[str, Any]
    screened_gdf: Any | None = None
