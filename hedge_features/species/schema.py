from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SpeciesTrainingSettings:
    species_name: str
    target_column: str
    geography_column: str | None = None
    cv_folds: int = 5
    min_positive_rows: int = 10
    max_iter: int = 400
    learning_rate: float = 0.2
    l2_strength: float = 0.01
    deterministic_output: bool = True


@dataclass(slots=True)
class SpeciesTrainingResult:
    model_artifact: dict[str, Any]
    model_card: dict[str, Any]
    domain_of_applicability: dict[str, Any]
    summary: dict[str, Any]
    predictor_df: Any
    target_series: Any
    geography_series: Any | None = None


@dataclass(slots=True)
class SpeciesPredictionSummary:
    loaded_species: list[str] = field(default_factory=list)
    counts_by_domain_status: dict[str, dict[str, int]] = field(default_factory=dict)
    mean_probability_by_species: dict[str, float] = field(default_factory=dict)
