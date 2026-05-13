from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class FrameworkManifest:
    name: str
    version: str
    created_date: str
    compatible_feature_profile_name: str
    compatible_feature_profile_version: str | None = None
    compatible_feature_profile_hash: str | None = None
    notes: str | None = None
    feature_registry_artifact: str = "feature_registry.json"
    thresholds_artifact: str = "triage_thresholds.json"
    confidence_rules_artifact: str = "confidence_rules.json"
    model_artifact: str = "triage_model.json"
    preprocessor_artifact: str | None = None
    species_models_dir: str = "species_models"
    species_targets_available: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameworkManifest":
        return cls(
            name=str(data["name"]),
            version=str(data["version"]),
            created_date=str(data.get("created_date", "")),
            compatible_feature_profile_name=str(data["compatible_feature_profile_name"]),
            compatible_feature_profile_version=_as_opt_str(data.get("compatible_feature_profile_version")),
            compatible_feature_profile_hash=_as_opt_str(data.get("compatible_feature_profile_hash")),
            notes=_as_opt_str(data.get("notes")),
            feature_registry_artifact=str(data.get("feature_registry_artifact", "feature_registry.json")),
            thresholds_artifact=str(data.get("thresholds_artifact", "triage_thresholds.json")),
            confidence_rules_artifact=str(data.get("confidence_rules_artifact", "confidence_rules.json")),
            model_artifact=str(data.get("model_artifact", "triage_model.json")),
            preprocessor_artifact=_as_opt_str(data.get("preprocessor_artifact")),
            species_models_dir=str(data.get("species_models_dir", "species_models")),
            species_targets_available=[str(x) for x in (data.get("species_targets_available") or [])],
        )


@dataclass(slots=True)
class FeatureRegistry:
    framework_name: str
    version: str
    compatible_feature_profile_name: str
    predictor_order: list[str]
    required_predictors: list[str]
    strict_gis_prefixes: list[str]
    forbidden_exact: list[str] = field(default_factory=list)
    forbidden_prefixes: list[str] = field(default_factory=list)
    forbidden_regex: list[str] = field(default_factory=list)
    status_columns: list[str] = field(default_factory=list)
    status_regex: list[str] = field(default_factory=list)
    coverage_flags: list[str] = field(default_factory=list)
    alias_map: dict[str, str] = field(default_factory=dict)
    outlier_bounds: dict[str, dict[str, float | None]] = field(default_factory=dict)
    notes: str | None = None

    @property
    def optional_predictors(self) -> list[str]:
        required = set(self.required_predictors)
        return [c for c in self.predictor_order if c not in required]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureRegistry":
        return cls(
            framework_name=str(data["framework_name"]),
            version=str(data.get("version", "1")),
            compatible_feature_profile_name=str(data.get("compatible_feature_profile_name", "bats_v1")),
            predictor_order=[str(x) for x in data.get("predictor_order", [])],
            required_predictors=[str(x) for x in data.get("required_predictors", [])],
            strict_gis_prefixes=[str(x) for x in data.get("strict_gis_prefixes", [])],
            forbidden_exact=[str(x) for x in data.get("forbidden_exact", [])],
            forbidden_prefixes=[str(x) for x in data.get("forbidden_prefixes", [])],
            forbidden_regex=[str(x) for x in data.get("forbidden_regex", [])],
            status_columns=[str(x) for x in data.get("status_columns", [])],
            status_regex=[str(x) for x in data.get("status_regex", [])],
            coverage_flags=[str(x) for x in data.get("coverage_flags", [])],
            alias_map={str(k): str(v) for k, v in (data.get("alias_map") or {}).items()},
            outlier_bounds={
                str(k): {
                    "min": _as_opt_float((v or {}).get("min")),
                    "max": _as_opt_float((v or {}).get("max")),
                }
                for k, v in (data.get("outlier_bounds") or {}).items()
            },
            notes=_as_opt_str(data.get("notes")),
        )


@dataclass(slots=True)
class ThresholdConfig:
    version: str
    band_low_upper: float
    band_high_lower: float
    policy_thresholds: dict[str, float]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThresholdConfig":
        band_cfg = data.get("band_thresholds") or {}
        return cls(
            version=str(data.get("version", "1")),
            band_low_upper=float(band_cfg.get("low_upper", 0.4)),
            band_high_lower=float(band_cfg.get("high_lower", 0.7)),
            policy_thresholds={str(k): float(v) for k, v in (data.get("policy_thresholds") or {}).items()},
        )


@dataclass(slots=True)
class ConfidenceRules:
    version: str
    major_reason_codes: list[str]
    strictness_profiles: dict[str, dict[str, Any]]
    default_strictness: str = "Standard"
    status_ok_values: list[str] = field(default_factory=lambda: ["ok"])
    merge_left_only_values: list[str] = field(default_factory=lambda: ["left_only"])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfidenceRules":
        return cls(
            version=str(data.get("version", "1")),
            major_reason_codes=[str(x) for x in data.get("major_reason_codes", [])],
            strictness_profiles={
                str(k): dict(v or {})
                for k, v in (data.get("strictness_profiles") or {}).items()
            },
            default_strictness=str(data.get("default_strictness", "Standard")),
            status_ok_values=[str(x).lower() for x in data.get("status_ok_values", ["ok"])],
            merge_left_only_values=[str(x).lower() for x in data.get("merge_left_only_values", ["left_only"])],
        )


@dataclass(slots=True)
class ScreeningSettings:
    mode: str = "Default"
    policy: str = "Recall-first"
    confidence_strictness: str = "Standard"
    strict_gis_only: bool = True
    predictor_inclusion_mode: str = "Strict GIS-only"
    use_packaged_band_thresholds: bool = True
    custom_band_low_upper: float | None = None
    custom_band_high_lower: float | None = None
    custom_policy_threshold: float | None = None
    allow_profile_mismatch: bool = False
    species_module_enabled: bool = False
    deterministic_output: bool = True


@dataclass(slots=True)
class ColumnAudit:
    total_columns: int
    total_rows: int
    detected_gis_predictor_columns: list[str]
    missing_expected_features: list[str]
    missing_required_features: list[str]
    extra_columns: list[str]
    excluded_columns: dict[str, str]
    status_columns_found: list[str]
    coverage_flags_found: list[str]
    severe_missingness_rows: int
    gis_feature_families_detected: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_columns": self.total_columns,
            "total_rows": self.total_rows,
            "detected_gis_predictor_columns": self.detected_gis_predictor_columns,
            "missing_expected_features": self.missing_expected_features,
            "missing_required_features": self.missing_required_features,
            "extra_columns": self.extra_columns,
            "excluded_columns": self.excluded_columns,
            "status_columns_found": self.status_columns_found,
            "coverage_flags_found": self.coverage_flags_found,
            "severe_missingness_rows": self.severe_missingness_rows,
            "gis_feature_families_detected": self.gis_feature_families_detected,
        }


@dataclass(slots=True)
class FeatureAlignmentResult:
    predictor_df: Any
    source_column_for_predictor: dict[str, str]
    missing_required_features: list[str]
    missing_optional_features: list[str]


@dataclass(slots=True)
class LoadedFramework:
    root_dir: Path
    manifest: FrameworkManifest
    feature_registry: FeatureRegistry
    thresholds: ThresholdConfig
    confidence_rules: ConfidenceRules
    model: Any
    preprocessor: Any | None = None
    species_models: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScreeningRunResult:
    results_df: Any
    column_audit: ColumnAudit
    run_summary: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def _as_opt_str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _as_opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None
