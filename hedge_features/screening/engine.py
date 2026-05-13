from __future__ import annotations

import re
from dataclasses import asdict
from datetime import datetime, timezone

from .confidence import evaluate_row_confidence, reason_code_frequencies, reason_codes_to_pipe, strictness_profile
from .config import build_framework_snapshot, compute_profile_hash_short, load_framework_bundle
from .schema import ColumnAudit, FeatureAlignmentResult, ScreeningRunResult, ScreeningSettings
from ..species import apply_species_models
from ..utils import dataframe_fingerprint, sha1_text, stable_json_dumps


DEFAULT_FRAMEWORK_NAME = "bats_screening_v1"
LOW_CONFIDENCE_ACTION = "Ecologist review required (do not auto-deprioritise)"
PRIORITISE_ACTION = "Prioritise for statics"
DEPRIORITISE_ACTION = "Deprioritise (desk-based only for now)"


class ScreeningInputError(ValueError):
    pass


def screen_dataframe(
    df,
    *,
    framework=None,
    framework_name: str = DEFAULT_FRAMEWORK_NAME,
    framework_dir: str | None = None,
    settings: ScreeningSettings | None = None,
    prediction_route: str = "uploaded_enriched_table",
    feature_profile_name: str | None = None,
    feature_profile_version: str | None = None,
    input_label: str | None = None,
) -> ScreeningRunResult:
    import numpy as np
    import pandas as pd

    if df is None:
        raise ScreeningInputError("No input table provided.")
    if len(df) == 0:
        raise ScreeningInputError("Input table contains no rows.")

    settings = settings or ScreeningSettings()
    framework = framework or load_framework_bundle(framework_name, framework_dir=framework_dir)
    registry = framework.feature_registry
    manifest = framework.manifest

    profile_name = feature_profile_name or manifest.compatible_feature_profile_name
    profile_hash = feature_profile_version or compute_profile_hash_short(profile_name)
    warnings: list[str] = []
    _check_profile_compatibility(
        framework=framework,
        profile_name=profile_name,
        profile_hash=profile_hash,
        allow_mismatch=settings.allow_profile_mismatch,
        warnings=warnings,
    )

    audit = build_column_audit(df, framework=framework, settings=settings)
    if not audit.detected_gis_predictor_columns:
        raise ScreeningInputError(
            "No GIS feature columns from the framework registry were detected. "
            "Upload a pre-enriched CSV/XLSX or run GIS enrichment first."
        )

    alignment = align_predictors(df, framework=framework)
    X = alignment.predictor_df
    if framework.preprocessor is not None and hasattr(framework.preprocessor, "transform"):
        X_model = framework.preprocessor.transform(X)
    else:
        X_model = X

    scores = _predict_scores(framework.model, X_model)
    if len(scores) != len(df):
        raise RuntimeError("Model returned a score array with the wrong length.")

    thresholds_used = _resolve_thresholds(framework=framework, settings=settings)
    band_low_upper = float(thresholds_used["band_low_upper"])
    band_high_lower = float(thresholds_used["band_high_lower"])
    policy_threshold = float(thresholds_used["policy_threshold"])
    if not (0.0 <= band_low_upper < band_high_lower <= 1.0):
        raise ScreeningInputError("Band thresholds must satisfy 0 <= low < high <= 1.")
    if not (0.0 <= policy_threshold <= 1.0):
        raise ScreeningInputError("Policy threshold must be within [0, 1].")

    total_predictors = max(len(registry.predictor_order), 1)
    coverage_pct = X.notna().sum(axis=1).astype("float64") / float(total_predictors)
    severe_threshold = float(
        strictness_profile(framework.confidence_rules, settings.confidence_strictness).get(
            "severe_missingness_threshold",
            strictness_profile(framework.confidence_rules, settings.confidence_strictness).get(
                "low_coverage_threshold", 0.45
            ),
        )
    )
    audit.severe_missingness_rows = int((coverage_pct < severe_threshold).sum())

    required_cols = list(registry.required_predictors)
    if required_cols:
        missing_required_counts = X[required_cols].isna().sum(axis=1).astype(int)
    else:
        missing_required_counts = pd.Series(0, index=X.index, dtype="int64")

    out = df.copy()
    run_payload = stable_json_dumps(
        {
            "framework_name": manifest.name,
            "framework_version": manifest.version,
            "profile_name": profile_name,
            "profile_hash": profile_hash,
            "settings": asdict(settings),
            "input_fingerprint": dataframe_fingerprint(df, length=24),
            "predictor_fingerprint": dataframe_fingerprint(X, length=24),
        }
    )
    run_id = f"scr_{sha1_text(run_payload, length=12)}"
    ts_utc = None if settings.deterministic_output else datetime.now(timezone.utc).isoformat()

    reason_pipe_values: list[str] = []
    confidence_levels: list[str] = []
    major_reason_counts: list[int] = []

    # Row-wise confidence logic is intentionally explicit and traceable.
    for idx in out.index:
        source_row = out.loc[idx].to_dict()
        predictor_row = X.loc[idx].to_dict()
        conf = evaluate_row_confidence(
            source_row=source_row,
            predictor_row=predictor_row,
            missing_required_feature_count=int(missing_required_counts.loc[idx]),
            gis_feature_coverage_pct=float(coverage_pct.loc[idx]),
            registry=registry,
            confidence_rules=framework.confidence_rules,
            strictness=settings.confidence_strictness,
        )
        reason_pipe_values.append(reason_codes_to_pipe(conf["reason_codes"]))
        confidence_levels.append(str(conf["confidence_level"]))
        major_reason_counts.append(int(conf["major_reason_code_count"]))

    score_series = pd.Series(np.asarray(scores, dtype="float64"), index=out.index).clip(0.0, 1.0)
    band_series = score_series.apply(lambda s: _score_to_band(float(s), band_low_upper, band_high_lower))
    confidence_series = pd.Series(confidence_levels, index=out.index, dtype="object")
    action_series = _recommended_action_series(
        scores=score_series,
        confidence_levels=confidence_series,
        policy_threshold=policy_threshold,
    )

    out["framework_name"] = manifest.name
    out["framework_version"] = manifest.version
    out["feature_profile_name"] = profile_name
    out["feature_profile_version"] = profile_hash
    out["analysis_run_id"] = run_id
    out["analysis_timestamp_utc"] = ts_utc
    out["prediction_route"] = prediction_route
    out["screening_policy"] = settings.policy
    out["survey_priority_score"] = score_series
    out["survey_priority_band"] = band_series
    out["confidence_level"] = confidence_series
    out["reason_codes"] = pd.Series(reason_pipe_values, index=out.index, dtype="object")
    out["recommended_action"] = action_series
    out["gis_feature_coverage_pct"] = coverage_pct.round(4)
    out["missing_required_feature_count"] = missing_required_counts
    out["major_reason_code_count"] = pd.Series(major_reason_counts, index=out.index, dtype="int64")
    out["band_threshold_low"] = band_low_upper
    out["band_threshold_high"] = band_high_lower
    out["policy_threshold"] = policy_threshold

    species_summary = None
    if settings.species_module_enabled:
        if framework.species_models:
            out, species_summary = apply_species_models(
                out,
                species_models=framework.species_models,
                predictor_df=X,
            )
        else:
            warnings.append(
                "Species-target screening was enabled, but no trained species model artefacts were found in the framework bundle."
            )

    summary = _build_run_summary(
        result_df=out,
        framework=framework,
        settings=settings,
        thresholds_used=thresholds_used,
        input_label=input_label,
        column_audit=audit,
        warnings=warnings,
        alignment=alignment,
        species_summary=species_summary,
    )
    return ScreeningRunResult(results_df=out, column_audit=audit, run_summary=summary, warnings=warnings)


def build_column_audit(df, *, framework, settings: ScreeningSettings) -> ColumnAudit:
    registry = framework.feature_registry
    excluded_columns: dict[str, str] = {}
    status_columns_found = _detect_status_columns(df.columns, registry=registry)
    coverage_flags_found = [c for c in registry.coverage_flags if c in df.columns]

    forbidden_exact = set(registry.forbidden_exact or [])
    forbidden_prefixes = tuple(registry.forbidden_prefixes or [])
    forbidden_patterns = [re.compile(pat) for pat in (registry.forbidden_regex or [])]
    survey_patterns = [
        re.compile(r"^(Autumn|Spring|Summer)\d{0,4}_", flags=re.IGNORECASE),
        re.compile(r"_SurveyFlag$", flags=re.IGNORECASE),
        re.compile(r"_Survey nights$", flags=re.IGNORECASE),
    ]

    for col in df.columns:
        if col in forbidden_exact:
            excluded_columns[col] = "forbidden_exact"
            continue
        if any(col.startswith(pfx) for pfx in forbidden_prefixes):
            excluded_columns[col] = "forbidden_prefix"
            continue
        if col in status_columns_found:
            excluded_columns[col] = "status_confidence_only"
            continue
        if col in coverage_flags_found:
            excluded_columns[col] = "coverage_flag_confidence_only"
            continue
        if any(p.search(col) for p in forbidden_patterns):
            excluded_columns[col] = "forbidden_regex"
            continue
        if col.startswith("Static_"):
            excluded_columns[col] = "field_survey_non_transfer"
            continue
        if col.startswith("HyNet_"):
            excluded_columns[col] = "project_admin_non_transfer"
            continue
        if any(p.search(col) for p in survey_patterns):
            excluded_columns[col] = "survey_label_or_metadata"
            continue

    present_predictors = _detect_present_registry_predictors(df, registry=registry)
    missing_expected = [c for c in registry.predictor_order if c not in present_predictors]
    missing_required = [c for c in registry.required_predictors if c not in present_predictors]

    extra_columns: list[str] = []
    predictor_set = set(registry.predictor_order)
    for col in df.columns:
        if col in excluded_columns:
            continue
        if col in predictor_set:
            continue
        if not settings.strict_gis_only:
            # In relaxed mode we still score on the versioned registry, but report extras explicitly.
            if _matches_any_prefix(col, registry.strict_gis_prefixes) or col.endswith("_status"):
                extra_columns.append(col)
                continue
            if col not in {"geometry"}:
                extra_columns.append(col)
                continue
        else:
            if col not in {"geometry"}:
                extra_columns.append(col)

    return ColumnAudit(
        total_columns=int(len(df.columns)),
        total_rows=int(len(df)),
        detected_gis_predictor_columns=present_predictors,
        missing_expected_features=missing_expected,
        missing_required_features=missing_required,
        extra_columns=sorted(extra_columns),
        excluded_columns=dict(sorted(excluded_columns.items())),
        status_columns_found=status_columns_found,
        coverage_flags_found=coverage_flags_found,
        severe_missingness_rows=0,
        gis_feature_families_detected=_family_counts(present_predictors),
    )


def align_predictors(df, *, framework) -> FeatureAlignmentResult:
    import pandas as pd

    registry = framework.feature_registry
    aligned = pd.DataFrame(index=df.index)
    source_map: dict[str, str] = {}
    missing_required: list[str] = []
    missing_optional: list[str] = []

    for predictor in registry.predictor_order:
        source_col = _find_source_column_for_predictor(df.columns, predictor, registry.alias_map)
        if source_col is None:
            aligned[predictor] = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
            if predictor in registry.required_predictors:
                missing_required.append(predictor)
            else:
                missing_optional.append(predictor)
        else:
            aligned[predictor] = df[source_col]
            source_map[predictor] = source_col

    return FeatureAlignmentResult(
        predictor_df=aligned,
        source_column_for_predictor=source_map,
        missing_required_features=missing_required,
        missing_optional_features=missing_optional,
    )


def _predict_scores(model, X):
    import numpy as np

    if hasattr(model, "predict_score"):
        scores = model.predict_score(X)
    elif hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if getattr(proba, "shape", (0, 0))[1] < 2:
            raise RuntimeError("predict_proba returned fewer than 2 columns.")
        scores = proba[:, 1]
    elif hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype="float64")
        scores = 1.0 / (1.0 + np.exp(-np.clip(raw, -20, 20)))
    elif hasattr(model, "predict"):
        scores = np.asarray(model.predict(X), dtype="float64")
    else:
        raise RuntimeError("Loaded model artefact does not support predict_score / predict_proba / decision_function / predict.")
    return np.asarray(scores, dtype="float64").clip(0.0, 1.0)


def _resolve_thresholds(*, framework, settings: ScreeningSettings) -> dict[str, Any]:
    thresholds = framework.thresholds
    policy_name = settings.policy
    if policy_name not in thresholds.policy_thresholds:
        raise ScreeningInputError(
            f"Unknown screening policy '{policy_name}'. Available: {sorted(thresholds.policy_thresholds)}"
        )
    band_low = thresholds.band_low_upper
    band_high = thresholds.band_high_lower
    if not settings.use_packaged_band_thresholds:
        if settings.custom_band_low_upper is None or settings.custom_band_high_lower is None:
            raise ScreeningInputError("Custom band thresholds require both low and high values.")
        band_low = float(settings.custom_band_low_upper)
        band_high = float(settings.custom_band_high_lower)
    policy_threshold = thresholds.policy_thresholds[policy_name]
    if settings.custom_policy_threshold is not None:
        policy_threshold = float(settings.custom_policy_threshold)
    return {
        "band_low_upper": float(band_low),
        "band_high_lower": float(band_high),
        "policy_threshold": float(policy_threshold),
        "policy_name": policy_name,
        "band_threshold_source": "packaged_defaults" if settings.use_packaged_band_thresholds else "custom",
    }


def _score_to_band(score: float, band_low_upper: float, band_high_lower: float) -> str:
    if score < band_low_upper:
        return "Low"
    if score < band_high_lower:
        return "Medium"
    return "High"


def _recommended_action_series(*, scores, confidence_levels, policy_threshold: float):
    import pandas as pd

    actions: list[str] = []
    for idx in scores.index:
        conf = str(confidence_levels.loc[idx])
        if conf == "Low":
            actions.append(LOW_CONFIDENCE_ACTION)
            continue
        score = float(scores.loc[idx])
        if score >= policy_threshold:
            actions.append(PRIORITISE_ACTION)
        else:
            actions.append(DEPRIORITISE_ACTION)
    return pd.Series(actions, index=scores.index, dtype="object")


def _build_run_summary(
    *,
    result_df,
    framework,
    settings: ScreeningSettings,
    thresholds_used: dict[str, Any],
    input_label: str | None,
    column_audit: ColumnAudit,
    warnings: list[str],
    alignment: FeatureAlignmentResult,
    species_summary=None,
) -> dict[str, Any]:
    from hedge_features import __version__ as app_version

    summary = {
        "app_version": app_version,
        "analysis_timestamp_utc": (
            None if settings.deterministic_output else datetime.now(timezone.utc).isoformat()
        ),
        "framework": build_framework_snapshot(framework),
        "input": {
            "input_label": input_label,
            "row_count": int(len(result_df)),
            "column_count": int(len(result_df.columns)),
        },
        "selected_settings": asdict(settings),
        "thresholds_used": dict(thresholds_used),
        "column_audit": column_audit.to_dict(),
        "alignment": {
            "source_column_for_predictor": alignment.source_column_for_predictor,
            "missing_required_features": alignment.missing_required_features,
            "missing_optional_features_count": len(alignment.missing_optional_features),
        },
        "counts_by_band": _series_counts(result_df["survey_priority_band"]),
        "counts_by_action": _series_counts(result_df["recommended_action"]),
        "counts_by_confidence": _series_counts(result_df["confidence_level"]),
        "reason_code_frequencies": reason_code_frequencies(result_df["reason_codes"]),
        "warnings": list(warnings),
    }
    if species_summary is not None:
        summary["species_models"] = {
            "loaded_species": list(species_summary.loaded_species),
            "counts_by_domain_status": dict(species_summary.counts_by_domain_status),
            "mean_probability_by_species": dict(species_summary.mean_probability_by_species),
        }
    return summary


def _series_counts(series) -> dict[str, int]:
    counts = series.astype("string").fillna("NA").value_counts(dropna=False)
    return {str(idx): int(val) for idx, val in counts.items()}


def _check_profile_compatibility(
    *,
    framework,
    profile_name: str,
    profile_hash: str | None,
    allow_mismatch: bool,
    warnings: list[str],
) -> None:
    manifest = framework.manifest
    if profile_name != manifest.compatible_feature_profile_name:
        msg = (
            f"Framework {manifest.name} expects feature profile '{manifest.compatible_feature_profile_name}', "
            f"but '{profile_name}' was selected."
        )
        if allow_mismatch:
            warnings.append(msg)
        else:
            raise ScreeningInputError(msg + " Use Advanced mode to allow profile mismatch.")
    expected_hash = manifest.compatible_feature_profile_hash
    if expected_hash and profile_hash and profile_hash != expected_hash:
        msg = (
            f"Feature profile hash mismatch for {profile_name}: framework expects {expected_hash}, got {profile_hash}."
        )
        if allow_mismatch:
            warnings.append(msg)
        else:
            raise ScreeningInputError(msg + " Use Advanced mode to allow profile mismatch.")


def _detect_present_registry_predictors(df, *, registry) -> list[str]:
    present: list[str] = []
    for predictor in registry.predictor_order:
        if _find_source_column_for_predictor(df.columns, predictor, registry.alias_map) is not None:
            present.append(predictor)
    return present


def _find_source_column_for_predictor(columns, predictor: str, alias_map: dict[str, str] | None = None) -> str | None:
    col_set = set(columns)
    if predictor in col_set:
        return predictor
    alias_map = alias_map or {}
    # Support both {canonical: alias} and {alias: canonical} mappings.
    direct_alias = alias_map.get(predictor)
    if direct_alias and direct_alias in col_set:
        return direct_alias
    for alias, canonical in alias_map.items():
        if canonical == predictor and alias in col_set:
            return alias
    return None


def _detect_status_columns(columns, *, registry) -> list[str]:
    compiled = [re.compile(pat) for pat in (registry.status_regex or [])]
    out: list[str] = []
    for col in columns:
        if col in (registry.status_columns or []):
            out.append(col)
            continue
        if any(p.search(col) for p in compiled):
            out.append(col)
    return sorted(set(out))


def _matches_any_prefix(col: str, prefixes: list[str]) -> bool:
    return any(col.startswith(pfx) for pfx in (prefixes or []))


def _family_counts(columns: list[str]) -> dict[str, int]:
    families = {
        "geom": 0,
        "net": 0,
        "dist": 0,
        "buf": 0,
        "pt": 0,
        "roostpx": 0,
        "mhb": 0,
    }
    for col in columns:
        if col.startswith("geom_"):
            families["geom"] += 1
        elif col.startswith("net_"):
            families["net"] += 1
        elif col.startswith("dist_"):
            families["dist"] += 1
        elif col.startswith("buf"):
            families["buf"] += 1
        elif col.startswith("pt_"):
            families["pt"] += 1
        elif col.startswith("roostpx_"):
            families["roostpx"] += 1
        elif col.startswith("mhb_"):
            families["mhb"] += 1
    return families
