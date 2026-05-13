from __future__ import annotations

from typing import Any


def strictness_profile(confidence_rules, strictness: str | None = None) -> dict[str, Any]:
    requested = (strictness or confidence_rules.default_strictness or "Standard").strip()
    profiles = confidence_rules.strictness_profiles or {}
    if requested in profiles:
        return dict(profiles[requested])
    for key, value in profiles.items():
        if key.lower() == requested.lower():
            return dict(value)
    if "Standard" in profiles:
        return dict(profiles["Standard"])
    if profiles:
        _, value = next(iter(profiles.items()))
        return dict(value)
    return {}


def evaluate_row_confidence(
    *,
    source_row: dict[str, Any],
    predictor_row: dict[str, Any],
    missing_required_feature_count: int,
    gis_feature_coverage_pct: float,
    registry,
    confidence_rules,
    strictness: str | None = None,
) -> dict[str, Any]:
    cfg = strictness_profile(confidence_rules, strictness)
    reason_codes: list[str] = []

    merge_values = {str(v).lower() for v in (confidence_rules.merge_left_only_values or ["left_only"])}
    merge_val = source_row.get("_merge")
    if merge_val is not None and str(merge_val).strip().lower() in merge_values:
        reason_codes.append("MERGE_LEFT_ONLY")

    low_cov_threshold = float(cfg.get("low_coverage_threshold", 0.45))
    if gis_feature_coverage_pct < low_cov_threshold:
        reason_codes.append("LOW_GIS_COVERAGE")

    if missing_required_feature_count > 0:
        reason_codes.append("MISSING_REQUIRED_FEATURES")

    if _row_has_status_gaps(source_row, registry=registry, confidence_rules=confidence_rules):
        reason_codes.append("STATUS_DATA_GAPS")

    if _outside_label_domain(source_row, cfg):
        reason_codes.append("OUTSIDE_LABEL_DOMAIN")

    if _has_outlier_feature_values(predictor_row, registry=registry):
        reason_codes.append("OUTLIER_FEATURE_VALUES")

    reason_codes = _dedupe_preserve_order(reason_codes)
    major_set = {str(x) for x in (confidence_rules.major_reason_codes or [])}
    major_reason_code_count = sum(1 for code in reason_codes if code in major_set)

    low_if_any = set(str(x) for x in (cfg.get("low_if_any_reason_codes") or []))
    low_if_missing_required = bool(cfg.get("low_if_missing_required", True))
    low_if_low_cov = bool(cfg.get("low_if_low_coverage", True))
    low_if_major_ge = int(cfg.get("low_if_major_reason_count_gte", 2))

    confidence_level = "High"
    if low_if_any.intersection(reason_codes):
        confidence_level = "Low"
    elif low_if_low_cov and "LOW_GIS_COVERAGE" in reason_codes:
        confidence_level = "Low"
    elif low_if_missing_required and "MISSING_REQUIRED_FEATURES" in reason_codes:
        confidence_level = "Low"
    elif major_reason_code_count >= low_if_major_ge:
        confidence_level = "Low"
    elif major_reason_code_count == 1:
        confidence_level = "Medium"

    return {
        "confidence_level": confidence_level,
        "reason_codes": reason_codes,
        "major_reason_code_count": major_reason_code_count,
    }


def reason_codes_to_pipe(reason_codes: list[str]) -> str:
    return "|".join(reason_codes)


def reason_code_frequencies(reason_code_values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in reason_code_values:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            parts = [str(x).strip() for x in value if str(x).strip()]
        else:
            parts = [p.strip() for p in str(value).split("|") if p.strip()]
        for part in parts:
            counts[part] = counts.get(part, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _row_has_status_gaps(source_row: dict[str, Any], *, registry, confidence_rules) -> bool:
    ok_values = {str(v).lower() for v in (confidence_rules.status_ok_values or ["ok"])}
    status_candidates: list[str] = []
    status_candidates.extend(registry.status_columns or [])
    status_candidates.extend(registry.coverage_flags or [])
    for col in status_candidates:
        if col not in source_row:
            continue
        value = source_row.get(col)
        if value is None:
            return True
        s = str(value).strip()
        if not s:
            return True
        if col.endswith("_coverage_flag"):
            if s.lower() not in {"1", "true", "yes", "covered", "ok"}:
                return True
            continue
        if s.lower() not in ok_values:
            return True
    return False


def _outside_label_domain(source_row: dict[str, Any], cfg: dict[str, Any]) -> bool:
    if not bool(cfg.get("outside_label_domain_static_hsi_enabled", True)):
        return False
    if "Static_HSI_Class" not in source_row:
        return False
    value = source_row.get("Static_HSI_Class")
    if value is None:
        return True
    s = str(value).strip()
    if not s:
        return True
    return s.lower() in {"poor", "unknown", "na", "n/a"}


def _has_outlier_feature_values(predictor_row: dict[str, Any], *, registry) -> bool:
    for feature, bounds in (registry.outlier_bounds or {}).items():
        if feature not in predictor_row:
            continue
        value = predictor_row.get(feature)
        if value is None:
            continue
        try:
            v = float(value)
        except Exception:
            continue
        if v != v:  # nan
            continue
        min_v = (bounds or {}).get("min")
        max_v = (bounds or {}).get("max")
        if min_v is not None and v < float(min_v):
            return True
        if max_v is not None and v > float(max_v):
            return True
    return False


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out

