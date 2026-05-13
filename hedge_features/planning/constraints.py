from __future__ import annotations


def apply_planning_constraints(
    candidates_gdf,
    *,
    settings,
    include_area_gdf=None,
    exclude_area_gdf=None,
):
    gdf = candidates_gdf.copy()
    include_union = _union_in_candidate_crs(include_area_gdf, target_crs=gdf.crs)
    exclude_union = _union_in_candidate_crs(exclude_area_gdf, target_crs=gdf.crs)

    gdf["within_include_area"] = 1
    gdf["outside_exclude_area"] = 1
    gdf["access_allowed"] = 1
    gdf["above_min_score"] = 1
    gdf["lighting_allowed"] = 1
    gdf["confidence_allowed"] = 1
    gdf["eligible_for_selection"] = 1
    gdf["constraint_reason_codes"] = ""

    if include_union is not None:
        include_mask = gdf.geometry.intersects(include_union)
        gdf["within_include_area"] = include_mask.astype(int)
    if exclude_union is not None:
        exclude_mask = gdf.geometry.intersects(exclude_union)
        gdf["outside_exclude_area"] = (~exclude_mask).astype(int)
    if settings.access_flag_column:
        gdf["access_allowed"] = gdf[settings.access_flag_column].apply(_is_allowed_access).astype(int)
    if settings.min_score is not None:
        gdf["above_min_score"] = (gdf["candidate_score"].astype(float) >= float(settings.min_score)).astype(int)
    if bool(getattr(settings, "reject_overlit_candidates", False)):
        gdf["lighting_allowed"] = (~gdf.apply(_is_overlit_candidate, axis=1)).astype(int)
    if bool(getattr(settings, "reject_low_confidence_candidates", False)):
        gdf["confidence_allowed"] = gdf.apply(_passes_confidence_gate, axis=1).astype(int)

    reasons: list[str] = []
    combined: list[str] = []
    for _, row in gdf.iterrows():
        reasons = []
        if not int(row["within_include_area"]):
            reasons.append("OUTSIDE_INCLUDE_AREA")
        if not int(row["outside_exclude_area"]):
            reasons.append("INSIDE_EXCLUDE_AREA")
        if not int(row["access_allowed"]):
            reasons.append("ACCESS_BLOCKED")
        if not int(row["above_min_score"]):
            reasons.append("BELOW_MIN_SCORE")
        if not int(row["lighting_allowed"]):
            reasons.append("OVERLIT_CORRIDOR")
        if not int(row["confidence_allowed"]):
            reasons.append("LOW_EVIDENCE_CONFIDENCE")
        combined.append("|".join(reasons))
    gdf["constraint_reason_codes"] = combined
    gdf["eligible_for_selection"] = (
        (gdf["within_include_area"].astype(int) == 1)
        & (gdf["outside_exclude_area"].astype(int) == 1)
        & (gdf["access_allowed"].astype(int) == 1)
        & (gdf["above_min_score"].astype(int) == 1)
        & (gdf["lighting_allowed"].astype(int) == 1)
        & (gdf["confidence_allowed"].astype(int) == 1)
    ).astype(int)
    gdf.loc[gdf["eligible_for_selection"] == 0, "planning_status"] = "ineligible"
    return gdf


def _union_in_candidate_crs(area_gdf, *, target_crs):
    if area_gdf is None or len(area_gdf) == 0:
        return None
    gdf = area_gdf
    if gdf.crs is not None and str(gdf.crs) != str(target_crs):
        gdf = gdf.to_crs(target_crs)
    return gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union


def _is_allowed_access(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    return s in {"1", "true", "yes", "y", "ok", "allowed", "access"}


def _is_overlit_candidate(row) -> bool:
    for column_name in ("planning_target_reason_codes", "evidence_reason_codes"):
        text = str(row.get(column_name, "")).strip().upper()
        if "LIGHTING_RISK_HIGH" in text or "LIGHT_DISTURBANCE_HIGH" in text:
            return True
    return False


def _passes_confidence_gate(row) -> bool:
    confidence = str(row.get("evidence_confidence_level", "")).strip().lower()
    domain = str(row.get("planning_target_domain_status", row.get("evidence_domain_status", ""))).strip().lower()
    if confidence == "low":
        return False
    if domain == "outside":
        return False
    return True
