"""HSI scoring engine — pure tabular maths, no GIS dependencies.

Two stages, deliberately separated so the UI can re-weight without recomputing GIS:

* :func:`resolve_structural` — weight-independent. Resolves the seven WSP suitability
  indices (field value > remote proxy > precautionary default), normalises each to 0-1,
  derives the WSP category (unweighted arithmetic mean of the raw scores) and the
  per-row confidence. Run once after GIS feature extraction.
* :func:`apply_scoring` — cheap and weight-dependent. Computes the weighted structural
  score A, the context score B (from ``ctx_*`` columns produced by :mod:`hsi.context`),
  blends them into the final priority, and ranks. Re-run on every slider change.

Ported and corrected from ``hedge_features/v2/bhsa.py`` (the inherited geometric mean is
replaced by a weighted arithmetic mean, matching the WSP worked example 2.71 -> Excellent).
"""

from __future__ import annotations

from typing import Any

from . import config


# --------------------------------------------------------------------------------------
# Small value helpers (mirrors the robust coercion used in the original engine)
# --------------------------------------------------------------------------------------

def _is_missing(value: Any) -> bool:
    try:
        import pandas as pd

        return bool(pd.isna(value))
    except Exception:
        return value is None


def _first_number(row, names: tuple[str, ...]) -> float | None:
    for name in names:
        if name not in row.index or _is_missing(row.get(name)):
            continue
        try:
            value = float(row.get(name))
        except Exception:
            continue
        if value == value:  # not NaN
            return value
    return None


def _first_bool(row, names: tuple[str, ...]) -> bool | None:
    for name in names:
        if name in row.index and not _is_missing(row.get(name)):
            raw = str(row.get(name)).strip().lower()
            if raw in {"1", "true", "yes", "y", "present", "1.0"}:
                return True
            if raw in {"0", "false", "no", "n", "absent", "0.0"}:
                return False
    return None


def _text(value: Any) -> str:
    return "" if _is_missing(value) else str(value).strip()


def _clip01(value: float) -> float:
    return 0.0 if value < 0 else 1.0 if value > 1 else float(value)


def _component(score: int | None, source: str, confidence: str, reason: str = "") -> dict[str, Any]:
    return {"score": score, "source": source, "confidence": confidence, "reason": reason}


def _missing(reason: str) -> dict[str, Any]:
    return _component(None, "missing", "Missing", reason)


# --------------------------------------------------------------------------------------
# WSP SI band functions (1..N). Copied verbatim from the validated original.
# --------------------------------------------------------------------------------------

def _si_height(value: float) -> int:
    return 1 if value < 1.0 else 2 if value < 2.0 else 3


def _si_width(value: float) -> int:
    return 1 if value < 1.0 else 2 if value < 1.5 else 3


def _si_gap(value: float) -> int:
    pct = value * 100.0 if 0.0 <= value <= 1.0 else value
    return 1 if pct > 20.0 else 2 if pct >= 10.0 else 3


def _si_arable_margin(value: float) -> int:
    return 1 if value <= 0.0 else 2 if value < 2.0 else 3 if value <= 5.0 else 4


def _si_trees(value: float) -> int:
    return 1 if value <= 0 else 2 if value <= 2 else 3 if value <= 6 else 4


def _si_species(value: float) -> int:
    return 1 if value <= 3 else 2 if value <= 6 else 3


# --------------------------------------------------------------------------------------
# Per-SI resolution: field value > remote proxy > (precautionary default / missing)
# --------------------------------------------------------------------------------------

def _direct_si(row, key: str) -> dict[str, Any] | None:
    """Pre-scored SI band supplied directly as a column (e.g. ``si1`` / ``SI1``)."""
    value = _first_number(row, (key, key.upper(), f"bhsa_{key}"))
    if value is None:
        return None
    band = int(round(value))
    return _component(band, "field", "High")


def _resolve_si1(row) -> dict[str, Any]:
    direct = _direct_si(row, "si1")
    if direct:
        return direct
    value = _first_number(row, ("si1_height_m", "hedge_height_m", "height_m"))
    if value is not None:
        return _component(_si_height(value), "field", "High")
    value = _first_number(row, ("hedge_struct_height_p90_5m", "hedge_struct_height_mean_5m"))
    if value is not None:
        return _component(_si_height(value), "proxy", "Medium")
    return _missing("No height field or LiDAR canopy-height proxy.")


def _resolve_si2(row) -> dict[str, Any]:
    direct = _direct_si(row, "si2")
    if direct:
        return direct
    value = _first_number(row, ("si2_width_m", "hedge_width_m", "width_m"))
    if value is not None:
        return _component(_si_width(value), "field", "High")
    value = _first_number(row, ("hedge_struct_width_proxy_m",))
    if value is not None:
        return _component(_si_width(value), "proxy", "Medium")
    return _missing("No width field or LiDAR width proxy.")


def _resolve_si3(row) -> dict[str, Any]:
    direct = _direct_si(row, "si3")
    if direct:
        return direct
    value = _first_number(row, ("si3_gappiness_pct", "gappiness_pct", "gap_pct"))
    if value is not None:
        return _component(_si_gap(value), "field", "High")
    value = _first_number(row, ("hedge_struct_gap_fraction_10m",))
    if value is None:
        continuity = _first_number(row, ("hedge_struct_canopy_continuity_10m",))
        value = None if continuity is None else 1.0 - _clip01(continuity)
    if value is not None:
        return _component(_si_gap(value), "proxy", "Medium")
    return _missing("No gappiness field or canopy-gap proxy.")


def _resolve_si4(row) -> dict[str, Any]:
    direct = _direct_si(row, "si4")
    if direct:
        return direct
    value = _first_number(row, ("si4_arable_margin_m", "arable_margin_m"))
    if value is not None:
        return _component(_si_arable_margin(value), "field", "High")
    # CROME-derived arable proportion (preferred) then WorldCover cropland proportion.
    crop = _first_number(row, ("crome_arable_pct",))
    crop_source = "CROME arable adjacency"
    if crop is None:
        crop = _first_number(row, ("buf100_worldcover_cropland_pct", "buf250_worldcover_cropland_pct"))
        crop_source = "WorldCover cropland proportion"
    if crop is not None:
        band = 1 if crop <= 0.05 else 2 if crop < 0.25 else 3 if crop < 0.60 else 4
        return _component(band, "proxy", "Low", f"SI4 inferred from {crop_source}, not measured margin width.")
    return _missing("No arable-margin field or cropland proxy.")


def _resolve_si5(row) -> dict[str, Any]:
    direct = _direct_si(row, "si5")
    if direct:
        return direct
    value = _first_number(row, ("si5_tree_count_50m", "trees_per_50m"))
    if value is not None:
        return _component(_si_trees(value), "field", "High")
    standard_pct = _first_number(row, ("hedge_struct_tree_standard_pct_10m",))
    if standard_pct is not None:
        band = 1 if standard_pct <= 0 else 2 if standard_pct < 0.15 else 3 if standard_pct < 0.45 else 4
        return _component(band, "proxy", "Medium", "SI5 inferred from LiDAR canopy-cover fraction, not literal crown counts.")
    tree_pct = _first_number(row, ("buf100_worldcover_tree_pct",))
    if tree_pct is not None:
        band = 1 if tree_pct <= 0.05 else 2 if tree_pct < 0.25 else 3 if tree_pct < 0.55 else 4
        return _component(band, "proxy", "Low", "SI5 inferred from WorldCover tree-cover proportion.")
    return _missing("No tree-count field or tree-cover proxy.")


def _resolve_si6(row, *, default_band: int) -> dict[str, Any]:
    direct = _direct_si(row, "si6")
    if direct:
        return direct
    value = _first_number(row, ("si6_woody_species_count_20m", "woody_species_count_20m"))
    if value is not None:
        return _component(_si_species(value), "field", "High")
    return _component(
        int(default_band),
        "default",
        "Low",
        "SI6 woody species diversity is not remotely verifiable; precautionary default applied. "
        "Supply woody_species_count_20m to override.",
    )


def _resolve_si7(row) -> dict[str, Any]:
    direct = _direct_si(row, "si7")
    if direct:
        return direct
    value = _first_bool(row, ("si7_wet_ditch_present", "wet_ditch_present"))
    if value is not None:
        return _component(2 if value else 1, "field", "High")
    river_density = _first_number(row, ("buf100_os_river_density_m_per_ha", "buf250_os_river_density_m_per_ha"))
    dist_water = _first_number(row, ("dist_os_river_m",))
    wetland = _first_number(row, ("buf100_worldcover_wetland_pct",)) or 0.0
    water = _first_number(row, ("buf100_worldcover_water_pct",)) or 0.0
    if river_density is not None or dist_water is not None or wetland > 0 or water > 0:
        present = (
            (river_density is not None and river_density > 5)
            or wetland >= 0.05
            or water >= 0.05
            or (dist_water is not None and dist_water <= 25)
        )
        return _component(2 if present else 1, "proxy", "Low", "SI7 from watercourse proximity proxy; unmapped ditches not captured.")
    return _missing("No wet-ditch field or watercourse proxy.")


SI_RESOLVERS = {
    "si1": _resolve_si1,
    "si2": _resolve_si2,
    "si3": _resolve_si3,
    "si4": _resolve_si4,
    "si5": _resolve_si5,
    "si7": _resolve_si7,
}


# --------------------------------------------------------------------------------------
# Stage 1: weight-independent structural resolution
# --------------------------------------------------------------------------------------

def _category_from_score(score: float | None) -> str:
    if score is None:
        return "Incomplete"
    if score < config.WSP_CATEGORY_GOOD_LOWER:
        return "Poor"
    if score < config.WSP_CATEGORY_EXCELLENT_LOWER:
        return "Good"
    return "Excellent"


def _normalise_si(key: str, band: int | None) -> float | None:
    if band is None:
        return None
    si_max = config.SI_MAX[key]
    if si_max <= 1:
        return _clip01(float(band) - 1.0)
    return _clip01((float(band) - 1.0) / (float(si_max) - 1.0))


def _resolve_row(row, *, si6_default_band: int) -> dict[str, Any]:
    components: dict[str, dict[str, Any]] = {}
    for key in config.SI_KEYS:
        if key == "si6":
            components[key] = _resolve_si6(row, default_band=si6_default_band)
        else:
            components[key] = SI_RESOLVERS[key](row)

    present_scores = [c["score"] for c in components.values() if c["score"] is not None]
    missing = [config.SI_LABELS[k] for k, c in components.items() if c["score"] is None]
    complete = not missing

    wsp_score = (sum(present_scores) / len(present_scores)) if present_scores else None
    category = _category_from_score(wsp_score)

    if not complete:
        confidence_level = "Incomplete"
    else:
        confidences = [c["confidence"] for c in components.values()]
        confidence_level = (
            "Low" if "Low" in confidences else "Medium" if "Medium" in confidences else "High"
        )
    confidence_score = sum(
        config.CONFIDENCE_WEIGHTS.get(c["confidence"], 0.0) for c in components.values()
    ) / len(components)

    field_verification_required = bool(
        not complete
        or any(c["confidence"] == "Low" for c in components.values())
        or components["si6"]["source"] != "field"
        or components["si7"]["source"] != "field"
    )

    survey_requirement = (
        config.SURVEY_REQUIREMENTS["Incomplete"]
        if not complete
        else config.SURVEY_REQUIREMENTS.get(category, config.SURVEY_REQUIREMENTS["Incomplete"])
    )

    notes = [c["reason"] for c in components.values() if c["reason"]]
    out: dict[str, Any] = {
        "hsi_wsp_score": round(float(wsp_score), 4) if wsp_score is not None else None,
        "hsi_wsp_category": category,
        "hsi_complete": bool(complete),
        "hsi_present_si_count": int(len(present_scores)),
        "hsi_confidence_level": confidence_level,
        "hsi_confidence_score": round(float(confidence_score), 4),
        "field_verification_required": field_verification_required,
        "hsi_survey_requirement": survey_requirement,
        "hsi_missing_reasons": "|".join(missing),
        "hsi_notes": "|".join(notes),
    }
    for key, comp in components.items():
        out[f"hsi_{key}_score"] = comp["score"]
        out[f"hsi_{key}_norm"] = _normalise_si(key, comp["score"])
        out[f"hsi_{key}_source"] = comp["source"]
        out[f"hsi_{key}_confidence"] = comp["confidence"]
    return out


def resolve_structural(df, *, si6_default_band: int = config.SI6_DEFAULT_BAND):
    """Resolve SI1-SI7, normalise, derive WSP category & confidence (weight-independent)."""
    import pandas as pd

    out = df.copy()
    rows = [_resolve_row(row, si6_default_band=si6_default_band) for _, row in out.iterrows()]
    resolved = pd.DataFrame(rows, index=out.index)
    for col in resolved.columns:
        out[col] = resolved[col]
    return out


# --------------------------------------------------------------------------------------
# Stage 2: weight-dependent scoring (cheap; safe to re-run on slider change)
# --------------------------------------------------------------------------------------

def _weighted_mean(row, keys: tuple[str, ...], norm_template: str, weights: dict[str, float]) -> float | None:
    num = 0.0
    den = 0.0
    for key in keys:
        col = norm_template.format(key=key)
        if col not in row.index:
            continue
        value = row.get(col)
        if _is_missing(value):
            continue
        weight = float(weights.get(key, 0.0))
        if weight <= 0:
            continue
        num += weight * float(value)
        den += weight
    if den <= 0:
        return None
    return _clip01(num / den)


def apply_scoring(df, settings: config.ScoreSettings | None = None):
    """Compute structural A, context B, final priority and rank from resolved columns."""
    import pandas as pd

    settings = settings or config.ScoreSettings()
    out = df.copy()

    a_values: list[float | None] = []
    b_values: list[float | None] = []
    for _, row in out.iterrows():
        a_values.append(_weighted_mean(row, config.SI_KEYS, "hsi_{key}_norm", settings.si_weights))
        b_values.append(_weighted_mean(row, config.CONTEXT_KEYS, "{key}", settings.context_weights))
    out["hsi_structural_A"] = [round(v, 4) if v is not None else None for v in a_values]
    out["hsi_context_B"] = [round(v, 4) if v is not None else None for v in b_values]

    alpha = float(settings.alpha)
    priority: list[float | None] = []
    for a, b in zip(a_values, b_values):
        if a is None and b is None:
            priority.append(None)
        elif b is None:
            priority.append(a)
        elif a is None:
            priority.append(b)
        else:
            priority.append(_clip01(alpha * a + (1.0 - alpha) * b))
    out["hsi_priority"] = [round(v, 4) if v is not None else None for v in priority]

    ranks = (
        pd.Series(priority, index=out.index)
        .rank(method="dense", ascending=False, na_option="bottom")
    )
    out["hsi_priority_rank"] = ranks.astype("Int64")
    return out


def score_hedgerows(df, *, settings: config.ScoreSettings | None = None):
    """Convenience: resolve structural indices then apply weighting in one call."""
    settings = settings or config.ScoreSettings()
    resolved = resolve_structural(df, si6_default_band=settings.si6_default_band)
    return apply_scoring(resolved, settings=settings)
