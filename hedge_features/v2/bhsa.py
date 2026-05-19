from __future__ import annotations

from dataclasses import asdict, dataclass
from math import prod
from typing import Any


BHSA_METHOD_VERSION = "bhsa_remote_v2_foundation"

SI_LABELS = {
    "si1": "Height",
    "si2": "Width",
    "si3": "Gappiness",
    "si4": "Arable field margin",
    "si5": "Trees present",
    "si6": "Woody species diversity",
    "si7": "Wet ditch",
}

SURVEY_REQUIREMENTS = {
    "Poor": "No further survey",
    "Good": "Seasonal automated static detector survey",
    "Excellent": "Seasonal automated static plus monthly modified DEFRA local-level surveys",
    "Incomplete": "Field verification required before survey effort can be reduced",
}


@dataclass(frozen=True, slots=True)
class BHSAScoringSettings:
    mode: str = "hybrid"
    hedgerow_id_column: str = "hedgerow_id"
    major_road_downgrade_distance_m: float = 50.0
    ecologist_adjustment_column: str = "bhsa_ecologist_adjustment_class"
    ecologist_adjustment_reason_column: str = "bhsa_ecologist_adjustment_reason"


def score_bhsa_table(df, *, settings: BHSAScoringSettings | None = None):
    """Score field, proxy, or hybrid BHSA inputs and return an annotated copy plus run summary."""
    import pandas as pd

    settings = settings or BHSAScoringSettings()
    mode = str(settings.mode or "hybrid").lower()
    if mode not in {"field", "proxy", "hybrid"}:
        raise ValueError("BHSA scoring mode must be one of: field, proxy, hybrid.")

    out = df.copy()
    rows: list[dict[str, Any]] = []
    for _, row in out.iterrows():
        rows.append(_score_row(row, settings=settings, mode=mode))
    result = pd.DataFrame(rows, index=out.index)
    for col in result.columns:
        out[col] = result[col]

    summary = {
        "method_version": BHSA_METHOD_VERSION,
        "settings": asdict(settings),
        "mode": mode,
        "row_count": int(len(out)),
        "scored_row_count": int(out["bhsa_score"].notna().sum()),
        "class_counts": _counts(out["bhsa_class"].tolist()),
        "confidence_counts": _counts(out["bhsa_confidence_level"].tolist()),
        "field_verification_required_count": int(out["field_verification_required"].astype(bool).sum()),
    }
    return out, summary


def _score_row(row, *, settings: BHSAScoringSettings, mode: str) -> dict[str, Any]:
    components = {
        "si1": _score_height(row, mode=mode),
        "si2": _score_width(row, mode=mode),
        "si3": _score_gap(row, mode=mode),
        "si4": _score_arable_margin(row, mode=mode),
        "si5": _score_trees(row, mode=mode),
        "si6": _score_species_diversity(row, mode=mode),
        "si7": _score_wet_ditch(row, mode=mode),
    }
    values = [item["score"] for item in components.values()]
    missing = [SI_LABELS[key] for key, item in components.items() if item["score"] is None]
    score = _geometric_mean(values) if not missing else None
    raw_class = _class_from_score(score)
    road_distance = _first_number(row, ("major_road_distance_m", "dist_major_road_m", "dist_a_road_m", "dist_primary_road_m"))
    road_downgraded = (
        road_distance is not None
        and raw_class in {"Good", "Excellent"}
        and road_distance <= float(settings.major_road_downgrade_distance_m)
    )
    road_class = _downgrade_class(raw_class) if road_downgraded else raw_class
    final_class, adjustment_applied = _apply_ecologist_adjustment(row, road_class, settings=settings)
    notes = [item["reason"] for item in components.values() if item["reason"]]
    if road_downgraded:
        notes.append(f"Downgraded because major road distance is {road_distance:g} m.")
    if adjustment_applied:
        reason = _text(row.get(settings.ecologist_adjustment_reason_column))
        notes.append(f"Professional judgement adjustment applied{': ' + reason if reason else ''}.")

    field_verification_required = bool(
        missing
        or any(item["confidence"] == "Low" for item in components.values())
        or components["si6"]["source"] != "field"
        or components["si7"]["source"] != "field"
    )
    out: dict[str, Any] = {
        "bhsa_method_version": BHSA_METHOD_VERSION,
        "bhsa_mode": mode,
        "bhsa_score": round(float(score), 6) if score is not None else None,
        "bhsa_class_raw": raw_class,
        "bhsa_class": final_class,
        "bhsa_survey_requirement": SURVEY_REQUIREMENTS.get(final_class, SURVEY_REQUIREMENTS["Incomplete"]),
        "bhsa_confidence_level": _overall_confidence(components, missing),
        "field_verification_required": field_verification_required,
        "bhsa_missing_reasons": "|".join(missing),
        "bhsa_major_road_downgraded": int(road_downgraded),
        "bhsa_adjustment_applied": int(adjustment_applied),
        "bhsa_notes": "|".join(notes),
    }
    for key, item in components.items():
        out[f"bhsa_{key}_score"] = item["score"]
        out[f"bhsa_{key}_source"] = item["source"]
        out[f"bhsa_{key}_confidence"] = item["confidence"]
    return out


def _score_height(row, *, mode: str) -> dict[str, Any]:
    direct = _direct_si(row, "si1", mode=mode)
    if direct:
        return direct
    value = _field_number(row, mode=mode, names=("si1_height_m", "bhsa_height_m", "hedge_height_m", "height_m"))
    if value is not None:
        return _component(_si_height(value), "field", "High")
    value = _proxy_number(row, mode=mode, names=("hedge_struct_height_mean_5m", "hedge_struct_height_p90_5m"))
    if value is not None:
        return _component(_si_height(value), "proxy", "Medium")
    return _missing("No height field or LiDAR structure proxy.")


def _score_width(row, *, mode: str) -> dict[str, Any]:
    direct = _direct_si(row, "si2", mode=mode)
    if direct:
        return direct
    value = _field_number(row, mode=mode, names=("si2_width_m", "bhsa_width_m", "hedge_width_m", "width_m"))
    if value is not None:
        return _component(_si_width(value), "field", "High")
    value = _proxy_number(row, mode=mode, names=("hedge_struct_width_proxy_m",))
    if value is not None:
        return _component(_si_width(value), "proxy", "Medium")
    return _missing("No width field or LiDAR width proxy.")


def _score_gap(row, *, mode: str) -> dict[str, Any]:
    direct = _direct_si(row, "si3", mode=mode)
    if direct:
        return direct
    value = _field_number(row, mode=mode, names=("si3_gappiness_pct", "bhsa_gappiness_pct", "gappiness_pct", "gap_pct"))
    if value is not None:
        return _component(_si_gap(value), "field", "High")
    value = _proxy_number(row, mode=mode, names=("hedge_struct_gap_fraction_10m",))
    if value is None:
        continuity = _proxy_number(row, mode=mode, names=("hedge_struct_canopy_continuity_10m",))
        value = None if continuity is None else 1.0 - _clip01(continuity)
    if value is not None:
        return _component(_si_gap(value), "proxy", "Medium")
    return _missing("No gappiness field or canopy-gap proxy.")


def _score_arable_margin(row, *, mode: str) -> dict[str, Any]:
    direct = _direct_si(row, "si4", mode=mode)
    if direct:
        return direct
    value = _field_number(row, mode=mode, names=("si4_arable_margin_m", "bhsa_arable_margin_m", "arable_margin_m"))
    if value is not None:
        return _component(_si_arable_margin(value), "field", "High")
    crop = _proxy_number(row, mode=mode, names=("buf100_worldcover_cropland_pct", "buf250_worldcover_cropland_pct"))
    if crop is not None:
        score = 1 if crop <= 0.05 else 2 if crop < 0.25 else 3 if crop < 0.60 else 4
        return _component(score, "proxy", "Low", "SI4 inferred from cropland proportion, not measured margin width.")
    return _missing("No arable margin field or cropland proxy.")


def _score_trees(row, *, mode: str) -> dict[str, Any]:
    direct = _direct_si(row, "si5", mode=mode)
    if direct:
        return direct
    value = _field_number(row, mode=mode, names=("si5_tree_count_50m", "bhsa_trees_per_50m", "trees_per_50m"))
    if value is not None:
        return _component(_si_trees(value), "field", "High")
    standard_pct = _proxy_number(row, mode=mode, names=("hedge_struct_tree_standard_pct_10m",))
    if standard_pct is not None:
        score = 1 if standard_pct <= 0 else 2 if standard_pct < 0.15 else 3 if standard_pct < 0.45 else 4
        return _component(score, "proxy", "Medium")
    tree_pct = _proxy_number(row, mode=mode, names=("buf100_worldcover_tree_pct", "mhb_corridor10_tree_pct"))
    if tree_pct is not None:
        score = 1 if tree_pct <= 0.05 else 2 if tree_pct < 0.25 else 3 if tree_pct < 0.55 else 4
        return _component(score, "proxy", "Low", "SI5 inferred from tree-cover proportion, not counted standards.")
    return _missing("No tree-count field or tree-cover proxy.")


def _score_species_diversity(row, *, mode: str) -> dict[str, Any]:
    direct = _direct_si(row, "si6", mode=mode)
    if direct:
        return direct
    value = _field_number(row, mode=mode, names=("si6_woody_species_count_20m", "bhsa_woody_species_count_20m", "woody_species_count_20m"))
    if value is not None:
        return _component(_si_species(value), "field", "High")
    if mode in {"proxy", "hybrid"}:
        tree = _first_number(row, ("buf100_worldcover_tree_pct", "mhb_corridor10_tree_pct")) or 0.0
        broad = _first_number(row, ("buf100_phi_broadleaved_woodland_pct",)) or 0.0
        awi = _first_number(row, ("dist_awi_ancwood_m", "roostpx_dist_ancwood_m"))
        patch = _first_number(row, ("buf100_worldcover_patch_richness",)) or 0.0
        if any(name in row.index for name in ("buf100_worldcover_tree_pct", "buf100_phi_broadleaved_woodland_pct", "dist_awi_ancwood_m", "buf100_worldcover_patch_richness")):
            signal = _clip01((0.45 * tree) + (0.35 * broad) + (0.15 if awi is not None and awi <= 100 else 0.0) + (0.05 * min(patch / 5.0, 1.0)))
            score = 1 if signal < 0.30 else 2 if signal < 0.60 else 3
            return _component(score, "proxy", "Low", "SI6 woody species diversity is not remotely verifiable; proxy requires field check.")
    return _missing("No woody species diversity field; SI6 cannot be remotely verified.")


def _score_wet_ditch(row, *, mode: str) -> dict[str, Any]:
    direct = _direct_si(row, "si7", mode=mode)
    if direct:
        return direct
    value = _field_bool(row, mode=mode, names=("si7_wet_ditch_present", "bhsa_wet_ditch_present", "wet_ditch_present"))
    if value is not None:
        return _component(2 if value else 1, "field", "High")
    if mode in {"proxy", "hybrid"}:
        river_density = _first_number(row, ("buf100_os_river_density_m_per_ha", "buf250_os_river_density_m_per_ha")) or 0.0
        dist_water = _first_number(row, ("mhb_water_dist_m", "dist_os_river_m"))
        wetland = _first_number(row, ("buf100_worldcover_wetland_pct", "buf100_phi_wetland_pct", "mhb_corridor10_wetland_pct")) or 0.0
        water = _first_number(row, ("buf100_worldcover_water_pct", "mhb_corridor10_water_pct")) or 0.0
        if river_density > 0 or dist_water is not None or wetland > 0 or water > 0:
            present = river_density > 5 or wetland >= 0.05 or water >= 0.05 or (dist_water is not None and dist_water <= 25)
            return _component(2 if present else 1, "proxy", "Low", "SI7 wet ditch is not remotely verifiable; watercourse proxy requires field check.")
    return _missing("No wet ditch field; SI7 cannot be remotely verified.")


def _direct_si(row, key: str, *, mode: str) -> dict[str, Any] | None:
    if mode == "proxy":
        return None
    value = _first_number(row, (key, f"bhsa_{key}", f"BHSA_{key.upper()}"))
    if value is None:
        return None
    return _component(int(round(value)), "field", "High")


def _component(score: int | None, source: str, confidence: str, reason: str = "") -> dict[str, Any]:
    return {"score": score, "source": source, "confidence": confidence, "reason": reason}


def _missing(reason: str) -> dict[str, Any]:
    return _component(None, "missing", "Missing", reason)


def _field_number(row, *, mode: str, names: tuple[str, ...]) -> float | None:
    return None if mode == "proxy" else _first_number(row, names)


def _proxy_number(row, *, mode: str, names: tuple[str, ...]) -> float | None:
    return None if mode == "field" else _first_number(row, names)


def _field_bool(row, *, mode: str, names: tuple[str, ...]) -> bool | None:
    if mode == "proxy":
        return None
    for name in names:
        if name in row.index and not _is_missing(row.get(name)):
            raw = str(row.get(name)).strip().lower()
            if raw in {"1", "true", "yes", "y", "present"}:
                return True
            if raw in {"0", "false", "no", "n", "absent"}:
                return False
    return None


def _first_number(row, names: tuple[str, ...]) -> float | None:
    for name in names:
        if name not in row.index or _is_missing(row.get(name)):
            continue
        try:
            value = float(row.get(name))
        except Exception:
            continue
        if value == value:
            return value
    return None


def _text(value) -> str:
    return "" if _is_missing(value) else str(value).strip()


def _is_missing(value) -> bool:
    try:
        import pandas as pd

        return bool(pd.isna(value))
    except Exception:
        return value is None


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


def _geometric_mean(values: list[int | None]) -> float:
    numeric = [float(v) for v in values if v is not None]
    return prod(numeric) ** (1.0 / len(numeric))


def _class_from_score(score: float | None) -> str:
    if score is None:
        return "Incomplete"
    if score < 1.70:
        return "Poor"
    if score <= 2.39:
        return "Good"
    return "Excellent"


def _downgrade_class(class_name: str) -> str:
    if class_name == "Excellent":
        return "Good"
    if class_name == "Good":
        return "Poor"
    return class_name


def _apply_ecologist_adjustment(row, class_name: str, *, settings: BHSAScoringSettings) -> tuple[str, bool]:
    if settings.ecologist_adjustment_column not in row.index:
        return class_name, False
    requested = _text(row.get(settings.ecologist_adjustment_column)).title()
    if requested in SURVEY_REQUIREMENTS and requested != class_name:
        return requested, True
    return class_name, False


def _overall_confidence(components: dict[str, dict[str, Any]], missing: list[str]) -> str:
    if missing:
        return "Incomplete"
    confidences = [item["confidence"] for item in components.values()]
    if any(c == "Low" for c in confidences):
        return "Low"
    if any(c == "Medium" for c in confidences):
        return "Medium"
    return "High"


def _clip01(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return float(value)


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
