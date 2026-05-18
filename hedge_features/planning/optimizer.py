from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations


@dataclass(slots=True)
class _SelectionState:
    selected_route_units: set[str] = field(default_factory=set)
    selected_corridor_units: set[str] = field(default_factory=set)
    selected_high_risk_corridors: set[str] = field(default_factory=set)
    guild_counts: dict[str, int] = field(default_factory=dict)
    selected_geoms: list[object] = field(default_factory=list)


def select_detector_locations(candidates_gdf, *, settings):
    strategy = str(getattr(settings, "optimizer_strategy", "greedy") or "greedy").lower()
    if strategy not in {"greedy", "greedy_coverage", "exact"}:
        raise ValueError(f"Unsupported optimizer strategy '{strategy}'. Supported: greedy, exact.")

    gdf = candidates_gdf.copy()
    _ensure_optimizer_columns(gdf)
    _prepare_optimizer_features(gdf, settings=settings)

    budget = max(int(settings.detector_budget), 0)
    if budget == 0:
        return gdf.iloc[0:0].copy(), gdf
    if strategy == "exact":
        return _select_detector_locations_exact(gdf, settings=settings)

    state = _SelectionState()
    rank = 1

    if settings.section_column and settings.section_minimum_counts:
        for section_name in sorted(settings.section_minimum_counts):
            needed = max(int(settings.section_minimum_counts.get(section_name, 0)), 0)
            while needed > 0 and rank <= budget:
                idx, metrics = _choose_next_candidate(
                    gdf,
                    settings=settings,
                    state=state,
                    section_filter=str(section_name),
                )
                if idx is None:
                    break
                _apply_selection(
                    gdf,
                    idx=idx,
                    rank=rank,
                    phase="section_minimum",
                    state=state,
                    metrics=metrics,
                    settings=settings,
                )
                rank += 1
                needed -= 1

    while rank <= budget:
        idx, metrics = _choose_next_candidate(
            gdf,
            settings=settings,
            state=state,
            section_filter=None,
        )
        if idx is None:
            break
        _apply_selection(
            gdf,
            idx=idx,
            rank=rank,
            phase="greedy_coverage",
            state=state,
            metrics=metrics,
            settings=settings,
        )
        rank += 1

    selected = gdf[gdf["selected_flag"].astype(int) == 1].copy()
    if not selected.empty:
        selected = selected.sort_values("selection_rank").reset_index(drop=True)

    gdf.loc[
        (gdf["eligible_for_selection"].astype(int) == 1) & (gdf["selected_flag"].astype(int) == 0),
        "planning_status",
    ] = "eligible_unselected"

    return selected, gdf


def _select_detector_locations_exact(gdf, *, settings):
    eligible = gdf[gdf["eligible_for_selection"].astype(int) == 1].copy()
    if eligible.empty:
        gdf.loc[gdf["selected_flag"].astype(int) == 0, "planning_status"] = "ineligible"
        return gdf.iloc[0:0].copy(), gdf

    max_candidates = max(int(getattr(settings, "exact_optimizer_max_candidates", 18)), 1)
    if len(eligible) > max_candidates:
        raise ValueError(
            f"Exact optimizer supports at most {max_candidates} eligible candidates in this build; "
            f"found {len(eligible)}. Use --optimizer greedy or raise exact_optimizer_max_candidates."
        )

    candidates = list(eligible.sort_values("candidate_id").index)
    budget = min(max(int(settings.detector_budget), 0), len(candidates))
    best_combo: tuple | None = None
    best_key: tuple[float, int, tuple[str, ...]] | None = None
    for size in range(1, budget + 1):
        for combo in combinations(candidates, size):
            if not _exact_combo_feasible(gdf, combo, settings=settings):
                continue
            value = _exact_combo_objective(gdf, combo, settings=settings, budget=budget)
            candidate_ids = tuple(str(gdf.at[idx, "candidate_id"]) for idx in combo)
            key = (float(value), int(size), tuple(reversed(candidate_ids)))
            if best_key is None or key > best_key:
                best_key = key
                best_combo = combo

    if not best_combo:
        selected = gdf.iloc[0:0].copy()
        gdf.loc[gdf["eligible_for_selection"].astype(int) == 1, "planning_status"] = "eligible_unselected"
        return selected, gdf

    state = _SelectionState()
    ordered = sorted(
        best_combo,
        key=lambda idx: (
            -_exact_candidate_base_value(gdf.loc[idx], settings=settings),
            str(gdf.at[idx, "candidate_id"]),
        ),
    )
    for rank, idx in enumerate(ordered, start=1):
        metrics = _marginal_metrics(gdf.loc[idx], settings=settings, state=state)
        _apply_selection(
            gdf,
            idx=idx,
            rank=rank,
            phase="exact_integer",
            state=state,
            metrics=metrics,
            settings=settings,
        )

    selected = gdf[gdf["selected_flag"].astype(int) == 1].copy()
    if not selected.empty:
        selected = selected.sort_values("selection_rank").reset_index(drop=True)
    gdf.loc[
        (gdf["eligible_for_selection"].astype(int) == 1) & (gdf["selected_flag"].astype(int) == 0),
        "planning_status",
    ] = "eligible_unselected"
    return selected, gdf


def _exact_combo_feasible(gdf, combo, *, settings) -> bool:
    if settings.section_column and settings.section_minimum_counts:
        route_counts: dict[str, int] = {}
        for idx in combo:
            route = str(gdf.at[idx, "optimization_route_unit"])
            route_counts[route] = route_counts.get(route, 0) + 1
        for section_name, needed in settings.section_minimum_counts.items():
            if route_counts.get(str(section_name), 0) < int(needed):
                return False
    min_spacing = float(settings.min_detector_spacing_m)
    for left_pos, left_idx in enumerate(combo):
        left_geom = gdf.at[left_idx, gdf.geometry.name]
        for right_idx in combo[left_pos + 1:]:
            if float(left_geom.distance(gdf.at[right_idx, gdf.geometry.name])) < min_spacing:
                return False
    return True


def _exact_combo_objective(gdf, combo, *, settings, budget: int) -> float:
    selected = gdf.loc[list(combo)]
    denom = max(float(budget), 1.0)
    route_weight = float(settings.objective_weight_route_coverage)
    corridor_weight = float(getattr(settings, "objective_weight_corridor_coverage", 0.0) or 0.0)
    base = sum(_exact_candidate_base_value(row, settings=settings) for _, row in selected.iterrows()) / denom
    route = selected["optimization_route_unit"].astype(str).nunique() / denom
    corridor = selected["optimization_corridor_unit"].astype(str).nunique() / denom
    guilds = [g for g in selected["optimization_primary_guild"].astype(str).tolist() if g != "unknown"]
    habitat = (len(set(guilds)) / max(float(len(set(guilds)) or 1), 1.0)) if guilds else 0.0
    high_risk = float(selected["optimization_high_risk_flag"].astype(int).sum()) / denom
    uncertainty = float(selected["_optimizer_uncertainty"].astype(float).sum()) / denom
    redundancy = _exact_redundancy_penalty(selected)
    return (
        (float(settings.objective_weight_base_score) * base)
        + (route_weight * route)
        + (corridor_weight * corridor)
        + (float(settings.objective_weight_habitat_representation) * habitat)
        + (float(settings.objective_weight_high_risk_coverage) * high_risk)
        + (float(settings.objective_weight_uncertainty_reduction) * uncertainty)
        - (float(settings.objective_weight_redundancy_penalty) * redundancy)
    )


def _exact_candidate_base_value(row, *, settings) -> float:
    return float(row["_optimizer_base_score_norm"])


def _exact_redundancy_penalty(selected) -> float:
    if selected.empty:
        return 0.0
    count = float(len(selected))
    duplicate_corridors = count - float(selected["optimization_corridor_unit"].astype(str).nunique())
    duplicate_routes = count - float(selected["optimization_route_unit"].astype(str).nunique())
    duplicate_guilds = count - float(selected["optimization_primary_guild"].astype(str).nunique())
    return _clip01(((0.45 * duplicate_corridors) + (0.25 * duplicate_routes) + (0.15 * duplicate_guilds)) / count)


def _ensure_optimizer_columns(gdf) -> None:
    defaults = {
        "selected_flag": 0,
        "selection_rank": None,
        "selection_phase": None,
        "planning_status": "candidate",
        "optimizer_strategy": None,
        "optimizer_marginal_gain": None,
        "optimizer_gain_base_score": None,
        "optimizer_gain_route_coverage": None,
        "optimizer_gain_habitat_representation": None,
        "optimizer_gain_high_risk_coverage": None,
        "optimizer_gain_uncertainty_reduction": None,
        "optimizer_penalty_redundancy": None,
        "optimizer_nearest_selected_m": None,
        "optimization_route_unit": None,
        "optimization_corridor_unit": None,
        "optimization_primary_guild": None,
        "optimization_target_scenario": None,
        "optimization_high_risk_flag": 0,
        "optimization_high_risk_score": 0.0,
    }
    for col, default in defaults.items():
        if col not in gdf.columns:
            gdf[col] = default


def _prepare_optimizer_features(gdf, *, settings) -> None:
    import pandas as pd

    gdf["optimization_corridor_unit"] = gdf["source_hf_uid"].astype(str)
    gdf["optimization_target_scenario"] = str(getattr(settings, "target_scenario", "all_bats") or "all_bats")
    if settings.section_column and settings.section_column in gdf.columns:
        gdf["optimization_route_unit"] = gdf[settings.section_column].astype(str)
    else:
        gdf["optimization_route_unit"] = gdf["source_hf_uid"].astype(str)

    if gdf["optimization_target_scenario"].astype(str).iloc[0] != "all_bats" and "planning_target_group" in gdf.columns:
        gdf["optimization_primary_guild"] = gdf["planning_target_group"].astype("string").fillna("unknown")
    elif "eco_primary_guild" in gdf.columns:
        gdf["optimization_primary_guild"] = gdf["eco_primary_guild"].astype("string").fillna("unknown")
    else:
        gdf["optimization_primary_guild"] = "unknown"

    base_score = pd.to_numeric(gdf.get("candidate_score"), errors="coerce").fillna(0.0)
    gdf["_optimizer_base_score_norm"] = _normalize_series(base_score)

    source_length = pd.to_numeric(gdf.get("source_length_m"), errors="coerce").fillna(0.0)
    gdf["_optimizer_route_length_norm"] = _normalize_series(source_length)

    if "planning_priority_score" in gdf.columns:
        high_risk_score = pd.to_numeric(gdf["planning_priority_score"], errors="coerce").fillna(0.0)
    elif "eco_suitability_score" in gdf.columns:
        high_risk_score = pd.to_numeric(gdf["eco_suitability_score"], errors="coerce").fillna(0.0)
    else:
        high_risk_score = base_score
    gdf["optimization_high_risk_score"] = high_risk_score.astype(float)

    eligible_scores = high_risk_score.loc[gdf["eligible_for_selection"].astype(int) == 1]
    if len(eligible_scores) > 0:
        threshold = float(eligible_scores.quantile(_clip01(float(settings.high_risk_quantile))))
    else:
        threshold = 0.0
    threshold = max(threshold, 0.65 if float(high_risk_score.max()) <= 1.0 else threshold)
    gdf["optimization_high_risk_flag"] = (
        pd.to_numeric(gdf["optimization_high_risk_score"], errors="coerce").fillna(0.0) >= threshold
    ).astype(int)

    if "evidence_confidence_score" in gdf.columns:
        confidence = pd.to_numeric(gdf["evidence_confidence_score"], errors="coerce").fillna(0.5)
        gdf["_optimizer_uncertainty"] = (1.0 - confidence.clip(lower=0.0, upper=1.0)).astype(float)
    else:
        gdf["_optimizer_uncertainty"] = 0.0

    if "eco_primary_guild" in gdf.columns:
        gdf["_optimizer_guild_strength"] = gdf.apply(_primary_guild_strength, axis=1).astype(float)
    else:
        gdf["_optimizer_guild_strength"] = 0.0


def _choose_next_candidate(gdf, *, settings, state: _SelectionState, section_filter: str | None):
    subset = gdf[
        (gdf["eligible_for_selection"].astype(int) == 1)
        & (gdf["selected_flag"].astype(int) == 0)
    ]
    if section_filter is not None:
        subset = subset[subset["optimization_route_unit"].astype(str) == str(section_filter)]
    if subset.empty:
        return None, None

    scored_rows: list[tuple[float, float, float, str, float, str, str, object, dict[str, float]]] = []
    for idx, row in subset.iterrows():
        if not _passes_spacing(row.geometry, state.selected_geoms, float(settings.min_detector_spacing_m)):
            continue
        metrics = _marginal_metrics(row, settings=settings, state=state)
        scored_rows.append(
            (
                float(metrics["total_gain"]),
                float(metrics["route_gain"]),
                float(metrics["high_risk_gain"]),
                str(row["optimization_route_unit"]),
                float(row["_optimizer_base_score_norm"]),
                str(row["optimization_primary_guild"]),
                str(row["candidate_id"]),
                idx,
                metrics,
            )
        )

    if not scored_rows:
        return None, None

    scored_rows.sort(
        key=lambda item: (
            -item[0],
            -item[1],
            -item[2],
            item[3],
            -item[4],
            item[5],
            item[6],
        )
    )
    _, _, _, _, _, _, _, idx, metrics = scored_rows[0]
    return idx, metrics


def _marginal_metrics(row, *, settings, state: _SelectionState) -> dict[str, float]:
    route_unit = str(row["optimization_route_unit"])
    corridor_unit = str(row["optimization_corridor_unit"])
    guild = str(row["optimization_primary_guild"])

    base_gain = float(row["_optimizer_base_score_norm"])
    route_gain = _route_coverage_gain(row, state=state)
    habitat_gain = _habitat_representation_gain(row, state=state)
    high_risk_gain = _high_risk_coverage_gain(row, state=state)
    uncertainty_gain = _uncertainty_reduction_gain(row, state=state)
    redundancy_penalty, nearest_selected_m = _redundancy_penalty(row, settings=settings, state=state)

    route_weight = float(settings.objective_weight_route_coverage)
    if getattr(settings, "objective_weight_corridor_coverage", None) is not None:
        route_weight += float(settings.objective_weight_corridor_coverage)
    total_gain = (
        (float(settings.objective_weight_base_score) * base_gain)
        + (route_weight * route_gain)
        + (float(settings.objective_weight_habitat_representation) * habitat_gain)
        + (float(settings.objective_weight_high_risk_coverage) * high_risk_gain)
        + (float(settings.objective_weight_uncertainty_reduction) * uncertainty_gain)
        - (float(settings.objective_weight_redundancy_penalty) * redundancy_penalty)
    )
    return {
        "route_unit": route_unit,
        "corridor_unit": corridor_unit,
        "guild": guild,
        "base_gain": _clip01(base_gain),
        "route_gain": _clip01(route_gain),
        "habitat_gain": _clip01(habitat_gain),
        "high_risk_gain": _clip01(high_risk_gain),
        "uncertainty_gain": _clip01(uncertainty_gain),
        "redundancy_penalty": _clip01(redundancy_penalty),
        "nearest_selected_m": nearest_selected_m,
        "total_gain": float(total_gain),
    }


def _route_coverage_gain(row, *, state: _SelectionState) -> float:
    route_unit = str(row["optimization_route_unit"])
    corridor_unit = str(row["optimization_corridor_unit"])
    route_length = float(row["_optimizer_route_length_norm"])

    gain = 0.0
    if route_unit not in state.selected_route_units:
        gain += 0.65
    if corridor_unit not in state.selected_corridor_units:
        corridor_bonus = 0.35 * (0.5 + (0.5 * route_length))
        if corridor_unit == route_unit:
            corridor_bonus *= 0.55
        gain += corridor_bonus
    return _clip01(gain)


def _habitat_representation_gain(row, *, state: _SelectionState) -> float:
    guild = str(row["optimization_primary_guild"])
    if guild == "unknown":
        return 0.0
    represented_count = int(state.guild_counts.get(guild, 0))
    guild_strength = float(row["_optimizer_guild_strength"])
    missing_bonus = 0.65 if represented_count == 0 else 0.0
    diminishing_bonus = guild_strength / float(1 + represented_count)
    return _clip01(missing_bonus + (0.35 * diminishing_bonus))


def _high_risk_coverage_gain(row, *, state: _SelectionState) -> float:
    corridor_unit = str(row["optimization_corridor_unit"])
    if int(row["optimization_high_risk_flag"]) != 1:
        return 0.0
    risk_strength = _clip01(float(row["optimization_high_risk_score"]))
    if corridor_unit not in state.selected_high_risk_corridors:
        return _clip01((0.75 * risk_strength) + (0.25 * float(row["_optimizer_base_score_norm"])))
    if corridor_unit not in state.selected_corridor_units:
        return _clip01(0.35 * risk_strength)
    return 0.0


def _uncertainty_reduction_gain(row, *, state: _SelectionState) -> float:
    corridor_unit = str(row["optimization_corridor_unit"])
    uncertainty = _clip01(float(row["_optimizer_uncertainty"]))
    if uncertainty <= 0:
        return 0.0
    base = 0.55 + (0.45 * float(row["_optimizer_base_score_norm"]))
    if corridor_unit not in state.selected_corridor_units:
        multiplier = 1.0
    else:
        multiplier = 0.45
    return _clip01(uncertainty * base * multiplier)


def _redundancy_penalty(row, *, settings, state: _SelectionState) -> tuple[float, float | None]:
    route_unit = str(row["optimization_route_unit"])
    corridor_unit = str(row["optimization_corridor_unit"])
    guild = str(row["optimization_primary_guild"])

    penalty = 0.0
    nearest_selected_m = None
    if corridor_unit in state.selected_corridor_units:
        penalty += 0.45
    if route_unit in state.selected_route_units:
        penalty += 0.20
    guild_count = int(state.guild_counts.get(guild, 0))
    if guild_count > 0 and guild != "unknown":
        penalty += min(0.25, 0.08 * guild_count)
    if state.selected_geoms:
        nearest_selected_m = min(float(row.geometry.distance(existing)) for existing in state.selected_geoms)
        soft_limit = max(
            float(settings.min_detector_spacing_m) * max(float(settings.soft_spacing_multiplier), 1.0),
            float(settings.min_detector_spacing_m),
        )
        if soft_limit > float(settings.min_detector_spacing_m):
            closeness = 1.0 - (
                (nearest_selected_m - float(settings.min_detector_spacing_m))
                / (soft_limit - float(settings.min_detector_spacing_m))
            )
            penalty += 0.35 * _clip01(closeness)
    return _clip01(penalty), nearest_selected_m


def _apply_selection(gdf, *, idx, rank: int, phase: str, state: _SelectionState, metrics: dict[str, float], settings) -> None:
    route_unit = str(metrics["route_unit"])
    corridor_unit = str(metrics["corridor_unit"])
    guild = str(metrics["guild"])

    gdf.at[idx, "selected_flag"] = 1
    gdf.at[idx, "selection_rank"] = int(rank)
    gdf.at[idx, "selection_phase"] = phase
    gdf.at[idx, "planning_status"] = "selected"
    gdf.at[idx, "optimizer_strategy"] = _optimizer_version_for_settings(settings)
    gdf.at[idx, "optimizer_marginal_gain"] = float(metrics["total_gain"])
    gdf.at[idx, "optimizer_gain_base_score"] = float(metrics["base_gain"])
    gdf.at[idx, "optimizer_gain_route_coverage"] = float(metrics["route_gain"])
    gdf.at[idx, "optimizer_gain_habitat_representation"] = float(metrics["habitat_gain"])
    gdf.at[idx, "optimizer_gain_high_risk_coverage"] = float(metrics["high_risk_gain"])
    gdf.at[idx, "optimizer_gain_uncertainty_reduction"] = float(metrics["uncertainty_gain"])
    gdf.at[idx, "optimizer_penalty_redundancy"] = float(metrics["redundancy_penalty"])
    gdf.at[idx, "optimizer_nearest_selected_m"] = metrics["nearest_selected_m"]

    state.selected_route_units.add(route_unit)
    state.selected_corridor_units.add(corridor_unit)
    if int(gdf.at[idx, "optimization_high_risk_flag"]) == 1:
        state.selected_high_risk_corridors.add(corridor_unit)
    state.guild_counts[guild] = state.guild_counts.get(guild, 0) + 1
    state.selected_geoms.append(gdf.at[idx, gdf.geometry.name])


def _optimizer_version_for_settings(settings) -> str:
    strategy = str(getattr(settings, "optimizer_strategy", "greedy") or "greedy").lower()
    if strategy == "exact":
        return "exact_integer_v1"
    return str(getattr(settings, "optimizer_version", "greedy_coverage_v1"))


def _passes_spacing(geom, selected_geoms, min_spacing_m: float) -> bool:
    if not selected_geoms:
        return True
    threshold = max(float(min_spacing_m), 0.0)
    return all(float(geom.distance(existing)) >= threshold for existing in selected_geoms)


def _normalize_series(series):
    minimum = float(series.min()) if len(series) else 0.0
    maximum = float(series.max()) if len(series) else 0.0
    if maximum <= minimum:
        if maximum > 0:
            return series.astype(float) / maximum
        return series.astype(float) * 0.0
    return (series.astype(float) - minimum) / (maximum - minimum)


def _primary_guild_strength(row) -> float:
    guild = str(row.get("optimization_primary_guild", "unknown"))
    if guild in {"edge_open", "edge_commuter"}:
        value = row.get("eco_guild_edge_score")
    elif guild == "open_air":
        value = row.get("eco_guild_open_air_score")
    elif guild == "clutter_linear":
        value = row.get("eco_guild_clutter_score")
    elif guild in {"woodland_sensitive", "woodland_specialist"}:
        value = row.get("eco_guild_woodland_score")
    else:
        return 0.0
    try:
        score = float(value)
    except Exception:
        return 0.0
    if score != score:
        return 0.0
    return _clip01(score)


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)
