from __future__ import annotations

from dataclasses import dataclass
from typing import Any


EVIDENCE_ENGINE_VERSION = "bankable_species_v1"
GUIDANCE_REGIME_VERSION = "bct4_ne2025_england_v1"

TARGET_SPECS = (
    {
        "slug": "edge_commuter",
        "label": "Edge commuter",
        "kind": "guild",
        "group": "edge_commuter",
        "model_status": "Ready",
    },
    {
        "slug": "open_air",
        "label": "Open air",
        "kind": "guild",
        "group": "open_air",
        "model_status": "Ready",
    },
    {
        "slug": "clutter_linear",
        "label": "Clutter linear",
        "kind": "guild",
        "group": "clutter_linear",
        "model_status": "Ready",
    },
    {
        "slug": "woodland_specialist",
        "label": "Woodland specialist",
        "kind": "guild",
        "group": "woodland_specialist",
        "model_status": "Ready",
    },
    {
        "slug": "common_pipistrelle",
        "label": "Common pipistrelle",
        "kind": "species",
        "group": "edge_commuter",
        "model_status": "Interim model",
    },
    {
        "slug": "soprano_pipistrelle",
        "label": "Soprano pipistrelle",
        "kind": "species",
        "group": "edge_commuter",
        "model_status": "Interim model",
    },
    {
        "slug": "noctule",
        "label": "Noctule",
        "kind": "species",
        "group": "open_air",
        "model_status": "Interim model",
    },
    {
        "slug": "serotine",
        "label": "Serotine",
        "kind": "species",
        "group": "open_air",
        "model_status": "Interim model",
    },
    {
        "slug": "barbastelle",
        "label": "Barbastelle",
        "kind": "species",
        "group": "woodland_specialist",
        "model_status": "Interim model",
    },
    {
        "slug": "myotis_spp",
        "label": "Myotis spp.",
        "kind": "grouped_taxon",
        "group": "clutter_linear",
        "model_status": "Grouped taxon",
    },
    {
        "slug": "plecotus_spp",
        "label": "Plecotus spp.",
        "kind": "grouped_taxon",
        "group": "clutter_linear",
        "model_status": "Grouped taxon",
    },
)


@dataclass(slots=True)
class PlanningEvidenceResult:
    gdf: Any
    summary: dict[str, Any]


def add_planning_evidence_scores(hedges_gdf, *, settings) -> PlanningEvidenceResult:
    import pandas as pd

    gdf = hedges_gdf.copy()
    _ensure_evidence_columns(gdf)
    feature_columns = _evidence_feature_columns()
    target_catalogue = [dict(spec) for spec in TARGET_SPECS]
    target_specs = {str(spec["slug"]): dict(spec) for spec in TARGET_SPECS}
    target_scenario = str(getattr(settings, "target_scenario", "all_bats") or "all_bats")
    if target_scenario != "all_bats" and target_scenario not in target_specs:
        raise ValueError(f"Unsupported planner target scenario '{target_scenario}'.")

    primary_guilds: list[str] = []
    confidence_levels: list[str] = []
    data_quality_states: list[str] = []
    target_domain_states: list[str] = []

    for idx, row in gdf.iterrows():
        values, presence = _feature_values(row)
        guild_scores = _guild_scores(values)
        primary_guild = max(sorted(guild_scores), key=lambda name: guild_scores[name])
        coverage_pct, missing_count = _coverage_metrics(row, feature_columns)
        confidence_score, confidence_level = _confidence_metrics(
            coverage_pct=coverage_pct,
            missing_count=missing_count,
            row=row,
            settings=settings,
        )
        data_quality_score, data_quality_state = _data_quality_metrics(
            values=values,
            coverage_pct=coverage_pct,
            confidence_score=confidence_score,
            row=row,
        )
        utility = _survey_utility_score(
            values=values,
            primary_guild=primary_guild,
            confidence_score=confidence_score,
            data_quality_score=data_quality_score,
        )

        target_outputs: dict[str, dict[str, Any]] = {}
        for spec in TARGET_SPECS:
            target_outputs[str(spec["slug"])] = _target_output(
                spec=spec,
                guild_scores=guild_scores,
                values=values,
                presence=presence,
                utility=utility,
                coverage_pct=coverage_pct,
                confidence_score=confidence_score,
                data_quality_score=data_quality_score,
            )

        all_bats_output = _all_bats_output(
            guild_scores=guild_scores,
            values=values,
            utility=utility,
            coverage_pct=coverage_pct,
            confidence_score=confidence_score,
            data_quality_score=data_quality_score,
        )
        active_output = all_bats_output if target_scenario == "all_bats" else target_outputs[target_scenario]
        overall_reason_codes = _overall_reason_codes(
            values=values,
            coverage_pct=coverage_pct,
            confidence_level=confidence_level,
            primary_guild=primary_guild,
            data_quality_state=data_quality_state,
        )

        _write_target_columns(gdf, idx=idx, target_outputs=target_outputs)
        gdf.at[idx, "eco_guild_edge_score"] = guild_scores["edge_commuter"]
        gdf.at[idx, "eco_guild_open_air_score"] = guild_scores["open_air"]
        gdf.at[idx, "eco_guild_clutter_score"] = guild_scores["clutter_linear"]
        gdf.at[idx, "eco_guild_woodland_score"] = guild_scores["woodland_specialist"]
        gdf.at[idx, "eco_primary_guild"] = primary_guild
        gdf.at[idx, "eco_suitability_score"] = all_bats_output["relative_suitability"]
        gdf.at[idx, "survey_utility_score"] = utility
        gdf.at[idx, "planning_priority_score"] = active_output["survey_priority"]
        gdf.at[idx, "planning_priority_rank_score"] = active_output["survey_priority"]
        gdf.at[idx, "planning_target_scenario"] = target_scenario
        gdf.at[idx, "planning_target_label"] = str(active_output["label"])
        gdf.at[idx, "planning_target_group"] = str(active_output["group"])
        gdf.at[idx, "planning_target_model_status"] = str(active_output["model_status"])
        gdf.at[idx, "planning_target_domain_status"] = str(active_output["domain_status"])
        gdf.at[idx, "planning_target_reason_codes"] = "|".join(active_output["reason_codes"])
        gdf.at[idx, "evidence_feature_coverage_pct"] = round(coverage_pct, 4)
        gdf.at[idx, "evidence_missing_feature_count"] = int(missing_count)
        gdf.at[idx, "evidence_confidence_score"] = round(confidence_score, 4)
        gdf.at[idx, "evidence_confidence_level"] = confidence_level
        gdf.at[idx, "data_quality_score"] = round(data_quality_score, 4)
        gdf.at[idx, "data_quality_state"] = data_quality_state
        gdf.at[idx, "evidence_domain_status"] = active_output["domain_status"]
        gdf.at[idx, "evidence_reason_codes"] = "|".join(overall_reason_codes)
        gdf.at[idx, "evidence_engine_version"] = EVIDENCE_ENGINE_VERSION
        gdf.at[idx, "guidance_regime_version"] = getattr(settings, "guidance_regime_version", GUIDANCE_REGIME_VERSION)

        primary_guilds.append(primary_guild)
        confidence_levels.append(confidence_level)
        data_quality_states.append(data_quality_state)
        target_domain_states.append(str(active_output["domain_status"]))

    summary = {
        "evidence_engine_version": EVIDENCE_ENGINE_VERSION,
        "guidance_regime_version": getattr(settings, "guidance_regime_version", GUIDANCE_REGIME_VERSION),
        "row_count": int(len(gdf)),
        "planning_target_scenario": target_scenario,
        "primary_guild_counts": _counts(primary_guilds),
        "confidence_level_counts": _counts(confidence_levels),
        "data_quality_state_counts": _counts(data_quality_states),
        "planning_target_domain_counts": _counts(target_domain_states),
        "target_catalogue": target_catalogue,
        "mean_planning_priority_score": (
            round(float(pd.to_numeric(gdf["planning_priority_score"], errors="coerce").mean()), 6)
            if len(gdf) > 0
            else None
        ),
    }
    return PlanningEvidenceResult(gdf=gdf, summary=summary)


def _ensure_evidence_columns(gdf) -> None:
    base_columns = (
        "eco_guild_edge_score",
        "eco_guild_open_air_score",
        "eco_guild_clutter_score",
        "eco_guild_woodland_score",
        "eco_primary_guild",
        "eco_suitability_score",
        "survey_utility_score",
        "planning_priority_score",
        "planning_priority_rank_score",
        "planning_target_scenario",
        "planning_target_label",
        "planning_target_group",
        "planning_target_model_status",
        "planning_target_domain_status",
        "planning_target_reason_codes",
        "evidence_feature_coverage_pct",
        "evidence_missing_feature_count",
        "evidence_confidence_score",
        "evidence_confidence_level",
        "data_quality_score",
        "data_quality_state",
        "evidence_domain_status",
        "evidence_reason_codes",
        "evidence_engine_version",
        "guidance_regime_version",
    )
    for col in base_columns:
        if col not in gdf.columns:
            gdf[col] = None
    for spec in TARGET_SPECS:
        slug = str(spec["slug"])
        for suffix in ("relative_suitability", "survey_priority", "domain_status", "reason_codes", "model_status"):
            column_name = f"{slug}_{suffix}"
            if column_name not in gdf.columns:
                gdf[column_name] = None


def _write_target_columns(gdf, *, idx, target_outputs: dict[str, dict[str, Any]]) -> None:
    for slug, output in target_outputs.items():
        gdf.at[idx, f"{slug}_relative_suitability"] = output["relative_suitability"]
        gdf.at[idx, f"{slug}_survey_priority"] = output["survey_priority"]
        gdf.at[idx, f"{slug}_domain_status"] = output["domain_status"]
        gdf.at[idx, f"{slug}_reason_codes"] = "|".join(output["reason_codes"])
        gdf.at[idx, f"{slug}_model_status"] = output["model_status"]


def _feature_values(row) -> tuple[dict[str, float], dict[str, bool]]:
    feature_map = {
        "tree": ("buf100_worldcover_tree_pct", "mhb_corridor10_tree_pct"),
        "built": ("buf100_worldcover_built_pct", "mhb_corridor10_built_pct"),
        "water": ("buf100_worldcover_water_pct", "mhb_corridor10_water_pct"),
        "wetland": ("buf100_worldcover_wetland_pct", "buf100_phi_wetland_pct", "mhb_corridor10_wetland_pct"),
        "phi_woodland": ("buf100_phi_broadleaved_woodland_pct", "buf500_phi_broadleaved_woodland_pct"),
        "roost_signal": ("roostpx_struct_proxy_score",),
        "river_prox": ("dist_os_river_m",),
        "awi_prox": ("dist_awi_ancwood_m",),
        "nightlight": ("buf100_nightlight_mean",),
        "road_density": ("buf100_os_road_density_m_per_ha",),
        "shelter": ("mhb_dem_shelter_idx_100m",),
        "slope": ("mhb_dem_slope_mean_50m",),
        "connectivity": ("net_degree_max",),
        "sinuosity": ("geom_sinuosity",),
        "length_signal": ("geom_length_m",),
        "hedge_height_mean": ("hedge_struct_height_mean_5m",),
        "hedge_height_p90": ("hedge_struct_height_p90_5m",),
        "hedge_continuity": ("hedge_struct_canopy_continuity_10m",),
        "hedge_gap": ("hedge_struct_gap_fraction_10m",),
        "tree_standards": ("hedge_struct_tree_standard_pct_10m",),
        "hedge_width": ("hedge_struct_width_proxy_m",),
        "project_lighting_dist": ("dist_project_lighting_m",),
    }
    values: dict[str, float] = {}
    presence: dict[str, bool] = {}
    for key, columns in feature_map.items():
        raw = _first_present(row, *columns)
        presence[key] = raw is not None
        values[key] = 0.0 if raw is None else float(raw)

    values["river_prox"] = _invdist(_first_present(row, "dist_os_river_m"), scale=150.0)
    values["awi_prox"] = _invdist(_first_present(row, "dist_awi_ancwood_m"), scale=250.0)
    values["night_darkness"] = 1.0 - _scaled(_first_present(row, "buf100_nightlight_mean"), max_value=60.0)
    values["road_quietness"] = 1.0 - _scaled(_first_present(row, "buf100_os_road_density_m_per_ha"), max_value=120.0)
    values["project_darkness"] = 1.0 - _invdist(_first_present(row, "dist_project_lighting_m"), scale=80.0)
    values["connectivity"] = _scaled(_first_present(row, "net_degree_max"), max_value=4.0)
    values["sinuosity"] = _scaled(((_first_present(row, "geom_sinuosity") or 1.0) - 1.0), max_value=1.5)
    values["length_signal"] = _scaled(_first_present(row, "geom_length_m"), max_value=250.0)
    values["hedge_height_mean"] = _scaled(_first_present(row, "hedge_struct_height_mean_5m"), max_value=8.0)
    values["hedge_height_p90"] = _scaled(_first_present(row, "hedge_struct_height_p90_5m"), max_value=10.0)
    values["hedge_continuity"] = _clip01(values["hedge_continuity"])
    values["hedge_gap"] = _clip01(values["hedge_gap"])
    values["tree_standards"] = _clip01(values["tree_standards"])
    values["hedge_width"] = _scaled(_first_present(row, "hedge_struct_width_proxy_m"), max_value=6.0)
    values["shelter"] = max(
        _clip01(values["shelter"]),
        1.0 - _scaled(_first_present(row, "mhb_dem_slope_mean_50m"), max_value=25.0),
    )
    values["dark_corridor"] = _clip01((0.60 * values["night_darkness"]) + (0.40 * values["project_darkness"]))
    values["lighting_risk"] = _clip01(1.0 - values["dark_corridor"])
    values["severance_risk"] = _clip01(
        (0.55 * (1.0 - values["road_quietness"]))
        + (0.25 * _clip01(values["built"]))
        + (0.20 * values["lighting_risk"])
    )
    values["hedge_quality"] = _clip01(
        (0.30 * values["hedge_continuity"])
        + (0.25 * values["hedge_height_mean"])
        + (0.15 * values["hedge_width"])
        + (0.20 * values["tree_standards"])
        + (0.10 * (1.0 - values["hedge_gap"]))
    )
    presence["night_darkness"] = presence["nightlight"]
    presence["project_darkness"] = presence["project_lighting_dist"]
    presence["dark_corridor"] = presence["nightlight"] or presence["project_lighting_dist"]
    presence["hedge_quality"] = (
        presence["hedge_height_mean"]
        or presence["hedge_height_p90"]
        or presence["hedge_continuity"]
        or presence["tree_standards"]
        or presence["hedge_width"]
    )
    return values, presence


def _guild_scores(values: dict[str, float]) -> dict[str, float]:
    edge_commuter = _clip01(
        (0.18 * values["tree"])
        + (0.14 * values["river_prox"])
        + (0.08 * values["water"])
        + (0.07 * values["wetland"])
        + (0.16 * values["roost_signal"])
        + (0.13 * values["dark_corridor"])
        + (0.08 * values["road_quietness"])
        + (0.07 * values["connectivity"])
        + (0.09 * values["hedge_quality"])
    )
    open_air = _clip01(
        (0.16 * values["length_signal"])
        + (0.14 * values["connectivity"])
        + (0.12 * values["river_prox"])
        + (0.10 * values["water"])
        + (0.10 * values["wetland"])
        + (0.12 * values["roost_signal"])
        + (0.08 * values["dark_corridor"])
        + (0.10 * values["tree_standards"])
        + (0.08 * values["road_quietness"])
    )
    clutter_linear = _clip01(
        (0.18 * values["tree"])
        + (0.10 * values["wetland"])
        + (0.10 * values["river_prox"])
        + (0.14 * values["dark_corridor"])
        + (0.16 * values["shelter"])
        + (0.12 * values["connectivity"])
        + (0.14 * values["hedge_quality"])
        + (0.06 * values["roost_signal"])
    )
    woodland_specialist = _clip01(
        (0.20 * values["phi_woodland"])
        + (0.16 * values["awi_prox"])
        + (0.14 * values["tree"])
        + (0.12 * values["dark_corridor"])
        + (0.10 * values["road_quietness"])
        + (0.10 * values["shelter"])
        + (0.13 * values["hedge_quality"])
        + (0.05 * values["wetland"])
    )
    return {
        "edge_commuter": edge_commuter,
        "open_air": open_air,
        "clutter_linear": clutter_linear,
        "woodland_specialist": woodland_specialist,
    }


def _target_output(
    *,
    spec: dict[str, Any],
    guild_scores: dict[str, float],
    values: dict[str, float],
    presence: dict[str, bool],
    utility: float,
    coverage_pct: float,
    confidence_score: float,
    data_quality_score: float,
) -> dict[str, Any]:
    slug = str(spec["slug"])
    group = str(spec["group"])
    if slug in guild_scores:
        suitability = guild_scores[slug]
    elif slug == "common_pipistrelle":
        suitability = _clip01(
            (0.72 * guild_scores["edge_commuter"])
            + (0.12 * values["roost_signal"])
            + (0.08 * values["hedge_quality"])
            - (0.04 * values["severance_risk"])
        )
    elif slug == "soprano_pipistrelle":
        suitability = _clip01(
            (0.64 * guild_scores["edge_commuter"])
            + (0.12 * values["water"])
            + (0.10 * values["wetland"])
            + (0.06 * values["river_prox"])
        )
    elif slug == "noctule":
        suitability = _clip01(
            (0.72 * guild_scores["open_air"])
            + (0.10 * values["tree_standards"])
            + (0.08 * values["length_signal"])
            - (0.05 * values["hedge_gap"])
        )
    elif slug == "serotine":
        suitability = _clip01(
            (0.52 * guild_scores["open_air"])
            + (0.18 * guild_scores["edge_commuter"])
            + (0.10 * values["roost_signal"])
            + (0.06 * values["hedge_width"])
            - (0.04 * values["lighting_risk"])
        )
    elif slug == "barbastelle":
        suitability = _clip01(
            (0.75 * guild_scores["woodland_specialist"])
            + (0.08 * values["dark_corridor"])
            + (0.06 * values["hedge_quality"])
            + (0.05 * values["tree_standards"])
            - (0.06 * values["lighting_risk"])
        )
    elif slug == "myotis_spp":
        suitability = _clip01(
            (0.78 * guild_scores["clutter_linear"])
            + (0.08 * values["water"])
            + (0.06 * values["wetland"])
        )
    elif slug == "plecotus_spp":
        suitability = _clip01(
            (0.72 * guild_scores["clutter_linear"])
            + (0.10 * values["roost_signal"])
            + (0.08 * values["dark_corridor"])
        )
    else:
        suitability = guild_scores[group]

    domain_status = _target_domain_status(
        spec=spec,
        presence=presence,
        coverage_pct=coverage_pct,
        confidence_score=confidence_score,
        data_quality_score=data_quality_score,
    )
    survey_priority = _clip01(
        (0.68 * suitability)
        + (0.22 * utility)
        + (0.10 * data_quality_score)
    )
    reason_codes = _target_reason_codes(
        spec=spec,
        values=values,
        domain_status=domain_status,
        suitability=suitability,
    )
    return {
        "slug": slug,
        "label": spec["label"],
        "group": group,
        "model_status": spec["model_status"],
        "relative_suitability": suitability,
        "survey_priority": survey_priority,
        "domain_status": domain_status,
        "reason_codes": reason_codes,
    }


def _all_bats_output(
    *,
    guild_scores: dict[str, float],
    values: dict[str, float],
    utility: float,
    coverage_pct: float,
    confidence_score: float,
    data_quality_score: float,
) -> dict[str, Any]:
    ranked = sorted(float(v) for v in guild_scores.values())
    top = ranked[-1]
    mid = ranked[-2] if len(ranked) > 1 else top
    suitability = _clip01((0.60 * top) + (0.25 * mid) + (0.15 * values["dark_corridor"]))
    domain_status = _domain_label((0.55 * coverage_pct) + (0.25 * confidence_score) + (0.20 * data_quality_score))
    survey_priority = _clip01((0.68 * suitability) + (0.22 * utility) + (0.10 * data_quality_score))
    return {
        "slug": "all_bats",
        "label": "All bats",
        "group": "multi_guild",
        "model_status": "Mixed guild evidence",
        "relative_suitability": suitability,
        "survey_priority": survey_priority,
        "domain_status": domain_status,
        "reason_codes": [f"PRIMARY_GUILD_{max(sorted(guild_scores), key=lambda name: guild_scores[name]).upper()}"],
    }


def _coverage_metrics(row, feature_columns: list[str]) -> tuple[float, int]:
    present = 0
    for col in feature_columns:
        if col not in row.index:
            continue
        value = row.get(col)
        if value is None:
            continue
        try:
            if value != value:
                continue
        except Exception:
            pass
        if str(value).strip() == "":
            continue
        present += 1
    total = max(len(feature_columns), 1)
    coverage_pct = present / float(total)
    return coverage_pct, int(total - present)


def _confidence_metrics(*, coverage_pct: float, missing_count: int, row, settings) -> tuple[float, str]:
    penalty = 0.0
    for flag_col in ("phi_coverage_flag", "awi_coverage_flag"):
        if flag_col in row.index:
            value = str(row.get(flag_col)).strip().lower()
            if value not in {"1", "true", "yes", "covered", "ok"}:
                penalty += 0.08
    if "hedge_struct_status" in row.index:
        if str(row.get("hedge_struct_status", "")).strip().lower() not in {"ok"}:
            penalty += 0.06
    confidence_score = _clip01(coverage_pct - penalty - (0.02 * min(missing_count, 4)))
    if confidence_score >= 0.8:
        return confidence_score, "High"
    if confidence_score >= 0.55:
        return confidence_score, "Medium"
    return confidence_score, "Low"


def _data_quality_metrics(*, values: dict[str, float], coverage_pct: float, confidence_score: float, row) -> tuple[float, str]:
    score = _clip01(
        (0.45 * coverage_pct)
        + (0.35 * confidence_score)
        + (0.10 * values["dark_corridor"])
        + (0.10 * values["hedge_quality"])
    )
    if "hedge_struct_status" in row.index and str(row.get("hedge_struct_status", "")).strip().lower() != "ok":
        score = _clip01(score - 0.08)
    if score >= 0.75:
        return score, "Measured"
    if score >= 0.55:
        return score, "Mixed"
    return score, "Weak"


def _survey_utility_score(*, values: dict[str, float], primary_guild: str, confidence_score: float, data_quality_score: float) -> float:
    representation_signal = max(
        values["connectivity"],
        values["river_prox"],
        values["phi_woodland"],
        values["hedge_quality"],
    )
    uncertainty_signal = _clip01(1.0 - confidence_score)
    guild_bonus_map = {
        "edge_commuter": values["hedge_quality"],
        "open_air": values["length_signal"],
        "clutter_linear": values["shelter"],
        "woodland_specialist": values["phi_woodland"],
    }
    guild_bonus = _clip01(guild_bonus_map.get(primary_guild, 0.0))
    return _clip01(
        (0.44 * representation_signal)
        + (0.18 * uncertainty_signal)
        + (0.16 * values["dark_corridor"])
        + (0.12 * guild_bonus)
        + (0.10 * data_quality_score)
    )


def _target_domain_status(
    *,
    spec: dict[str, Any],
    presence: dict[str, bool],
    coverage_pct: float,
    confidence_score: float,
    data_quality_score: float,
) -> str:
    required_keys = {
        "edge_commuter": ("tree", "roost_signal", "river_prox"),
        "open_air": ("connectivity", "length_signal"),
        "clutter_linear": ("tree", "shelter", "hedge_continuity"),
        "woodland_specialist": ("phi_woodland", "awi_prox", "hedge_continuity"),
        "common_pipistrelle": ("tree", "roost_signal", "hedge_continuity"),
        "soprano_pipistrelle": ("water", "wetland", "tree"),
        "noctule": ("tree_standards", "length_signal"),
        "serotine": ("roost_signal", "hedge_width"),
        "barbastelle": ("phi_woodland", "awi_prox", "dark_corridor"),
        "myotis_spp": ("water", "wetland", "hedge_continuity"),
        "plecotus_spp": ("roost_signal", "dark_corridor", "hedge_continuity"),
    }
    slug = str(spec["slug"])
    required = required_keys.get(slug, required_keys.get(str(spec["group"]), ()))
    if required:
        present_fraction = sum(1 for key in required if presence.get(key, False)) / float(len(required))
    else:
        present_fraction = 1.0
    domain_score = _clip01(
        (0.40 * coverage_pct)
        + (0.25 * confidence_score)
        + (0.20 * data_quality_score)
        + (0.15 * present_fraction)
    )
    return _domain_label(domain_score)


def _target_reason_codes(*, spec: dict[str, Any], values: dict[str, float], domain_status: str, suitability: float) -> list[str]:
    codes = [f"TARGET_{str(spec['slug']).upper()}"]
    if domain_status != "Inside":
        codes.append(f"DOMAIN_{domain_status.upper()}")
    if suitability >= 0.7:
        codes.append("RELATIVE_SUITABILITY_STRONG")
    if values["dark_corridor"] >= 0.6:
        codes.append("DARK_CORRIDOR_STRONG")
    if values["hedge_quality"] >= 0.6:
        codes.append("HEDGE_STRUCTURE_STRONG")
    if values["lighting_risk"] >= 0.6:
        codes.append("LIGHTING_RISK_HIGH")
    return codes


def _overall_reason_codes(*, values: dict[str, float], coverage_pct: float, confidence_level: str, primary_guild: str, data_quality_state: str) -> list[str]:
    codes: list[str] = [f"PRIMARY_GUILD_{primary_guild.upper()}"]
    if values["tree"] >= 0.4:
        codes.append("TREE_CORRIDOR_STRONG")
    if max(values["water"], values["river_prox"], values["wetland"]) >= 0.45:
        codes.append("WATER_CONTEXT_STRONG")
    if max(values["phi_woodland"], values["awi_prox"]) >= 0.45:
        codes.append("WOODLAND_CONTEXT_STRONG")
    if values["roost_signal"] >= 0.55:
        codes.append("ROOST_PROXY_STRONG")
    if max(values["connectivity"], values["sinuosity"]) >= 0.55:
        codes.append("LINEAR_CONNECTIVITY_STRONG")
    if values["hedge_quality"] >= 0.6:
        codes.append("HEDGE_STRUCTURE_STRONG")
    if values["lighting_risk"] >= 0.55:
        codes.append("LIGHT_DISTURBANCE_HIGH")
    if values["severance_risk"] >= 0.55:
        codes.append("SEVERANCE_RISK_HIGH")
    if coverage_pct < 0.55:
        codes.append("LOW_EVIDENCE_COVERAGE")
    if confidence_level == "Low":
        codes.append("LOW_EVIDENCE_CONFIDENCE")
    if data_quality_state == "Weak":
        codes.append("WEAK_DATA_SUPPORT")
    return codes


def _evidence_feature_columns() -> list[str]:
    return [
        "geom_length_m",
        "geom_sinuosity",
        "net_degree_max",
        "dist_os_river_m",
        "dist_awi_ancwood_m",
        "dist_project_lighting_m",
        "buf100_worldcover_tree_pct",
        "buf100_worldcover_built_pct",
        "buf100_worldcover_water_pct",
        "buf100_worldcover_wetland_pct",
        "buf100_nightlight_mean",
        "buf100_os_road_density_m_per_ha",
        "roostpx_struct_proxy_score",
        "mhb_dem_shelter_idx_100m",
        "buf100_phi_broadleaved_woodland_pct",
        "hedge_struct_height_mean_5m",
        "hedge_struct_height_p90_5m",
        "hedge_struct_canopy_continuity_10m",
        "hedge_struct_gap_fraction_10m",
        "hedge_struct_tree_standard_pct_10m",
        "hedge_struct_width_proxy_m",
    ]


def _first_present(row, *columns: str) -> float | None:
    for col in columns:
        value = _num(row, col)
        if value is not None:
            return value
    return None


def _num(row, col: str) -> float | None:
    if col not in row.index:
        return None
    value = row.get(col)
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def _scaled(value: float | None, *, max_value: float) -> float:
    if value is None or max_value <= 0:
        return 0.0
    return _clip01(value / float(max_value))


def _invdist(value: float | None, *, scale: float) -> float:
    if value is None:
        return 0.0
    return _clip01(1.0 / (1.0 + (max(value, 0.0) / float(scale))))


def _domain_label(score: float) -> str:
    if score >= 0.78:
        return "Inside"
    if score >= 0.55:
        return "Borderline"
    return "Outside"


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))
