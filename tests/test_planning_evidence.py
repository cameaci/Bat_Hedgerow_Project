import pytest

gpd = pytest.importorskip("geopandas")
from shapely.geometry import LineString  # noqa: E402

from hedge_features.planning import PlanningSettings, add_planning_evidence_scores, plan_static_detectors  # noqa: E402


def _enriched_hedges():
    return gpd.GeoDataFrame(
        {
            "hf_uid": ["h1", "h2"],
            "geom_length_m": [140.0, 120.0],
            "geom_sinuosity": [1.25, 1.08],
            "net_degree_max": [4, 2],
            "dist_os_river_m": [20.0, 250.0],
            "dist_awi_ancwood_m": [30.0, 500.0],
            "buf100_worldcover_tree_pct": [0.65, 0.25],
            "buf100_worldcover_built_pct": [0.05, 0.45],
            "buf100_worldcover_water_pct": [0.12, 0.01],
            "buf100_worldcover_wetland_pct": [0.10, 0.00],
            "buf100_nightlight_mean": [4.0, 35.0],
            "buf100_os_road_density_m_per_ha": [10.0, 60.0],
            "roostpx_struct_proxy_score": [0.70, 0.25],
            "mhb_roost_proxy_score": [0.70, 0.25],
            "mhb_dem_shelter_idx_100m": [0.72, 0.20],
            "buf100_phi_broadleaved_woodland_pct": [0.45, 0.00],
            "buf100_phi_wetland_pct": [0.08, 0.00],
            "phi_coverage_flag": [1, 1],
            "awi_coverage_flag": [1, 1],
        },
        geometry=[
            LineString([(0, 0), (140, 0)]),
            LineString([(0, 200), (120, 200)]),
        ],
        crs="EPSG:27700",
    )


def test_evidence_engine_populates_planning_columns():
    settings = PlanningSettings(detector_budget=1)
    result = add_planning_evidence_scores(_enriched_hedges(), settings=settings)
    out = result.gdf

    required = {
        "eco_guild_edge_score",
        "eco_guild_open_air_score",
        "eco_guild_clutter_score",
        "eco_guild_woodland_score",
        "eco_primary_guild",
        "eco_suitability_score",
        "survey_utility_score",
        "planning_priority_score",
        "planning_target_scenario",
        "planning_target_domain_status",
        "common_pipistrelle_relative_suitability",
        "barbastelle_survey_priority",
        "evidence_confidence_level",
        "evidence_reason_codes",
        "evidence_engine_version",
    }
    assert required.issubset(set(out.columns))
    assert result.summary["evidence_engine_version"] == "bankable_species_v1"
    assert out.loc[0, "planning_priority_score"] > out.loc[1, "planning_priority_score"]
    assert out.loc[0, "eco_primary_guild"] == "clutter_linear"
    assert out.loc[0, "planning_target_scenario"] == "all_bats"
    assert out.loc[0, "planning_target_domain_status"] in {"Inside", "Borderline", "Outside"}


def test_planner_uses_computed_priority_score_by_default():
    settings = PlanningSettings(
        detector_budget=1,
        candidate_spacing_m=50.0,
        endpoint_offset_m=10.0,
        min_detector_spacing_m=80.0,
    )
    result = plan_static_detectors(_enriched_hedges(), settings=settings)

    assert "planning_priority_score" in result.candidates_gdf.columns
    assert "survey_utility_score" in result.candidates_gdf.columns
    assert "evidence_confidence_level" in result.candidates_gdf.columns
    top_source = str(result.selected_gdf.iloc[0]["source_hf_uid"])
    assert top_source == "h1"
    candidate_rows = result.candidates_gdf[result.candidates_gdf["source_hf_uid"] == "h1"]
    assert candidate_rows["candidate_score"].astype(float).eq(candidate_rows["planning_priority_score"].astype(float)).all()


def test_duplicate_roost_proxy_column_does_not_change_evidence_score():
    hedges_a = _enriched_hedges()
    hedges_b = _enriched_hedges()
    hedges_a["mhb_roost_proxy_score"] = [0.0, 1.0]
    hedges_b["mhb_roost_proxy_score"] = [1.0, 0.0]

    settings = PlanningSettings(detector_budget=1)
    result_a = add_planning_evidence_scores(hedges_a, settings=settings).gdf
    result_b = add_planning_evidence_scores(hedges_b, settings=settings).gdf

    assert result_a["planning_priority_score"].astype(float).tolist() == result_b["planning_priority_score"].astype(float).tolist()
    assert result_a["eco_suitability_score"].astype(float).tolist() == result_b["eco_suitability_score"].astype(float).tolist()


def test_target_scenario_changes_planning_priority_projection():
    hedges = _enriched_hedges()

    common = add_planning_evidence_scores(
        hedges,
        settings=PlanningSettings(detector_budget=1, target_scenario="common_pipistrelle"),
    ).gdf
    barbastelle = add_planning_evidence_scores(
        hedges,
        settings=PlanningSettings(detector_budget=1, target_scenario="barbastelle"),
    ).gdf

    assert common["planning_target_scenario"].astype(str).unique().tolist() == ["common_pipistrelle"]
    assert barbastelle["planning_target_scenario"].astype(str).unique().tolist() == ["barbastelle"]
    assert common["planning_priority_score"].astype(float).tolist() != barbastelle["planning_priority_score"].astype(float).tolist()
