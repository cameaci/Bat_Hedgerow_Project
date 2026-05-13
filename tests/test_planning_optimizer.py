import pytest

gpd = pytest.importorskip("geopandas")
from shapely.geometry import Point  # noqa: E402

from hedge_features.planning.optimizer import select_detector_locations  # noqa: E402
from hedge_features.planning.schema import PlanningSettings  # noqa: E402


def _candidate_gdf():
    return gpd.GeoDataFrame(
        {
            "candidate_id": ["c1", "c2", "c3"],
            "source_hf_uid": ["h1", "h2", "h3"],
            "section_name": ["A", "A", "B"],
            "candidate_chainage_m": [20.0, 20.0, 20.0],
            "source_length_m": [250.0, 250.0, 250.0],
            "candidate_score": [1.00, 0.98, 0.82],
            "eligible_for_selection": [1, 1, 1],
            "eco_primary_guild": ["edge_open", "edge_open", "woodland_sensitive"],
            "eco_guild_edge_score": [0.90, 0.88, 0.25],
            "eco_guild_clutter_score": [0.50, 0.45, 0.30],
            "eco_guild_woodland_score": [0.20, 0.18, 0.86],
            "planning_priority_score": [0.95, 0.94, 0.80],
            "evidence_confidence_score": [0.90, 0.88, 0.35],
            "selected_flag": [0, 0, 0],
            "planning_status": ["candidate", "candidate", "candidate"],
        },
        geometry=[Point(0, 0), Point(200, 0), Point(400, 0)],
        crs="EPSG:27700",
    )


def _redundancy_candidate_gdf():
    return gpd.GeoDataFrame(
        {
            "candidate_id": ["c1", "c1b", "c2"],
            "source_hf_uid": ["h1", "h1", "h2"],
            "section_name": ["A", "A", "B"],
            "candidate_chainage_m": [20.0, 220.0, 20.0],
            "source_length_m": [300.0, 300.0, 260.0],
            "candidate_score": [0.97, 0.95, 0.90],
            "eligible_for_selection": [1, 1, 1],
            "eco_primary_guild": ["edge_open", "edge_open", "clutter_linear"],
            "eco_guild_edge_score": [0.90, 0.88, 0.40],
            "eco_guild_clutter_score": [0.45, 0.44, 0.82],
            "eco_guild_woodland_score": [0.15, 0.14, 0.20],
            "planning_priority_score": [0.93, 0.91, 0.86],
            "evidence_confidence_score": [0.85, 0.84, 0.55],
            "selected_flag": [0, 0, 0],
            "planning_status": ["candidate", "candidate", "candidate"],
        },
        geometry=[Point(0, 0), Point(220, 0), Point(460, 0)],
        crs="EPSG:27700",
    )


def test_optimizer_prefers_route_and_habitat_coverage_over_top_k_score():
    settings = PlanningSettings(
        detector_budget=2,
        min_detector_spacing_m=100.0,
        section_column="section_name",
        use_evidence_engine=False,
        score_column="candidate_score",
    )
    selected, scored = select_detector_locations(_candidate_gdf(), settings=settings)

    assert selected["candidate_id"].astype(str).tolist() == ["c1", "c3"]
    assert set(selected["optimization_route_unit"].astype(str)) == {"A", "B"}
    assert set(selected["optimization_primary_guild"].astype(str)) == {"edge_open", "woodland_sensitive"}
    assert scored.loc[scored["candidate_id"] == "c1", "optimizer_strategy"].iloc[0] == "greedy_coverage_v1"


def test_optimizer_penalizes_redundant_same_corridor_selection():
    settings = PlanningSettings(
        detector_budget=2,
        min_detector_spacing_m=100.0,
        section_column="section_name",
        use_evidence_engine=False,
        score_column="candidate_score",
    )
    selected, _ = select_detector_locations(_redundancy_candidate_gdf(), settings=settings)

    assert selected["candidate_id"].astype(str).tolist() == ["c1", "c2"]
    assert selected["source_hf_uid"].astype(str).tolist() == ["h1", "h2"]


def test_optimizer_selection_is_deterministic_for_same_candidate_set():
    settings = PlanningSettings(
        detector_budget=2,
        min_detector_spacing_m=100.0,
        section_column="section_name",
        use_evidence_engine=False,
        score_column="candidate_score",
    )
    selected1, scored1 = select_detector_locations(_candidate_gdf(), settings=settings)
    selected2, scored2 = select_detector_locations(_candidate_gdf(), settings=settings)

    assert selected1["candidate_id"].astype(str).tolist() == selected2["candidate_id"].astype(str).tolist()
    assert scored1["optimization_high_risk_flag"].astype(int).tolist() == scored2["optimization_high_risk_flag"].astype(int).tolist()
    assert selected1["optimizer_marginal_gain"].astype(float).tolist() == selected2["optimizer_marginal_gain"].astype(float).tolist()
