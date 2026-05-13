import pytest

gpd = pytest.importorskip("geopandas")
from shapely.geometry import LineString, Polygon  # noqa: E402

from hedge_features.planning import PlanningSettings, plan_static_detectors  # noqa: E402


def _hedges():
    return gpd.GeoDataFrame(
        {
            "hf_uid": ["h1", "h2", "h3"],
            "survey_priority_score": [0.9, 0.5, 0.8],
            "section_name": ["A", "B", "B"],
            "access_ok": [1, 1, 0],
        },
        geometry=[
            LineString([(0, 0), (100, 0)]),
            LineString([(0, 200), (100, 200)]),
            LineString([(0, 400), (100, 400)]),
        ],
        crs="EPSG:27700",
    )


def test_planner_generates_deterministic_candidates_and_selection():
    settings = PlanningSettings(
        detector_budget=2,
        candidate_spacing_m=40.0,
        endpoint_offset_m=10.0,
        min_detector_spacing_m=80.0,
        score_column="survey_priority_score",
    )
    result1 = plan_static_detectors(_hedges(), settings=settings)
    result2 = plan_static_detectors(_hedges(), settings=settings)

    assert result1.candidates_gdf["candidate_id"].tolist() == result2.candidates_gdf["candidate_id"].tolist()
    assert result1.selected_gdf["candidate_id"].tolist() == result2.selected_gdf["candidate_id"].tolist()
    assert result1.run_summary["planning_run_id"] == result2.run_summary["planning_run_id"]
    assert len(result1.selected_gdf) == 2


def test_planner_respects_access_and_section_minimums():
    settings = PlanningSettings(
        detector_budget=2,
        candidate_spacing_m=40.0,
        endpoint_offset_m=10.0,
        min_detector_spacing_m=80.0,
        score_column="survey_priority_score",
        access_flag_column="access_ok",
        section_column="section_name",
        section_minimum_counts={"B": 1},
    )
    result = plan_static_detectors(_hedges(), settings=settings)

    selected_sections = result.selected_gdf["section_name"].astype(str).tolist()
    selected_hedges = result.selected_gdf["source_hf_uid"].astype(str).tolist()
    assert "B" in selected_sections
    assert "h3" not in selected_hedges
    assert "h1" in selected_hedges
    assert "h2" in selected_hedges


def test_planner_applies_include_and_exclude_constraints():
    hedges = _hedges()
    include_area = gpd.GeoDataFrame(
        {"name": ["include"]},
        geometry=[Polygon([(-10, -10), (120, -10), (120, 260), (-10, 260)])],
        crs="EPSG:27700",
    )
    exclude_area = gpd.GeoDataFrame(
        {"name": ["exclude"]},
        geometry=[Polygon([(35, -20), (65, -20), (65, 20), (35, 20)])],
        crs="EPSG:27700",
    )
    settings = PlanningSettings(
        detector_budget=2,
        candidate_spacing_m=40.0,
        endpoint_offset_m=10.0,
        min_detector_spacing_m=80.0,
        score_column="survey_priority_score",
    )
    result = plan_static_detectors(
        hedges,
        settings=settings,
        include_area_gdf=include_area,
        exclude_area_gdf=exclude_area,
    )

    assert (result.candidates_gdf["within_include_area"].astype(int) == 0).any()
    assert (result.candidates_gdf["outside_exclude_area"].astype(int) == 0).any()
    assert (result.candidates_gdf["eligible_for_selection"].astype(int) == 0).any()
