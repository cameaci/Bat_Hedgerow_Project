import json
from pathlib import Path

import pytest

gpd = pytest.importorskip("geopandas")
from shapely.geometry import LineString  # noqa: E402

from hedge_features.planning import PlanningSettings, plan_static_detectors, write_planning_outputs  # noqa: E402


def _hedges():
    return gpd.GeoDataFrame(
        {
            "hf_uid": ["h1", "h2"],
            "survey_priority_score": [0.85, 0.45],
            "section_name": ["North", "South"],
        },
        geometry=[
            LineString([(0, 0), (100, 0)]),
            LineString([(0, 200), (100, 200)]),
        ],
        crs="EPSG:27700",
    )


def test_write_planning_outputs_creates_evidence_pack_artifacts(tmp_path):
    result = plan_static_detectors(
        _hedges(),
        settings=PlanningSettings(
            detector_budget=1,
            candidate_spacing_m=40.0,
            endpoint_offset_m=10.0,
            min_detector_spacing_m=80.0,
            score_column="survey_priority_score",
        ),
    )

    output_path = tmp_path / "static_plan.gpkg"
    written = write_planning_outputs(
        result,
        output_path,
        source_name="enriched.gpkg",
        source_metadata={
            "profile_name": "bats_bankable_england_v2",
            "dataset_snapshot": "test-snapshot",
            "guidance_regime_version": "bct4_ne2025_england_v1",
            "data_catalogue": {"summary": {"enabled_dataset_count": 4}},
            "feature_health": {"summary": {"counts_by_support_state": {"measured_or_derived": 8}}},
        },
    )

    assert set(written) == {
        "screened_gpkg",
        "candidate_gpkg",
        "chosen_detector_set",
        "run_manifest",
        "evidence_report",
        "data_catalogue",
        "feature_health",
        "method_statement",
    }
    for path in written.values():
        assert Path(path).exists()

    manifest = json.loads((tmp_path / "static_plan_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["framework_versions"]["optimizer_version"] == "greedy_coverage_v1"
    assert manifest["dataset_provenance"]["profile_name"] == "bats_bankable_england_v2"
    assert manifest["guidance_regime_version"] == "bct4_ne2025_england_v1"
    assert manifest["outputs"]["screened_gpkg"].endswith("static_plan.gpkg")

    report_text = (tmp_path / "static_plan_evidence_report.md").read_text(encoding="utf-8")
    assert "Why Selected" in report_text
    assert "Why Not Selected" in report_text
    assert "Dataset Provenance" in report_text
    assert "Data Health" in report_text

    method_text = (tmp_path / "static_plan_method_statement.md").read_text(encoding="utf-8")
    assert "Method Statement" in method_text
    assert "Target scenario" in method_text

    candidates_gdf = gpd.read_file(tmp_path / "static_plan_candidates.gpkg")
    screened_gdf = gpd.read_file(tmp_path / "static_plan.gpkg")
    selected_gdf = gpd.read_file(tmp_path / "static_plan_selected.gpkg")

    assert "bank_why_selected" in candidates_gdf.columns
    assert "bank_why_not_selected" in candidates_gdf.columns
    assert "bank_confidence" in candidates_gdf.columns
    assert "bank_domain_status" in candidates_gdf.columns
    assert "bank_selected_candidate_count" in screened_gdf.columns
    assert len(selected_gdf) == 1
