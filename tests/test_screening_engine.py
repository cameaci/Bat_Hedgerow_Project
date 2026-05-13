from pathlib import Path

import pandas as pd

from hedge_features.screening import ScreeningSettings, load_framework_bundle, screen_dataframe
from hedge_features.screening.confidence import evaluate_row_confidence
from hedge_features.screening.engine import LOW_CONFIDENCE_ACTION, align_predictors, build_column_audit
from hedge_features.screening.io import read_attribute_table


def _sample_enriched_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hf_uid": ["h1", "h2", "h3"],
            "geom_length_m": [120.0, 250.0, 80.0],
            "geom_endpoint_dist_m": [90.0, 170.0, 60.0],
            "geom_sinuosity": [1.2, 1.45, 1.1],
            "net_degree_max": [3, 4, 2],
            "dist_os_road_m": [40.0, 200.0, 10.0],
            "dist_os_river_m": [30.0, 400.0, 20.0],
            "buf100_os_road_density_m_per_ha": [25.0, 50.0, 15.0],
            "buf100_worldcover_tree_pct": [0.55, 0.20, 0.70],
            "buf100_worldcover_built_pct": [0.10, 0.45, 0.05],
            "buf100_worldcover_water_pct": [0.05, 0.00, 0.10],
            "buf100_worldcover_wetland_pct": [0.15, 0.00, 0.20],
            "buf100_nightlight_mean": [10.0, 40.0, 5.0],
            "roostpx_struct_proxy_score": [0.70, 0.20, 0.85],
            "mhb_roost_proxy_score": [0.65, 0.30, 0.80],
            "mhb_dem_shelter_idx_100m": [0.60, 0.30, 0.75],
            "geom_centroid_x": [1000.0, 2000.0, 3000.0],
            "geom_centroid_y": [1000.0, 2000.0, 3000.0],
            "net_component_id": [1, 1, 2],
            "deploy_status": ["ok", "missing_columns", "ok"],
            "roostpx_status": ["ok", "ok", "ok"],
            "depwx_status": ["ok", "ok", "ok"],
            "moon_status": ["ok", "ok", "ok"],
            "phi_coverage_flag": [1, 0, 1],
            "awi_coverage_flag": [1, 1, 1],
            "Static_HSI_Class": ["Good", "Poor", "Good"],
            "HyNet_ProjectCode": ["A", "A", "B"],
            "Autumn22_presence": [1, 0, 1],
            "Autumn22_SurveyFlag": [1, 1, 1],
            "_merge": ["both", "both", "both"],
        }
    )


def test_column_governance_excludes_forbidden_and_keeps_strict_gis_predictors():
    framework = load_framework_bundle("bats_screening_v1")
    df = _sample_enriched_df()
    audit = build_column_audit(df, framework=framework, settings=ScreeningSettings())

    assert "geom_length_m" in audit.detected_gis_predictor_columns
    assert "geom_sinuosity" in audit.detected_gis_predictor_columns
    assert audit.excluded_columns["geom_centroid_x"] == "forbidden_exact"
    assert audit.excluded_columns["geom_centroid_y"] == "forbidden_exact"
    assert audit.excluded_columns["net_component_id"] == "forbidden_exact"
    assert audit.excluded_columns["deploy_status"] == "status_confidence_only"
    assert audit.excluded_columns["Static_HSI_Class"] == "field_survey_non_transfer"
    assert audit.excluded_columns["HyNet_ProjectCode"] == "project_admin_non_transfer"
    assert audit.excluded_columns["Autumn22_SurveyFlag"] == "survey_label_or_metadata"
    assert "deploy_status" in audit.status_columns_found
    assert "phi_coverage_flag" in audit.coverage_flags_found


def test_feature_alignment_preserves_registry_order_and_flags_missing_required_optional():
    framework = load_framework_bundle("bats_screening_v1")
    df = _sample_enriched_df().drop(columns=["geom_sinuosity", "buf250_worldcover_tree_pct"], errors="ignore")
    alignment = align_predictors(df, framework=framework)

    assert list(alignment.predictor_df.columns) == framework.feature_registry.predictor_order
    assert "geom_sinuosity" in alignment.missing_required_features
    assert "buf250_worldcover_tree_pct" in alignment.missing_optional_features
    assert alignment.predictor_df["geom_sinuosity"].isna().all()


def test_confidence_logic_low_coverage_multiple_reasons_and_clean_rows():
    framework = load_framework_bundle("bats_screening_v1")
    registry = framework.feature_registry
    rules = framework.confidence_rules

    low_cov = evaluate_row_confidence(
        source_row={"_merge": "both"},
        predictor_row={},
        missing_required_feature_count=0,
        gis_feature_coverage_pct=0.2,
        registry=registry,
        confidence_rules=rules,
        strictness="Standard",
    )
    assert low_cov["confidence_level"] == "Low"
    assert "LOW_GIS_COVERAGE" in low_cov["reason_codes"]

    multi = evaluate_row_confidence(
        source_row={"_merge": "left_only", "deploy_status": "missing_columns"},
        predictor_row={"geom_length_m": 999999.0},
        missing_required_feature_count=1,
        gis_feature_coverage_pct=0.8,
        registry=registry,
        confidence_rules=rules,
        strictness="Standard",
    )
    assert multi["confidence_level"] == "Low"
    assert multi["major_reason_code_count"] >= 2

    clean = evaluate_row_confidence(
        source_row={"_merge": "both", "deploy_status": "ok"},
        predictor_row={"geom_length_m": 120.0, "geom_sinuosity": 1.2},
        missing_required_feature_count=0,
        gis_feature_coverage_pct=0.9,
        registry=registry,
        confidence_rules=rules,
        strictness="Standard",
    )
    assert clean["confidence_level"] == "High"
    assert clean["reason_codes"] == []


def test_policy_defaults_to_recall_first_and_low_confidence_override_runs_before_action_threshold():
    framework = load_framework_bundle("bats_screening_v1")
    df = _sample_enriched_df()

    # Force row 0 to remain high-scoring but low-confidence via missing required predictor.
    df.loc[0, "geom_sinuosity"] = None
    df.loc[0, "deploy_status"] = "ok"
    df.loc[0, "_merge"] = "both"
    df.loc[0, "phi_coverage_flag"] = 1
    df.loc[0, "awi_coverage_flag"] = 1
    df.loc[0, "roostpx_struct_proxy_score"] = 1.0
    df.loc[0, "mhb_roost_proxy_score"] = 1.0
    df.loc[0, "buf100_worldcover_tree_pct"] = 0.95
    df.loc[0, "buf100_worldcover_built_pct"] = 0.0
    df.loc[0, "dist_os_river_m"] = 5.0
    df.loc[0, "dist_os_road_m"] = 5.0

    out = screen_dataframe(
        df,
        framework=framework,
        settings=ScreeningSettings(custom_policy_threshold=0.0),
    )
    row0 = out.results_df.iloc[0]

    assert set(out.results_df["screening_policy"]) == {"Recall-first"}
    assert row0["confidence_level"] == "Low"
    assert row0["recommended_action"] == LOW_CONFIDENCE_ACTION
    assert row0["survey_priority_score"] >= row0["policy_threshold"]


def test_end_to_end_csv_and_xlsx_inputs_produce_required_screening_columns(tmp_path: Path):
    framework = load_framework_bundle("bats_screening_v1")
    df = _sample_enriched_df()
    required_cols = {
        "framework_name",
        "framework_version",
        "feature_profile_name",
        "feature_profile_version",
        "analysis_run_id",
        "analysis_timestamp_utc",
        "prediction_route",
        "screening_policy",
        "survey_priority_score",
        "survey_priority_band",
        "confidence_level",
        "reason_codes",
        "recommended_action",
        "gis_feature_coverage_pct",
        "missing_required_feature_count",
        "major_reason_code_count",
        "band_threshold_low",
        "band_threshold_high",
        "policy_threshold",
    }

    csv_path = tmp_path / "enriched.csv"
    df.to_csv(csv_path, index=False)
    loaded_csv = read_attribute_table(csv_path)
    run_csv = screen_dataframe(loaded_csv, framework=framework)
    assert required_cols.issubset(set(run_csv.results_df.columns))
    assert set(run_csv.results_df["prediction_route"]) == {"uploaded_enriched_table"}

    openpyxl = __import__("importlib").util.find_spec("openpyxl")
    if openpyxl is None:
        return
    xlsx_path = tmp_path / "enriched.xlsx"
    df.to_excel(xlsx_path, index=False)
    loaded_xlsx = read_attribute_table(xlsx_path)
    run_xlsx = screen_dataframe(loaded_xlsx, framework=framework)
    assert required_cols.issubset(set(run_xlsx.results_df.columns))
    assert set(run_xlsx.results_df["prediction_route"]) == {"uploaded_enriched_table"}
