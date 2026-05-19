from __future__ import annotations

import pandas as pd

from hedge_features.v2 import (
    BHSACalibrationSettings,
    StaticAcousticSummarySettings,
    build_validation_diagnostics,
    build_v2_evidence_pack,
    calibrate_bhsa_weights,
    parse_acoustic_survey_table,
    score_bhsa_table,
    summarise_static_acoustics,
)


def test_static_acoustic_summary_groups_by_hedgerow_season_species():
    detections = pd.DataFrame(
        {
            "hedgerow_id": ["h1", "h1", "h1", "h2"],
            "species": ["Pipistrellus", "Pipistrellus", "Myotis", "Noctule"],
            "timestamp": ["2026-05-01T22:00:00Z", "2026-05-02T22:00:00Z", "2026-05-01T23:00:00Z", "2026-09-01T21:00:00Z"],
            "passes": [4, 6, 3, 10],
        }
    )

    out, summary = summarise_static_acoustics(
        detections,
        settings=StaticAcousticSummarySettings(datetime_column="timestamp", activity_column="passes"),
    )

    pip = out.loc[(out["hedgerow_id"] == "h1") & (out["acoustic_species"] == "Pipistrellus")].iloc[0]
    assert pip["survey_season"] == "Spring"
    assert pip["acoustic_total_passes"] == 10.0
    assert pip["acoustic_nights"] == 2
    assert pip["acoustic_passes_per_night"] == 5.0
    assert summary["hedgerow_count"] == 2
    assert "detector_model" in summary["missing_effort_metadata_fields"]
    assert "acoustic_hedgerow_season_total_passes" in out.columns


def test_acoustic_parser_maps_common_aliases_from_text_table():
    text = """HedgeID,Night,Season,Taxon,Calls,Static_ID,Recorder_Model,Mic_Height_m,Verified
h1,2026-05-01,Spring,Pipistrellus,7,D1,SM4BAT,2.0,verified
"""

    out, audit = parse_acoustic_survey_table(text)

    assert out.loc[0, "hedgerow_id"] == "h1"
    assert out.loc[0, "survey_season"] == "Spring"
    assert out.loc[0, "acoustic_species"] == "Pipistrellus"
    assert out.loc[0, "acoustic_passes"] == 7
    assert audit["column_mapping"]["detector_model"] == "Recorder_Model"
    assert audit["missing_effort_metadata_fields"] == []


def test_static_acoustic_summary_flags_missing_effort_metadata():
    detections = pd.DataFrame(
        {
            "hedgerow_id": ["h1"],
            "species": ["Myotis"],
            "datetime": ["2026-07-01T22:00:00Z"],
            "passes": [3],
        }
    )

    out, summary = summarise_static_acoustics(
        detections,
        settings=StaticAcousticSummarySettings(activity_column="passes"),
    )

    assert out.loc[0, "detector_effort_complete"] == 0
    assert "missing_detector_model" in out.loc[0, "acoustic_qa_comparability_flag"]
    assert set(summary["missing_effort_metadata_fields"]) == {"detector_id", "detector_model", "microphone_height_m"}


def test_calibration_scaffold_fits_weights_when_paired_data_are_sufficient():
    rows = []
    for idx in range(40):
        high = int(idx >= 20)
        row = {f"bhsa_si{i}_score": (3 if high else 1) for i in range(1, 8)}
        row["high_activity_label"] = high
        rows.append(row)
    df = pd.DataFrame(rows)

    result = calibrate_bhsa_weights(df, settings=BHSACalibrationSettings(min_sample_size=20))

    assert result["status"] == "ready_for_technical_review"
    assert result["do_not_use_calibrated_model"] is False
    assert set(result["fitted_weights"]) == {f"bhsa_si{i}_score" for i in range(1, 8)}
    assert result["cross_validation"]["auc_mean"] == 1.0
    assert result["equal_prior_baseline"]["auc"] == 1.0


def test_calibration_warns_when_class_balance_is_insufficient():
    rows = []
    for idx in range(30):
        high = int(idx == 29)
        row = {f"bhsa_si{i}_score": (3 if high else 1) for i in range(1, 8)}
        row["high_activity_label"] = high
        rows.append(row)

    result = calibrate_bhsa_weights(
        pd.DataFrame(rows),
        settings=BHSACalibrationSettings(min_sample_size=20, min_class_count=5),
    )

    assert result["do_not_use_calibrated_model"] is True
    assert "class balance is too weak" in " ".join(result["warnings"])


def test_validation_diagnostics_reports_confusion_matrix_and_auc():
    df = pd.DataFrame(
        {
            "hedgerow_id": ["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10"],
            "bhsa_score": [2.8, 2.6, 2.2, 1.8, 1.4, 1.3, 2.5, 1.2, 1.9, 1.1],
            "bhsa_class": ["Excellent", "Excellent", "Good", "Good", "Poor", "Poor", "Excellent", "Poor", "Good", "Poor"],
            "acoustic_total_passes": [20, 10, 0, 4, 0, 8, 0, 0, 3, 0],
        }
    )

    result = build_validation_diagnostics(df)

    assert result["status"] == "ready_for_technical_review"
    assert result["confusion_matrix"]["true_positive"] == 4
    assert result["confusion_matrix"]["false_positive_high_score_no_evidence"] == 2
    assert result["confusion_matrix"]["false_negative_low_score_positive_evidence"] == 1
    assert result["metrics"]["sensitivity"] == 0.8
    assert result["metrics"]["auc"] is not None


def test_v2_evidence_pack_contains_manifest_and_method_statement(tmp_path):
    bhsa, _ = score_bhsa_table(
        pd.DataFrame(
            {
                "hedgerow_id": ["h1"],
                "hedge_height_m": [2.2],
                "hedge_width_m": [1.6],
                "gappiness_pct": [8.0],
                "arable_margin_m": [6.0],
                "trees_per_50m": [7],
                "woody_species_count_20m": [8],
                "wet_ditch_present": [True],
            }
        )
    )

    validation = build_validation_diagnostics(
        bhsa.assign(acoustic_total_passes=[5.0]),
    )
    pack = build_v2_evidence_pack(
        bhsa_gdf=bhsa,
        readiness_report={"status": "ready"},
        validation_diagnostics=validation,
        output_dir=tmp_path,
    )

    assert pack["manifest"]["evidence_pack_version"] == "bat_hedgerow_intelligence_pack_v2"
    assert pack["manifest"]["bhsa"]["row_count"] == 1
    assert "Bat Hedgerow Intelligence Platform V2 Method Statement" in pack["method_statement_md"]
    assert "## Validation Diagnostics" in pack["method_statement_md"]
    assert (tmp_path / "v2_run_manifest.json").exists()
    assert (tmp_path / "v2_bhsa_decision_table.csv").exists()
    assert (tmp_path / "v2_field_verification_table.csv").exists()
    assert (tmp_path / "v2_acoustic_validation_summary.csv").exists()
    assert (tmp_path / "v2_detector_deployment_rationale.csv").exists()
    assert list(pack["tables"]["field_verification_table"].columns) == [
        "hedgerow_id",
        "hf_uid",
        "section_id",
        "bhsa_class",
        "bhsa_confidence_level",
        "field_verification_required",
        "verification_reason",
        "bhsa_missing_reasons",
        "bhsa_notes",
        "bhsa_si6_source",
        "bhsa_si6_confidence",
        "bhsa_si7_source",
        "bhsa_si7_confidence",
    ]
