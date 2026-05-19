from __future__ import annotations

import pandas as pd

from hedge_features.v2 import (
    BHSACalibrationSettings,
    StaticAcousticSummarySettings,
    build_v2_evidence_pack,
    calibrate_bhsa_weights,
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


def test_v2_evidence_pack_contains_manifest_and_method_statement():
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

    pack = build_v2_evidence_pack(bhsa_gdf=bhsa, readiness_report={"status": "ready"})

    assert pack["manifest"]["evidence_pack_version"] == "bat_hedgerow_intelligence_pack_v2"
    assert pack["manifest"]["bhsa"]["row_count"] == 1
    assert "Bat Hedgerow Intelligence Platform V2 Method Statement" in pack["method_statement_md"]
