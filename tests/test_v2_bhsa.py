from __future__ import annotations

import pandas as pd

from hedge_features.v2 import BHSAScoringSettings, score_bhsa_table


def test_field_bhsa_scores_thresholds_and_major_road_downgrade():
    df = pd.DataFrame(
        {
            "hedgerow_id": ["h1"],
            "hedge_height_m": [2.2],
            "hedge_width_m": [1.6],
            "gappiness_pct": [8.0],
            "arable_margin_m": [6.0],
            "trees_per_50m": [7],
            "woody_species_count_20m": [8],
            "wet_ditch_present": [True],
            "major_road_distance_m": [35.0],
        }
    )

    out, summary = score_bhsa_table(df, settings=BHSAScoringSettings(mode="field"))

    row = out.iloc[0]
    assert row["bhsa_si1_score"] == 3
    assert row["bhsa_si4_score"] == 4
    assert row["bhsa_si7_score"] == 2
    assert row["bhsa_class_raw"] == "Excellent"
    assert row["bhsa_class"] == "Good"
    assert row["bhsa_major_road_downgraded"] == 1
    assert row["bhsa_confidence_level"] == "High"
    assert bool(row["field_verification_required"]) is False
    assert summary["class_counts"]["Good"] == 1


def test_proxy_bhsa_flags_si6_and_si7_as_field_verification_required():
    df = pd.DataFrame(
        {
            "hedgerow_id": ["h1"],
            "hedge_struct_height_mean_5m": [2.4],
            "hedge_struct_width_proxy_m": [1.8],
            "hedge_struct_gap_fraction_10m": [0.05],
            "buf100_worldcover_cropland_pct": [0.45],
            "hedge_struct_tree_standard_pct_10m": [0.50],
            "buf100_worldcover_tree_pct": [0.70],
            "buf100_phi_broadleaved_woodland_pct": [0.25],
            "dist_awi_ancwood_m": [40.0],
            "buf100_os_river_density_m_per_ha": [12.0],
        }
    )

    out, _ = score_bhsa_table(df, settings=BHSAScoringSettings(mode="proxy"))

    row = out.iloc[0]
    assert row["bhsa_score"] is not None
    assert row["bhsa_si6_source"] == "proxy"
    assert row["bhsa_si7_source"] == "proxy"
    assert row["bhsa_confidence_level"] == "Low"
    assert bool(row["field_verification_required"]) is True
    assert "SI6 woody species diversity is not remotely verifiable" in row["bhsa_notes"]


def test_field_mode_without_si6_si7_is_incomplete():
    df = pd.DataFrame(
        {
            "hedgerow_id": ["h1"],
            "hedge_height_m": [2.1],
            "hedge_width_m": [1.7],
            "gappiness_pct": [5.0],
            "arable_margin_m": [3.0],
            "trees_per_50m": [4],
        }
    )

    out, _ = score_bhsa_table(df, settings=BHSAScoringSettings(mode="field"))

    assert out.iloc[0]["bhsa_class"] == "Incomplete"
    assert out.iloc[0]["bhsa_score"] is None
    assert "Woody species diversity" in out.iloc[0]["bhsa_missing_reasons"]
    assert "Wet ditch" in out.iloc[0]["bhsa_missing_reasons"]
