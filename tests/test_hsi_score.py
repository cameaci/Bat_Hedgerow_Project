"""Tests for the HSI scoring engine (pure tabular maths, no GIS required)."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from hsi import config
from hsi.score import (
    _normalise_si,
    _si_arable_margin,
    _si_gap,
    _si_height,
    _si_species,
    _si_trees,
    _si_width,
    apply_scoring,
    resolve_structural,
)


def _row(**si):
    """Build a 1-row DataFrame using direct pre-scored SI columns (si1..si7)."""
    return pd.DataFrame([si])


def test_wsp_worked_example_arithmetic_mean():
    # WSP example: (3,3,3,3,3,2,2) -> arithmetic mean 2.714 -> Excellent.
    df = _row(si1=3, si2=3, si3=3, si4=3, si5=3, si6=2, si7=2)
    out = resolve_structural(df)
    score = out.loc[0, "hsi_wsp_score"]
    assert score == pytest.approx(19 / 7, abs=1e-3)          # 2.714, NOT geometric (2.69)
    assert out.loc[0, "hsi_wsp_category"] == "Excellent"
    assert bool(out.loc[0, "hsi_complete"]) is True


def test_category_thresholds():
    poor = resolve_structural(_row(si1=1, si2=1, si3=1, si4=2, si5=2, si6=1, si7=1))
    assert poor.loc[0, "hsi_wsp_category"] == "Poor"        # mean ~1.28
    good = resolve_structural(_row(si1=2, si2=2, si3=2, si4=2, si5=2, si6=2, si7=1))
    assert good.loc[0, "hsi_wsp_category"] == "Good"         # mean ~1.857


def test_si_band_boundaries():
    assert _si_height(2.0) == 3 and _si_height(1.99) == 2 and _si_height(1.0) == 2 and _si_height(0.99) == 1
    assert _si_width(1.5) == 3 and _si_width(1.49) == 2 and _si_width(0.99) == 1
    # gappiness accepts fraction (0-1) or percent
    assert _si_gap(0.05) == 3 and _si_gap(0.10) == 2 and _si_gap(0.20) == 2 and _si_gap(0.21) == 1
    assert _si_arable_margin(6.0) == 4 and _si_arable_margin(3.0) == 3 and _si_arable_margin(1.0) == 2 and _si_arable_margin(0.0) == 1
    assert _si_trees(7) == 4 and _si_trees(4) == 3 and _si_trees(1) == 2 and _si_trees(0) == 1
    assert _si_species(8) == 3 and _si_species(5) == 2 and _si_species(2) == 1


def test_normalisation_scales_to_unit_interval():
    assert _normalise_si("si1", 1) == 0.0 and _normalise_si("si1", 2) == 0.5 and _normalise_si("si1", 3) == 1.0
    assert _normalise_si("si4", 1) == 0.0 and _normalise_si("si4", 4) == 1.0      # max 4
    assert _normalise_si("si7", 1) == 0.0 and _normalise_si("si7", 2) == 1.0      # max 2
    assert _normalise_si("si1", None) is None


def test_si6_precautionary_default():
    out = resolve_structural(_row(si1=3, si2=3, si3=3, si4=3, si5=3, si7=2))  # no si6
    assert out.loc[0, "hsi_si6_score"] == config.SI6_DEFAULT_BAND
    assert out.loc[0, "hsi_si6_source"] == "default"
    assert out.loc[0, "hsi_si6_confidence"] == "Low"
    assert bool(out.loc[0, "field_verification_required"]) is True


def test_si6_field_override():
    out = resolve_structural(_row(si1=3, si2=3, si3=3, si4=3, si5=3, si6=None,
                                  woody_species_count_20m=8, si7=2))
    assert out.loc[0, "hsi_si6_score"] == 3
    assert out.loc[0, "hsi_si6_source"] == "field"
    assert out.loc[0, "hsi_si6_confidence"] == "High"


def test_incomplete_when_structural_si_missing():
    # No LiDAR-derived SI1/SI2/SI3 -> Incomplete confidence, but a provisional score still computed.
    out = resolve_structural(_row(si4=3, si5=3, si7=2))
    assert bool(out.loc[0, "hsi_complete"]) is False
    assert out.loc[0, "hsi_confidence_level"] == "Incomplete"
    assert out.loc[0, "hsi_survey_requirement"] == config.SURVEY_REQUIREMENTS["Incomplete"]
    assert out.loc[0, "hsi_wsp_score"] is not None  # provisional from present SIs


def test_weighting_and_blend():
    resolved = resolve_structural(_row(si1=3, si2=3, si3=3, si4=3, si5=3, si6=2, si7=2))
    resolved["ctx_woodland"] = 0.2
    resolved["ctx_water"] = 0.2
    resolved["ctx_connectivity"] = 0.2
    resolved["ctx_roost"] = 0.2
    resolved["ctx_darkness"] = 0.2
    resolved["ctx_road_severance"] = 0.2

    equal = apply_scoring(resolved, config.ScoreSettings(alpha=1.0))
    a = equal.loc[0, "hsi_structural_A"]
    assert equal.loc[0, "hsi_priority"] == pytest.approx(a)            # alpha=1 -> priority == A

    pure_ctx = apply_scoring(resolved, config.ScoreSettings(alpha=0.0))
    assert pure_ctx.loc[0, "hsi_priority"] == pytest.approx(0.2)        # alpha=0 -> priority == B (all 0.2)

    # Zeroing a weight changes A but not the WSP category.
    weights = dict(config.DEFAULT_SI_WEIGHTS)
    weights["si6"] = 0.0
    reweighted = apply_scoring(resolved, config.ScoreSettings(si_weights=weights, alpha=1.0))
    assert reweighted.loc[0, "hsi_structural_A"] != pytest.approx(a)
    assert reweighted.loc[0, "hsi_wsp_category"] == equal.loc[0, "hsi_wsp_category"]


def test_ranking_orders_by_priority():
    df = pd.DataFrame([
        {"si1": 3, "si2": 3, "si3": 3, "si4": 4, "si5": 4, "si6": 3, "si7": 2},  # best
        {"si1": 1, "si2": 1, "si3": 1, "si4": 1, "si5": 1, "si6": 1, "si7": 1},  # worst
    ])
    scored = apply_scoring(resolve_structural(df))
    assert scored.loc[0, "hsi_priority_rank"] == 1
    assert scored.loc[1, "hsi_priority_rank"] == 2
