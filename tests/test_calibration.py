"""Tests for HSI calibration against survey activity (scipy-only)."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")

from hsi import config
from hsi.calibration import _auc, calibrate, join_activity, optimize_weights
from hsi.score import apply_scoring, resolve_structural


def _scored(n=12):
    rows = []
    for i in range(n):
        v = 1 + (i % 3)  # 1,2,3 spread
        rows.append({"hf_uid": f"h{i}", "si1": v, "si2": v, "si3": v, "si4": v, "si5": v, "si6": v, "si7": min(2, v)})
    resolved = resolve_structural(pd.DataFrame(rows))
    for k in config.CONTEXT_KEYS:
        resolved[k] = 0.5
    return apply_scoring(resolved)


def test_join_and_calibrate_positive_correlation():
    scored = _scored()
    activity = pd.DataFrame({"hf_uid": scored["hf_uid"], "calls": (scored["hsi_priority"] * 100).round()})
    joined = join_activity(scored, activity, activity_col="calls", join_col="hf_uid")
    report = calibrate(joined, activity_col="activity")
    assert report["n"] >= 3
    assert report["spearman_priority_vs_activity"] is not None
    assert report["spearman_priority_vs_activity"] > 0.8  # activity built to track priority


def test_auc_perfect_separation():
    assert _auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == pytest.approx(1.0)
    assert _auc([0.1, 0.2, 0.3], [False, False, False]) is None  # no positives


def test_optimize_returns_valid_weights():
    scored = _scored()
    activity = (scored["hsi_structural_A"] * 100).to_numpy()
    weights = optimize_weights(scored, activity, config.ScoreSettings())
    assert set(weights.keys()) == set(config.SI_KEYS)
    assert all(v >= 0 for v in weights.values())
