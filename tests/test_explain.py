"""Tests for explainability (contribution breakdown + sensitivity)."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from hsi import config
from hsi.explain import explain_hedgerow, sensitivity
from hsi.score import apply_scoring, resolve_structural


def _scored():
    df = pd.DataFrame([
        {"hf_uid": "a", "si1": 3, "si2": 3, "si3": 2, "si4": 3, "si5": 4, "si6": 2, "si7": 2},
        {"hf_uid": "b", "si1": 1, "si2": 1, "si3": 1, "si4": 1, "si5": 1, "si6": 1, "si7": 1},
    ])
    resolved = resolve_structural(df)
    for k in config.CONTEXT_KEYS:
        resolved[k] = 0.5
    return apply_scoring(resolved)


def test_contributions_sum_to_priority():
    scored = _scored()
    expl = explain_hedgerow(scored, "a")
    total = sum(c["contribution"] for c in expl["contributions"])
    assert total == pytest.approx(expl["priority"], abs=2e-3)
    # every SI and context factor that is present appears
    factors = {c["factor"] for c in expl["contributions"]}
    assert config.SI_LABELS["si1"] in factors
    assert config.CONTEXT_LABELS["ctx_woodland"] in factors


def test_sensitivity_nonneg_and_sorted():
    scored = _scored()
    res = sensitivity(scored, delta=0.5)
    assert res, "expected sensitivity rows"
    assert all(r["impact"] >= 0 for r in res)
    assert res == sorted(res, key=lambda d: d["impact"], reverse=True)
