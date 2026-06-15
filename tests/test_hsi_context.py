"""Tests for landscape-context normalisation helpers (no GIS required)."""

from __future__ import annotations

import pytest

from hsi.context import _fardist, _invdist, _mean_present, _scaled


def test_invdist_closer_is_better():
    assert _invdist(0.0, 150.0) == pytest.approx(1.0)
    assert _invdist(150.0, 150.0) == pytest.approx(0.5)
    assert _invdist(None, 150.0) is None


def test_fardist_is_complement():
    assert _fardist(0.0, 100.0) == pytest.approx(0.0)
    assert _fardist(100.0, 100.0) == pytest.approx(0.5)
    assert _fardist(None, 100.0) is None


def test_scaled_caps_at_one():
    assert _scaled(0.15, 0.30) == pytest.approx(0.5)
    assert _scaled(0.5, 0.30) == pytest.approx(1.0)
    assert _scaled(-1.0, 0.30) == pytest.approx(0.0)
    assert _scaled(None, 0.30) is None


def test_mean_present_skips_missing():
    assert _mean_present([1.0, None, 0.0]) == pytest.approx(0.5)
    assert _mean_present([None, None]) is None
    assert _mean_present([]) is None
