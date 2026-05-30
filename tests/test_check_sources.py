"""Tests for the data-source probe (offline only — no network calls)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_sources.py"
_spec = importlib.util.spec_from_file_location("check_sources", _SCRIPT)
check_sources = importlib.util.module_from_spec(_spec)
sys.modules["check_sources"] = check_sources
_spec.loader.exec_module(check_sources)


def test_classify_statuses():
    assert check_sources._classify("x", 5, True, {}) == "OK"
    assert check_sources._classify("x", 0, True, {}) == "EMPTY"
    assert check_sources._classify("x", None, False, {"local_only": True}) == "MISSING"
    assert check_sources._classify("x", None, False, {}) == "NO_DATA"
    assert check_sources._classify("x", None, True, {}) == "OK"  # rasters report no count


def test_offline_probe_degrades_without_network():
    pytest.importorskip("geopandas")
    aoi = check_sources._aoi_gdf_default()
    rows = check_sources.probe(aoi, only={"ea_lidar_dtm", "ne_phi"}, allow_live=False)
    by_name = {r["source"]: r for r in rows}
    # ea_lidar_dtm now has a WCS auto-provider, so with live disabled it degrades to NO_DATA.
    assert by_name["ea_lidar_dtm"]["status"] in {"NO_DATA", "MISSING"}
    assert by_name["ne_phi"]["status"] == "NO_DATA"          # live disabled, no local copy
    # No source should report OK with live disabled and no local data.
    assert all(r["status"] != "OK" for r in rows)
