from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")

from hedge_features.features.raster import _categorical_landscape_metrics  # noqa: E402


def test_categorical_landscape_metrics_describe_patch_edge_and_core_area():
    arr = np.ma.array(
        [
            [1, 1, 1, 2],
            [1, 1, 1, 2],
            [1, 1, 1, 2],
            [2, 2, 2, 2],
        ],
        mask=False,
    )

    metrics = _categorical_landscape_metrics(
        arr,
        nodata=None,
        class_codes=[1],
        pixel_width_m=10.0,
        pixel_height_m=10.0,
    )

    assert metrics["largest_patch_index"] == 9 / 16
    assert metrics["core_area_pct"] == 1 / 16
    assert math.isclose(metrics["edge_density_m_per_ha"], 750.0, rel_tol=1e-6)


def test_categorical_landscape_metrics_return_zero_for_absent_class():
    arr = np.ma.array([[2, 2], [2, 2]], mask=False)

    metrics = _categorical_landscape_metrics(
        arr,
        nodata=None,
        class_codes=[1],
        pixel_width_m=10.0,
        pixel_height_m=10.0,
    )

    assert metrics == {
        "edge_density_m_per_ha": 0.0,
        "largest_patch_index": 0.0,
        "core_area_pct": 0.0,
    }
