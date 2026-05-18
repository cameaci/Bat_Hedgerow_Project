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


def test_categorical_landscape_metrics_respect_mask_and_nodata():
    arr = np.ma.array(
        [
            [1, 1, 99],
            [1, 2, 99],
            [2, 2, 99],
        ],
        mask=[
            [False, False, False],
            [False, False, True],
            [False, False, False],
        ],
    )

    metrics = _categorical_landscape_metrics(
        arr,
        nodata=99,
        class_codes=[1],
        pixel_width_m=10.0,
        pixel_height_m=10.0,
    )

    assert metrics["largest_patch_index"] == 3 / 6
    assert metrics["core_area_pct"] == 0.0
    assert metrics["edge_density_m_per_ha"] > 0.0


def test_unsupported_landscape_metric_raises_clear_error():
    from hedge_features.features.raster import add_raster_categorical_proportions_in_buffers

    with pytest.raises(ValueError, match="Unsupported landscape metric"):
        add_raster_categorical_proportions_in_buffers(
            __import__("pandas").DataFrame({"id": [1]}),
            None,
            radii_m=[100],
            class_map={"tree": [1]},
            column_template="buf{radius}_{class_name}_pct",
            landscape_class_names=["tree"],
            landscape_metrics=["not_a_metric"],
            landscape_column_templates={"not_a_metric": "buf{radius}_{class_name}_bad"},
        )
