from __future__ import annotations

import pytest
from click import BadParameter

from hedge_features.cli import _planning_weight_kwargs


def test_planning_weight_kwargs_maps_supported_aliases():
    out = _planning_weight_kwargs(("base_score=0.5", "corridor_coverage=0.2", "redundancy_penalty=0.1"))

    assert out == {
        "objective_weight_base_score": 0.5,
        "objective_weight_corridor_coverage": 0.2,
        "objective_weight_redundancy_penalty": 0.1,
    }


def test_planning_weight_kwargs_rejects_unknown_alias():
    with pytest.raises(BadParameter, match="Invalid --objective-weight key"):
        _planning_weight_kwargs(("unknown=1.0",))
