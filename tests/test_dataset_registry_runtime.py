from pathlib import Path

from hedge_features.datasets.registry import DatasetRegistry


def test_runtime_resolution_overrides_profile_path(tmp_path: Path):
    local1 = tmp_path / "a.gpkg"
    local2 = tmp_path / "b.gpkg"
    local1.write_text("x", encoding="utf-8")
    local2.write_text("y", encoding="utf-8")

    reg = DatasetRegistry(profile_datasets={"foo": {"path": str(local1), "license": "L1"}})
    reg.set_runtime_resolution("foo", path=str(local2), mode="on_demand", license="L2", metadata={"provider": "x"})
    ref = reg.resolve("foo")
    assert ref.path == str(local2)
    assert ref.mode == "on_demand"
    assert ref.license == "L2"
    assert ref.metadata["provider"] == "x"
    assert ref.metadata["exists"] is True

