from pathlib import Path

import pandas as pd
import pytest

from hedge_features.models import RunMetadata
from hedge_features.screening import load_framework_bundle, screen_dataframe


def _sample_screening_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hf_uid": ["h1", "h2"],
            "geom_length_m": [120.0, 80.0],
            "geom_sinuosity": [1.2, 1.1],
            "net_degree_max": [3, 2],
            "dist_os_road_m": [40.0, 10.0],
            "dist_os_river_m": [30.0, 20.0],
            "buf100_os_road_density_m_per_ha": [25.0, 15.0],
            "buf100_worldcover_tree_pct": [0.55, 0.70],
            "buf100_worldcover_built_pct": [0.10, 0.05],
            "buf100_worldcover_water_pct": [0.05, 0.10],
            "buf100_worldcover_wetland_pct": [0.15, 0.20],
            "buf100_nightlight_mean": [10.0, 5.0],
            "roostpx_struct_proxy_score": [0.70, 0.85],
            "mhb_roost_proxy_score": [0.65, 0.80],
            "mhb_dem_shelter_idx_100m": [0.60, 0.75],
            "deploy_status": ["ok", "ok"],
            "roostpx_status": ["ok", "ok"],
            "depwx_status": ["ok", "ok"],
            "moon_status": ["ok", "ok"],
            "phi_coverage_flag": [1, 1],
            "awi_coverage_flag": [1, 1],
            "_merge": ["both", "both"],
        }
    )


def test_screening_outputs_are_deterministic_by_default():
    framework = load_framework_bundle("bats_screening_v1")
    df = _sample_screening_df()

    run1 = screen_dataframe(df, framework=framework)
    run2 = screen_dataframe(df, framework=framework)

    assert run1.results_df["analysis_run_id"].tolist() == run2.results_df["analysis_run_id"].tolist()
    assert run1.results_df["analysis_timestamp_utc"].isna().all()
    assert run2.results_df["analysis_timestamp_utc"].isna().all()
    assert run1.run_summary["analysis_timestamp_utc"] is None
    assert run1.run_summary["analysis_timestamp_utc"] == run2.run_summary["analysis_timestamp_utc"]


def test_run_metadata_build_is_stable_in_deterministic_mode(tmp_path: Path):
    input_path = tmp_path / "in.gpkg"
    output_path = tmp_path / "out.gpkg"
    input_path.write_text("x", encoding="utf-8")
    datasets = [{"name": "worldcover", "path": "C:/data/worldcover.tif", "version": "snapshot-1"}]

    m1 = RunMetadata.build(
        tool_version="0.1.0",
        input_path=input_path,
        output_path=output_path,
        working_crs="EPSG:27700",
        export_crs=None,
        profile_name="bats_v1",
        profile_path=None,
        datasets=datasets,
        deterministic_output=True,
        frozen_datasets_only=True,
    )
    m2 = RunMetadata.build(
        tool_version="0.1.0",
        input_path=input_path,
        output_path=output_path,
        working_crs="EPSG:27700",
        export_crs=None,
        profile_name="bats_v1",
        profile_path=None,
        datasets=datasets,
        deterministic_output=True,
        frozen_datasets_only=True,
    )

    assert m1.run_id == m2.run_id
    assert m1.run_timestamp_utc is None
    assert m2.run_timestamp_utc is None


def test_dataset_snapshot_manifest_records_checksum_and_metadata(tmp_path: Path):
    import hashlib
    import json

    from hedge_features.datasets.auto_fetch import AutoDataFetcher

    fetcher = AutoDataFetcher.__new__(AutoDataFetcher)
    fetcher.profile_datasets = {
        "roads": {
            "license": "ODbL 1.0",
            "attribution": "Contains OpenStreetMap data",
        }
    }
    data_path = tmp_path / "roads.gpkg"
    data_path.write_bytes(b"snapshot contents")
    meta = {
        "provider": "osm_overpass_lines",
        "provider_url": "https://example.test/overpass",
        "bbox_wgs84": [-1.0, 51.0, -0.5, 51.5],
        "version": "live-overpass",
        "cache_hit": False,
    }

    returned = fetcher._write_snapshot_manifest(
        "roads",
        "osm_overpass_lines",
        data_path,
        meta,
        bbox_wgs84=(-1.0, 51.0, -0.5, 51.5),
    )

    manifest_path = tmp_path / "SNAPSHOT.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_sha = hashlib.sha256(b"snapshot contents").hexdigest()
    assert manifest["manifest_version"] == "dataset_snapshot_v1"
    assert manifest["dataset_name"] == "roads"
    assert manifest["provider_type"] == "osm_overpass_lines"
    assert manifest["provider_metadata"] == meta
    assert manifest["sha256"] == expected_sha
    assert returned["snapshot_manifest_path"] == str(manifest_path)
    assert returned["snapshot_checksum_sha256"] == expected_sha

    cached = fetcher._cached_dataset_metadata(data_path, {"provider": "fallback"})
    assert cached["provider"] == "osm_overpass_lines"
    assert cached["cache_hit"] is True
    assert cached["snapshot_manifest_path"] == str(manifest_path)
