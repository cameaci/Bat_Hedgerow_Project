from __future__ import annotations

import json

import pandas as pd
import pytest
from click.testing import CliRunner

from hedge_features.acoustics import AcousticImportSettings, import_acoustic_evidence


class HedgeFrame(pd.DataFrame):
    _metadata = ["crs"]

    @property
    def _constructor(self):
        return HedgeFrame


def _hedge_table():
    hedges = HedgeFrame({"hf_uid": ["h1", "h2"]})
    hedges.crs = "EPSG:27700"
    return hedges


def test_import_acoustic_evidence_links_by_hedge_id_and_aggregates():
    detections = pd.DataFrame(
        {
            "hedge_id": ["h1", "h1", "h2", "missing"],
            "species": ["Pipistrellus pipistrellus", "Myotis daubentonii", "Pipistrellus pipistrellus", "Noctule"],
            "guild": ["edge", "water", "edge", "open"],
            "confidence": [0.95, 0.65, 0.4, 0.99],
            "timestamp": ["2026-05-01T22:00:00Z", "2026-05-01T22:10:00Z", "2026-05-02T22:00:00Z", "2026-05-03T22:00:00Z"],
            "calls": [3, 2, 5, 9],
        }
    )
    out, summary = import_acoustic_evidence(
        _hedge_table(),
        detections,
        settings=AcousticImportSettings(
            detection_hedge_id_column="hedge_id",
            min_confidence=0.5,
            activity_column="calls",
        ),
    )

    row_h1 = out.set_index("hf_uid").loc["h1"]
    row_h2 = out.set_index("hf_uid").loc["h2"]
    assert row_h1["acoustic_detection_count"] == 2
    assert row_h1["acoustic_species_count"] == 2
    assert row_h1["acoustic_species_list"] == "Myotis daubentonii|Pipistrellus pipistrellus"
    assert row_h1["acoustic_guild_list"] == "edge|water"
    assert row_h1["acoustic_activity_sum"] == 5
    assert row_h1["acoustic_max_confidence"] == 0.95
    assert row_h2["acoustic_detection_count"] == 0
    assert summary["input_detection_records"] == 4
    assert summary["records_after_confidence_filter"] == 3
    assert summary["matched_detection_records"] == 2
    assert summary["hedgerows_with_acoustic_evidence"] == 1


def test_import_acoustic_evidence_spatially_matches_nearest_hedgerow():
    gpd = pytest.importorskip("geopandas")
    pytest.importorskip("shapely")
    from shapely.geometry import LineString

    hedges = gpd.GeoDataFrame(
        {"hf_uid": ["h1", "h2"]},
        geometry=[LineString([(0, 0), (100, 0)]), LineString([(0, 100), (100, 100)])],
        crs="EPSG:27700",
    )
    detections = pd.DataFrame(
        {
            "y": [5.0, 100.0],
            "x": [20.0, 20.0],
            "class": ["Pipistrellus", "Noctule"],
            "prob": [0.9, 0.8],
        }
    )
    out, summary = import_acoustic_evidence(
        hedges,
        detections,
        settings=AcousticImportSettings(
            source_format="batdetect2",
            detections_crs="EPSG:27700",
            max_distance_m=30.0,
        ),
    )

    rows = out.set_index("hf_uid")
    assert rows.loc["h1", "acoustic_detection_count"] == 1
    assert rows.loc["h1", "acoustic_species_list"] == "Pipistrellus"
    assert rows.loc["h2", "acoustic_detection_count"] == 1
    assert summary["matched_detection_records"] == 2


def test_import_acoustics_cli_writes_geodata_and_summary(tmp_path):
    gpd = pytest.importorskip("geopandas")
    pytest.importorskip("shapely")
    from shapely.geometry import LineString
    from hedge_features.cli import main

    hedges_path = tmp_path / "hedges.geojson"
    detections_path = tmp_path / "detections.csv"
    output_path = tmp_path / "acoustic.geojson"
    gpd.GeoDataFrame(
        {"hf_uid": ["h1", "h2"]},
        geometry=[LineString([(0, 0), (100, 0)]), LineString([(0, 100), (100, 100)])],
        crs="EPSG:27700",
    ).to_file(hedges_path, driver="GeoJSON")
    pd.DataFrame(
        {
            "hedge_id": ["h1", "h2"],
            "species": ["Pipistrellus", "Myotis"],
            "confidence": [0.9, 0.8],
        }
    ).to_csv(detections_path, index=False)

    result = CliRunner().invoke(
        main,
        [
            "import-acoustics",
            "--hedges",
            str(hedges_path),
            "--detections",
            str(detections_path),
            "--output",
            str(output_path),
            "--detection-hedge-id-col",
            "hedge_id",
            "--json-summary",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["matched_detection_records"] == 2
    written = gpd.read_file(output_path)
    assert sorted(written["acoustic_species_list"].tolist()) == ["Myotis", "Pipistrellus"]
    assert (tmp_path / "METADATA.json").exists()


def test_import_acoustic_evidence_reports_column_audit_and_drop_reasons():
    detections = pd.DataFrame(
        {
            "hedge_id": ["h1", "missing", "h2"],
            "species": ["Pipistrellus", "Noctule", "Myotis"],
            "confidence": [0.9, 0.8, 0.2],
        }
    )

    _, summary = import_acoustic_evidence(
        _hedge_table(),
        detections,
        settings=AcousticImportSettings(
            detection_hedge_id_column="hedge_id",
            min_confidence=0.5,
        ),
    )

    assert summary["adapter_version"] == "acoustic_import_v2"
    assert summary["column_audit"]["canonical_to_source"]["species"] == "species"
    assert summary["column_audit"]["canonical_to_source"]["confidence"] == "confidence"
    assert "datetime" in summary["column_audit"]["missing_optional_columns"]
    assert summary["drop_reason_counts"] == {
        "below_min_confidence": 1,
        "unmatched_hedgerow_id": 1,
    }


def test_import_acoustic_evidence_adds_nightly_activity_metrics():
    detections = pd.DataFrame(
        {
            "hedge_id": ["h1", "h1", "h1", "h1"],
            "timestamp": [
                "2026-05-01T22:00:00Z",
                "2026-05-02T01:00:00Z",  # same acoustic night with noon rollover
                "2026-05-03T22:00:00Z",
                "2026-05-05T22:00:00Z",
            ],
        }
    )

    out, _ = import_acoustic_evidence(
        _hedge_table(),
        detections,
        settings=AcousticImportSettings(
            detection_hedge_id_column="hedge_id",
            datetime_column="timestamp",
            acoustic_timezone="UTC",
            night_rollover_hour=12,
        ),
    )

    row = out.set_index("hf_uid").loc["h1"]
    assert row["acoustic_night_count"] == 3
    assert row["acoustic_detections_per_night_max"] == 2
    assert row["acoustic_detections_per_night_mean"] == round(4 / 3, 6)
    assert row["acoustic_active_nights_pct"] == 0.6


def test_import_acoustic_evidence_reports_invalid_spatial_rows():
    gpd = pytest.importorskip("geopandas")
    pytest.importorskip("shapely")
    from shapely.geometry import LineString

    hedges = gpd.GeoDataFrame(
        {"hf_uid": ["h1"]},
        geometry=[LineString([(0, 0), (100, 0)])],
        crs="EPSG:27700",
    )
    detections = pd.DataFrame(
        {
            "y": [5.0, None, 1000.0],
            "x": [20.0, 20.0, 20.0],
            "class": ["Pipistrellus", "Noctule", "Myotis"],
        }
    )

    _, summary = import_acoustic_evidence(
        hedges,
        detections,
        settings=AcousticImportSettings(
            source_format="batdetect2",
            detections_crs="EPSG:27700",
            max_distance_m=30.0,
        ),
    )

    assert summary["matched_detection_records"] == 1
    assert summary["drop_reason_counts"]["invalid_coordinates"] == 1
    assert summary["drop_reason_counts"]["unmatched_spatial"] == 1
