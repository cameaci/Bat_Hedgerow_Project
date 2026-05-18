from __future__ import annotations

import json

import pandas as pd
from click.testing import CliRunner

from hedge_features.acoustics import AcousticValidationSettings, validate_acoustic_evidence
from hedge_features.cli import main


def _validation_df():
    return pd.DataFrame(
        {
            "hf_uid": ["h1", "h2", "h3", "h4", "h5"],
            "survey_priority_score": [0.9, 0.8, 0.2, 0.1, 0.5],
            "acoustic_detection_count": [0, 3, 2, 0, 1],
            "acoustic_species_list": ["", "Pipistrellus", "Myotis|Pipistrellus", "", "Noctule"],
            "acoustic_guild_list": ["", "edge", "water|edge", "", "open"],
        }
    )


def test_validate_acoustic_evidence_flags_score_evidence_mismatches():
    annotated, summary = validate_acoustic_evidence(
        _validation_df(),
        settings=AcousticValidationSettings(
            score_column="survey_priority_score",
            low_score_threshold=0.33,
            high_score_threshold=0.67,
            id_column="hf_uid",
        ),
    )

    cases = annotated.set_index("hf_uid")["acoustic_validation_case"].to_dict()
    assert cases["h1"] == "high_score_no_acoustic_evidence"
    assert cases["h2"] == "aligned_high_presence"
    assert cases["h3"] == "low_score_with_acoustic_evidence"
    assert cases["h4"] == "aligned_low_absence"
    assert cases["h5"] == "medium_or_mixed"
    assert summary["validation_case_counts"]["high_score_no_acoustic_evidence"] == 1
    assert summary["validation_case_counts"]["low_score_with_acoustic_evidence"] == 1
    assert summary["acoustic_presence_by_score_band"]["high"] == {"present": 1, "absent": 1}
    assert summary["species_by_score_band"]["low"] == {"Myotis": 1, "Pipistrellus": 1}
    assert summary["guild_by_score_band"]["medium"] == {"open": 1}
    assert summary["examples"]["high_score_no_acoustic_evidence"][0]["hf_uid"] == "h1"


def test_validate_acoustic_evidence_requires_score_and_presence_columns():
    try:
        validate_acoustic_evidence(
            pd.DataFrame({"survey_priority_score": [0.5]}),
            settings=AcousticValidationSettings(),
        )
    except ValueError as exc:
        assert "Acoustic presence column" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected missing acoustic presence column error")


def test_validate_acoustics_cli_writes_summary_and_annotated_csv(tmp_path):
    input_path = tmp_path / "screened.csv"
    json_path = tmp_path / "validation.json"
    csv_path = tmp_path / "validation_rows.csv"
    _validation_df().to_csv(input_path, index=False)

    result = CliRunner().invoke(
        main,
        [
            "validate-acoustics",
            "--input",
            str(input_path),
            "--output-json",
            str(json_path),
            "--output-csv",
            str(csv_path),
            "--json-summary",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["validation_case_counts"]["high_score_no_acoustic_evidence"] == 1
    assert payload["written"] == {"json": str(json_path), "csv": str(csv_path)}
    written_summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert written_summary["row_count"] == 5
    written_rows = pd.read_csv(csv_path)
    assert "acoustic_validation_case" in written_rows.columns
