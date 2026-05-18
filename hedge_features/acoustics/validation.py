from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AcousticValidationSettings:
    """Settings for comparing GIS screening scores with acoustic evidence."""

    score_column: str = "survey_priority_score"
    acoustic_presence_column: str = "acoustic_detection_count"
    acoustic_presence_threshold: float = 1.0
    low_score_threshold: float = 0.33
    high_score_threshold: float = 0.67
    id_column: str | None = "hf_uid"
    species_list_column: str = "acoustic_species_list"
    guild_list_column: str = "acoustic_guild_list"
    max_examples: int = 20


def validate_acoustic_evidence(df, *, settings: AcousticValidationSettings) -> tuple[Any, dict[str, Any]]:
    """Compare GIS priority scores with segment-level acoustic evidence.

    Returns a copy of the input dataframe with validation helper columns plus a JSON-serialisable
    summary that highlights high-scoring segments without acoustic evidence and low-scoring
    segments with acoustic evidence.
    """
    import pandas as pd

    if settings.score_column not in df.columns:
        raise ValueError(f"Score column '{settings.score_column}' was not found.")
    if settings.acoustic_presence_column not in df.columns:
        raise ValueError(f"Acoustic presence column '{settings.acoustic_presence_column}' was not found.")

    out = df.copy()
    score = pd.to_numeric(out[settings.score_column], errors="coerce")
    presence_count = pd.to_numeric(out[settings.acoustic_presence_column], errors="coerce").fillna(0.0)
    presence_flag = presence_count >= float(settings.acoustic_presence_threshold)
    bands = score.map(
        lambda value: _score_band(
            value,
            low_threshold=float(settings.low_score_threshold),
            high_threshold=float(settings.high_score_threshold),
        )
    )
    cases = [
        _validation_case(score_band=band, acoustic_present=bool(present))
        for band, present in zip(bands.tolist(), presence_flag.tolist())
    ]

    out["acoustic_presence_flag"] = presence_flag.astype(int)
    out["acoustic_validation_score_band"] = bands
    out["acoustic_validation_case"] = cases

    summary = {
        "settings": asdict(settings),
        "row_count": int(len(out)),
        "scored_row_count": int(score.notna().sum()),
        "acoustic_presence_count": int(presence_flag.sum()),
        "acoustic_absence_count": int((~presence_flag).sum()),
        "score_band_counts": _counts(bands),
        "validation_case_counts": _counts(cases),
        "acoustic_presence_by_score_band": _presence_by_band(bands, presence_flag),
        "species_by_score_band": _token_counts_by_band(out, bands, settings.species_list_column),
        "guild_by_score_band": _token_counts_by_band(out, bands, settings.guild_list_column),
        "examples": _examples(out, settings=settings),
    }
    return out, summary


def _score_band(value, *, low_threshold: float, high_threshold: float) -> str:
    import pandas as pd

    if pd.isna(value):
        return "missing"
    numeric = float(value)
    if numeric >= high_threshold:
        return "high"
    if numeric <= low_threshold:
        return "low"
    return "medium"


def _validation_case(*, score_band: str, acoustic_present: bool) -> str:
    if score_band == "missing":
        return "missing_score"
    if score_band == "high" and not acoustic_present:
        return "high_score_no_acoustic_evidence"
    if score_band == "low" and acoustic_present:
        return "low_score_with_acoustic_evidence"
    if score_band == "high" and acoustic_present:
        return "aligned_high_presence"
    if score_band == "low" and not acoustic_present:
        return "aligned_low_absence"
    return "medium_or_mixed"


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _presence_by_band(bands, presence_flag) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for band, present in zip(bands.tolist(), presence_flag.tolist()):
        band_key = str(band)
        out.setdefault(band_key, {"present": 0, "absent": 0})
        out[band_key]["present" if bool(present) else "absent"] += 1
    return dict(sorted(out.items()))


def _token_counts_by_band(df, bands, column: str) -> dict[str, dict[str, int]]:
    if column not in df.columns:
        return {}
    out: dict[str, dict[str, int]] = {}
    for band, raw_value in zip(bands.tolist(), df[column].tolist()):
        tokens = _split_tokens(raw_value)
        if not tokens:
            continue
        band_key = str(band)
        out.setdefault(band_key, {})
        for token in tokens:
            out[band_key][token] = out[band_key].get(token, 0) + 1
    return {band: dict(sorted(values.items())) for band, values in sorted(out.items())}


def _split_tokens(value) -> list[str]:
    import pandas as pd

    if pd.isna(value):
        return []
    tokens: list[str] = []
    for token in str(value).replace(",", "|").split("|"):
        token = token.strip()
        if token and token.lower() not in {"nan", "none", "unknown"}:
            tokens.append(token)
    return sorted(set(tokens))


def _examples(df, *, settings: AcousticValidationSettings) -> dict[str, list[dict[str, Any]]]:
    case_names = (
        "high_score_no_acoustic_evidence",
        "low_score_with_acoustic_evidence",
    )
    out: dict[str, list[dict[str, Any]]] = {}
    for case_name in case_names:
        subset = df.loc[df["acoustic_validation_case"] == case_name].head(max(int(settings.max_examples), 0))
        rows: list[dict[str, Any]] = []
        for _, row in subset.iterrows():
            item: dict[str, Any] = {
                "score": _json_scalar(row.get(settings.score_column)),
                "acoustic_count": _json_scalar(row.get(settings.acoustic_presence_column)),
            }
            if settings.id_column and settings.id_column in df.columns:
                item[str(settings.id_column)] = _json_scalar(row.get(settings.id_column))
            rows.append(item)
        out[case_name] = rows
    return out


def _json_scalar(value):
    import pandas as pd

    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
