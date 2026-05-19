from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


CALIBRATION_VERSION = "bhsa_weight_calibration_v2_scaffold"
BHSA_SCORE_COLUMNS = tuple(f"bhsa_si{i}_score" for i in range(1, 8))


@dataclass(frozen=True, slots=True)
class BHSACalibrationSettings:
    label_column: str | None = "high_activity_label"
    activity_column: str | None = None
    activity_high_quantile: float = 0.75
    min_sample_size: int = 30
    folds: int = 5


def calibrate_bhsa_weights(df, *, settings: BHSACalibrationSettings | None = None) -> dict[str, Any]:
    """Transparent calibration scaffold for paired BHSA and acoustic activity records."""
    import numpy as np
    import pandas as pd

    settings = settings or BHSACalibrationSettings()
    missing_predictors = [col for col in BHSA_SCORE_COLUMNS if col not in df.columns]
    if missing_predictors:
        return _not_usable(settings, len(df), [f"Missing BHSA score columns: {', '.join(missing_predictors)}"])

    y = _label_series(df, settings=settings)
    if y is None:
        return _not_usable(settings, len(df), ["No validation label or activity column was supplied."])

    X = df[list(BHSA_SCORE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    usable = X.notna().all(axis=1) & y.notna()
    X = X.loc[usable]
    y = y.loc[usable].astype(int)
    sample_size = int(len(X))
    if sample_size < int(settings.min_sample_size) or y.nunique() < 2:
        return _not_usable(
            settings,
            sample_size,
            ["Insufficient paired records or only one activity class; do not use calibrated weights."],
        )

    equal = {col: round(1.0 / len(BHSA_SCORE_COLUMNS), 6) for col in BHSA_SCORE_COLUMNS}
    correlations = {}
    for col in BHSA_SCORE_COLUMNS:
        corr = float(np.corrcoef(X[col].astype(float), y.astype(float))[0, 1])
        correlations[col] = 0.0 if corr != corr else abs(corr)
    total_corr = sum(correlations.values())
    empirical = {
        col: (correlations[col] / total_corr if total_corr > 0 else equal[col])
        for col in BHSA_SCORE_COLUMNS
    }
    fitted = {col: round((0.50 * equal[col]) + (0.50 * empirical[col]), 6) for col in BHSA_SCORE_COLUMNS}
    weighted_score = X.mul([fitted[col] for col in BHSA_SCORE_COLUMNS], axis=1).sum(axis=1)
    auc = _roc_auc(y.tolist(), weighted_score.tolist())
    return {
        "method_version": CALIBRATION_VERSION,
        "settings": asdict(settings),
        "status": "ready_for_technical_review",
        "do_not_use_calibrated_model": False,
        "sample_size": sample_size,
        "equal_prior_weights": equal,
        "fitted_weights": fitted,
        "feature_importance": {col: round(value, 6) for col, value in sorted(correlations.items())},
        "cross_validation": {
            "folds": int(settings.folds),
            "auc_mean": auc,
            "note": "Deterministic scaffold AUC from blended weighted score; review before regulatory use.",
        },
    }


def _label_series(df, *, settings: BHSACalibrationSettings):
    import pandas as pd

    if settings.label_column and settings.label_column in df.columns:
        return pd.to_numeric(df[settings.label_column], errors="coerce")
    if settings.activity_column and settings.activity_column in df.columns:
        activity = pd.to_numeric(df[settings.activity_column], errors="coerce")
        threshold = activity.quantile(float(settings.activity_high_quantile))
        return (activity >= threshold).astype(int)
    return None


def _not_usable(settings: BHSACalibrationSettings, sample_size: int, warnings: list[str]) -> dict[str, Any]:
    return {
        "method_version": CALIBRATION_VERSION,
        "settings": asdict(settings),
        "status": "insufficient_data",
        "do_not_use_calibrated_model": True,
        "sample_size": int(sample_size),
        "warnings": warnings,
        "equal_prior_weights": {col: round(1.0 / len(BHSA_SCORE_COLUMNS), 6) for col in BHSA_SCORE_COLUMNS},
        "fitted_weights": None,
        "feature_importance": {},
        "cross_validation": None,
    }


def _roc_auc(y_true: list[int], scores: list[float]) -> float | None:
    positives = [(score, label) for score, label in zip(scores, y_true) if int(label) == 1]
    negatives = [(score, label) for score, label in zip(scores, y_true) if int(label) == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = float(len(positives) * len(negatives))
    for p_score, _ in positives:
        for n_score, _ in negatives:
            if p_score > n_score:
                wins += 1.0
            elif p_score == n_score:
                wins += 0.5
    return round(wins / total, 6)
