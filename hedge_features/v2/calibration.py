from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


CALIBRATION_VERSION = "bhsa_weight_calibration_v2"
BHSA_SCORE_COLUMNS = tuple(f"bhsa_si{i}_score" for i in range(1, 8))


@dataclass(frozen=True, slots=True)
class BHSACalibrationSettings:
    label_column: str | None = "high_activity_label"
    activity_column: str | None = None
    activity_high_quantile: float = 0.75
    min_sample_size: int = 30
    min_class_count: int = 5
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
    class_balance = {str(k): int(v) for k, v in y.value_counts().sort_index().items()}
    reliability_warnings = _reliability_warnings(settings=settings, sample_size=sample_size, class_balance=class_balance)
    if reliability_warnings:
        return _not_usable(
            settings,
            sample_size,
            reliability_warnings,
            class_balance=class_balance,
        )

    equal = {col: round(1.0 / len(BHSA_SCORE_COLUMNS), 6) for col in BHSA_SCORE_COLUMNS}
    fitted, correlations = _fit_weights(X, y, equal=equal)
    equal_score = _weighted_score(X, equal)
    weighted_score = _weighted_score(X, fitted)
    baseline_auc = _roc_auc(y.tolist(), equal_score.tolist())
    auc = _roc_auc(y.tolist(), weighted_score.tolist())
    cv = _cross_validation_summary(X, y, settings=settings, equal=equal)
    return {
        "method_version": CALIBRATION_VERSION,
        "settings": asdict(settings),
        "status": "ready_for_technical_review",
        "do_not_use_calibrated_model": False,
        "sample_size": sample_size,
        "class_balance": class_balance,
        "reliability_warnings": [],
        "equal_prior_weights": equal,
        "equal_prior_baseline": {"auc": baseline_auc},
        "evidence_adjusted_weights": fitted,
        "fitted_weights": fitted,
        "feature_importance": {col: round(value, 6) for col, value in sorted(correlations.items())},
        "cross_validation": {
            "folds": int(settings.folds),
            "folds_used": cv["folds_used"],
            "auc_mean": auc,
            "auc_by_fold": cv["auc_by_fold"],
            "full_sample_auc": auc,
            "baseline_auc": baseline_auc,
            "note": "Deterministic hybrid weighting summary; retain equal-prior model until reviewed by a bat specialist.",
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


def _fit_weights(X, y, *, equal: dict[str, float]):
    import numpy as np

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
    return fitted, correlations


def _weighted_score(X, weights: dict[str, float]):
    return X.mul([weights[col] for col in BHSA_SCORE_COLUMNS], axis=1).sum(axis=1)


def _cross_validation_summary(X, y, *, settings: BHSACalibrationSettings, equal: dict[str, float]) -> dict[str, Any]:
    import numpy as np

    folds = max(2, int(settings.folds))
    if len(X) < folds:
        return {"folds_used": 0, "auc_by_fold": []}
    aucs = []
    positions = np.arange(len(X))
    for fold in range(folds):
        test_mask = positions % folds == fold
        train_mask = ~test_mask
        y_train = y.iloc[train_mask]
        y_test = y.iloc[test_mask]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue
        weights, _ = _fit_weights(X.iloc[train_mask], y_train, equal=equal)
        scores = _weighted_score(X.iloc[test_mask], weights)
        auc = _roc_auc(y_test.astype(int).tolist(), scores.astype(float).tolist())
        if auc is not None:
            aucs.append(auc)
    return {"folds_used": len(aucs), "auc_by_fold": aucs}


def _reliability_warnings(
    *,
    settings: BHSACalibrationSettings,
    sample_size: int,
    class_balance: dict[str, int],
) -> list[str]:
    warnings = []
    if sample_size < int(settings.min_sample_size):
        warnings.append(
            f"Only {sample_size} paired record(s) are available; minimum for calibration is {int(settings.min_sample_size)}."
        )
    if len(class_balance) < 2:
        warnings.append("Only one activity class is present; fitted BHSA weights would not be defensible.")
    elif min(class_balance.values()) < int(settings.min_class_count):
        warnings.append(
            f"Minority activity class has fewer than {int(settings.min_class_count)} record(s); class balance is too weak."
        )
    return warnings


def _not_usable(
    settings: BHSACalibrationSettings,
    sample_size: int,
    warnings: list[str],
    *,
    class_balance: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "method_version": CALIBRATION_VERSION,
        "settings": asdict(settings),
        "status": "insufficient_data",
        "do_not_use_calibrated_model": True,
        "sample_size": int(sample_size),
        "warnings": warnings,
        "class_balance": class_balance or {},
        "reliability_warnings": warnings,
        "equal_prior_weights": {col: round(1.0 / len(BHSA_SCORE_COLUMNS), 6) for col in BHSA_SCORE_COLUMNS},
        "equal_prior_baseline": None,
        "evidence_adjusted_weights": None,
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
