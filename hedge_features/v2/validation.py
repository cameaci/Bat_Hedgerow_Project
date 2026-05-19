from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


VALIDATION_DIAGNOSTICS_VERSION = "bhsa_acoustic_validation_diagnostics_v2"


@dataclass(frozen=True, slots=True)
class ValidationDiagnosticsSettings:
    hedgerow_id_column: str = "hedgerow_id"
    score_column: str = "bhsa_score"
    class_column: str = "bhsa_class"
    acoustic_activity_column: str = "acoustic_total_passes"
    positive_activity_threshold: float = 1.0
    survey_required_classes: tuple[str, ...] = ("Good", "Excellent")
    survey_required_score_threshold: float = 1.70
    excellent_score_threshold: float = 2.40
    min_sample_size_for_auc: int = 10
    max_examples: int = 10


def build_validation_diagnostics(df, *, settings: ValidationDiagnosticsSettings | None = None) -> dict[str, Any]:
    """Compare BHSA survey-effort decisions against structured acoustic evidence."""
    import pandas as pd

    settings = settings or ValidationDiagnosticsSettings()
    missing = [col for col in (settings.score_column, settings.acoustic_activity_column) if col not in df.columns]
    if missing:
        return _not_ready(settings, len(df), [f"Missing required validation column(s): {', '.join(missing)}"])

    work = df.copy()
    work["_bhsa_score"] = pd.to_numeric(work[settings.score_column], errors="coerce")
    work["_acoustic_activity"] = pd.to_numeric(work[settings.acoustic_activity_column], errors="coerce")
    usable = work["_bhsa_score"].notna() & work["_acoustic_activity"].notna()
    work = work.loc[usable].copy()
    if work.empty:
        return _not_ready(settings, 0, ["No rows have both a BHSA score and acoustic activity value."])

    if settings.class_column in work.columns:
        class_text = work[settings.class_column].astype("string")
        survey_required = class_text.isin(settings.survey_required_classes)
    else:
        survey_required = work["_bhsa_score"] >= float(settings.survey_required_score_threshold)
    acoustic_positive = work["_acoustic_activity"] >= float(settings.positive_activity_threshold)
    work["_bhsa_survey_required"] = survey_required.astype(bool)
    work["_acoustic_positive"] = acoustic_positive.astype(bool)

    tp = int((work["_bhsa_survey_required"] & work["_acoustic_positive"]).sum())
    fp = int((work["_bhsa_survey_required"] & ~work["_acoustic_positive"]).sum())
    tn = int((~work["_bhsa_survey_required"] & ~work["_acoustic_positive"]).sum())
    fn = int((~work["_bhsa_survey_required"] & work["_acoustic_positive"]).sum())
    metrics = {
        "sensitivity": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "precision": _safe_div(tp, tp + fp),
        "negative_predictive_value": _safe_div(tn, tn + fn),
        "accuracy": _safe_div(tp + tn, len(work)),
        "auc": _roc_auc(work["_acoustic_positive"].astype(int).tolist(), work["_bhsa_score"].astype(float).tolist())
        if len(work) >= int(settings.min_sample_size_for_auc)
        else None,
    }
    caveats = _diagnostic_caveats(work, settings=settings, dropped_rows=int((~usable).sum()), metrics=metrics)

    return {
        "method_version": VALIDATION_DIAGNOSTICS_VERSION,
        "settings": asdict(settings),
        "status": "ready_for_technical_review",
        "sample_size": int(len(work)),
        "thresholds": {
            "survey_required_score_threshold": float(settings.survey_required_score_threshold),
            "excellent_score_threshold": float(settings.excellent_score_threshold),
            "positive_activity_threshold": float(settings.positive_activity_threshold),
        },
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive_high_score_no_evidence": fp,
            "true_negative": tn,
            "false_negative_low_score_positive_evidence": fn,
        },
        "metrics": metrics,
        "class_activity_summary": _class_activity_summary(work, settings=settings),
        "examples": _diagnostic_examples(work, settings=settings),
        "caveats": caveats,
    }


def _not_ready(settings: ValidationDiagnosticsSettings, sample_size: int, caveats: list[str]) -> dict[str, Any]:
    return {
        "method_version": VALIDATION_DIAGNOSTICS_VERSION,
        "settings": asdict(settings),
        "status": "not_ready",
        "sample_size": int(sample_size),
        "thresholds": {
            "survey_required_score_threshold": float(settings.survey_required_score_threshold),
            "excellent_score_threshold": float(settings.excellent_score_threshold),
            "positive_activity_threshold": float(settings.positive_activity_threshold),
        },
        "confusion_matrix": {},
        "metrics": {},
        "class_activity_summary": [],
        "examples": {},
        "caveats": caveats,
    }


def _class_activity_summary(work, *, settings: ValidationDiagnosticsSettings) -> list[dict[str, Any]]:
    class_col = settings.class_column if settings.class_column in work.columns else None
    if class_col is None:
        work = work.assign(_derived_bhsa_class=work["_bhsa_score"].map(_class_from_score))
        class_col = "_derived_bhsa_class"
    rows = []
    for class_name, group in work.groupby(class_col, dropna=False, sort=True):
        rows.append(
            {
                "bhsa_class": str(class_name),
                "hedgerow_count": int(len(group)),
                "acoustic_positive_count": int(group["_acoustic_positive"].sum()),
                "acoustic_total_activity": float(group["_acoustic_activity"].sum()),
                "acoustic_mean_activity": round(float(group["_acoustic_activity"].mean()), 6),
                "mean_bhsa_score": round(float(group["_bhsa_score"].mean()), 6),
            }
        )
    return rows


def _diagnostic_examples(work, *, settings: ValidationDiagnosticsSettings) -> dict[str, list[dict[str, Any]]]:
    id_col = settings.hedgerow_id_column if settings.hedgerow_id_column in work.columns else None
    columns = [col for col in (id_col, settings.class_column if settings.class_column in work.columns else None) if col]
    columns.extend(["_bhsa_score", "_acoustic_activity"])
    examples = {
        "high_score_no_acoustic_evidence": work.loc[
            work["_bhsa_survey_required"] & ~work["_acoustic_positive"], columns
        ]
        .sort_values(["_bhsa_score", "_acoustic_activity"], ascending=[False, True])
        .head(int(settings.max_examples)),
        "low_score_positive_acoustic_evidence": work.loc[
            ~work["_bhsa_survey_required"] & work["_acoustic_positive"], columns
        ]
        .sort_values(["_acoustic_activity", "_bhsa_score"], ascending=[False, True])
        .head(int(settings.max_examples)),
    }
    return {key: _records(value) for key, value in examples.items()}


def _diagnostic_caveats(work, *, settings: ValidationDiagnosticsSettings, dropped_rows: int, metrics: dict[str, Any]) -> list[str]:
    caveats = []
    if dropped_rows:
        caveats.append(f"{dropped_rows} row(s) were excluded because BHSA score or acoustic activity was missing.")
    if len(work) < int(settings.min_sample_size_for_auc):
        caveats.append("Sample size is below the AUC reliability threshold; use case review rather than model performance claims.")
    if work["_acoustic_positive"].nunique() < 2:
        caveats.append("Only one acoustic evidence class is present; sensitivity/specificity and AUC are limited.")
    if metrics.get("auc") is None:
        caveats.append("ROC/AUC was not calculated because sample size or class balance is insufficient.")
    caveats.append("Validation reflects the supplied survey-result table only; absence of evidence is not proof of absence.")
    return caveats


def _records(df) -> list[dict[str, Any]]:
    renamed = df.rename(columns={"_bhsa_score": "bhsa_score", "_acoustic_activity": "acoustic_activity"})
    return renamed.to_dict(orient="records")


def _class_from_score(score: float | None) -> str:
    if score is None or score != score:
        return "Incomplete"
    if score < 1.70:
        return "Poor"
    if score <= 2.39:
        return "Good"
    return "Excellent"


def _safe_div(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


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
