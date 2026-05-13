from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .schema import SpeciesTrainingResult, SpeciesTrainingSettings


def train_species_model(
    df,
    *,
    framework,
    settings: SpeciesTrainingSettings,
) -> SpeciesTrainingResult:
    import pandas as pd

    if settings.target_column not in df.columns:
        raise ValueError(f"Target column '{settings.target_column}' was not found.")
    if settings.geography_column and settings.geography_column not in df.columns:
        raise ValueError(f"Geography column '{settings.geography_column}' was not found.")

    predictor_df = _align_training_predictors(df, framework=framework)
    target = _coerce_binary_target(df[settings.target_column])
    if int(target.sum()) < int(settings.min_positive_rows):
        raise ValueError(
            f"Training data has only {int(target.sum())} positive rows; at least {int(settings.min_positive_rows)} are required."
        )
    geography = None
    if settings.geography_column:
        geography = df[settings.geography_column].astype("string").fillna("NA")

    preproc = _fit_feature_preprocessor(predictor_df)
    X = _transform_predictors(predictor_df, preproc=preproc)
    y = target.to_numpy(dtype="float64")
    coef, intercept = _fit_logistic_regression(
        X,
        y,
        learning_rate=float(settings.learning_rate),
        max_iter=int(settings.max_iter),
        l2_strength=float(settings.l2_strength),
    )

    fitted_probs = _predict_probability_matrix(X, coef=coef, intercept=intercept)
    cross_validation = _cross_validation_summary(
        predictor_df,
        target,
        settings=settings,
    )
    geography_holdout = _geography_holdout_summary(
        predictor_df,
        target,
        geography_series=geography,
        settings=settings,
    )
    domain_of_applicability = _build_domain_of_applicability(
        predictor_df,
        settings=settings,
    )
    model_artifact = _build_model_artifact(
        framework=framework,
        settings=settings,
        preproc=preproc,
        coef=coef,
        intercept=intercept,
    )
    model_card = _build_model_card(
        framework=framework,
        settings=settings,
        target=target,
        fitted_probs=fitted_probs,
        cross_validation=cross_validation,
        geography_holdout=geography_holdout,
        predictor_df=predictor_df,
    )
    summary = {
        "species_name": settings.species_name,
        "row_count": int(len(target)),
        "positive_count": int(target.sum()),
        "positive_rate": float(target.mean()),
        "framework_name": framework.manifest.name,
        "framework_version": framework.manifest.version,
        "cross_validation": cross_validation,
        "geography_holdout": geography_holdout,
    }
    return SpeciesTrainingResult(
        model_artifact=model_artifact,
        model_card=model_card,
        domain_of_applicability=domain_of_applicability,
        summary=summary,
        predictor_df=predictor_df,
        target_series=target,
        geography_series=geography,
    )


def _align_training_predictors(df, *, framework):
    import pandas as pd

    registry = framework.feature_registry
    aligned = pd.DataFrame(index=df.index)
    missing_required: list[str] = []
    for predictor in registry.predictor_order:
        source_col = _find_source_column_for_predictor(df.columns, predictor, registry.alias_map)
        if source_col is None:
            aligned[predictor] = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
            if predictor in registry.required_predictors:
                missing_required.append(predictor)
        else:
            aligned[predictor] = df[source_col]
    if missing_required:
        raise ValueError(
            "Training data is missing required framework predictors: " + ", ".join(sorted(missing_required))
        )
    return aligned


def _fit_feature_preprocessor(predictor_df) -> dict[str, dict[str, float]]:
    import pandas as pd

    preproc: dict[str, dict[str, float]] = {}
    for col in predictor_df.columns:
        numeric = pd.to_numeric(predictor_df[col], errors="coerce")
        median = float(numeric.median()) if numeric.notna().any() else 0.0
        filled = numeric.fillna(median)
        mean = float(filled.mean()) if len(filled) else 0.0
        std = float(filled.std(ddof=0)) if len(filled) else 0.0
        preproc[str(col)] = {
            "median": median,
            "mean": mean,
            "std": std if std > 1e-9 else 1.0,
        }
    return preproc


def _transform_predictors(predictor_df, *, preproc: dict[str, dict[str, float]]) -> np.ndarray:
    import pandas as pd

    arrays: list[np.ndarray] = []
    for col in predictor_df.columns:
        numeric = pd.to_numeric(predictor_df[col], errors="coerce")
        cfg = preproc[str(col)]
        filled = numeric.fillna(float(cfg["median"]))
        scaled = (filled.astype("float64") - float(cfg["mean"])) / float(cfg["std"])
        arrays.append(scaled.to_numpy(dtype="float64"))
    if not arrays:
        return np.zeros((len(predictor_df), 0), dtype="float64")
    return np.column_stack(arrays)


def _fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float,
    max_iter: int,
    l2_strength: float,
) -> tuple[np.ndarray, float]:
    n_samples = max(int(X.shape[0]), 1)
    n_features = int(X.shape[1])
    coef = np.zeros(n_features, dtype="float64")
    positive_rate = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
    intercept = float(np.log(positive_rate / (1.0 - positive_rate)))
    for _ in range(max(int(max_iter), 1)):
        probs = _predict_probability_matrix(X, coef=coef, intercept=intercept)
        error = probs - y
        grad_coef = (X.T @ error) / n_samples
        grad_coef += float(l2_strength) * coef
        grad_intercept = float(error.mean())
        coef -= float(learning_rate) * grad_coef
        intercept -= float(learning_rate) * grad_intercept
    return coef, intercept


def _predict_probability_matrix(X: np.ndarray, *, coef: np.ndarray, intercept: float) -> np.ndarray:
    linear = np.clip((X @ coef) + float(intercept), -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-linear))


def _coerce_binary_target(series):
    import pandas as pd

    lowered = series.astype("string").str.strip().str.lower()
    mapping = {
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "present": 1,
        "absent": 0,
    }
    mapped = lowered.map(mapping)
    numeric = pd.to_numeric(series, errors="coerce")
    out = numeric.where(numeric.notna(), mapped)
    if out.isna().any():
        raise ValueError("Target column must be coercible to binary 0/1 values.")
    unique = sorted(set(int(v) for v in out.astype(int).tolist()))
    if any(v not in {0, 1} for v in unique):
        raise ValueError("Target column must contain only binary 0/1 values.")
    return out.astype("int64")


def _cross_validation_summary(predictor_df, target, *, settings: SpeciesTrainingSettings) -> dict[str, Any]:
    folds = _stratified_folds(target.to_numpy(dtype="int64"), n_folds=int(settings.cv_folds))
    fold_metrics: list[dict[str, Any]] = []
    for fold_no, test_idx in enumerate(folds, start=1):
        train_idx = np.array([idx for idx in range(len(target)) if idx not in set(test_idx)], dtype="int64")
        metrics = _fit_and_score_split(
            predictor_df,
            target,
            train_idx=train_idx,
            test_idx=np.asarray(test_idx, dtype="int64"),
            settings=settings,
            label=f"fold_{fold_no}",
        )
        fold_metrics.append(metrics)
    return _aggregate_metric_records(
        fold_metrics,
        strategy="stratified_cv",
        fold_count=len(fold_metrics),
    )


def _geography_holdout_summary(
    predictor_df,
    target,
    *,
    geography_series,
    settings: SpeciesTrainingSettings,
) -> dict[str, Any]:
    if geography_series is None:
        return {"status": "not_available", "reason": "No geography column was provided."}
    groups = [str(x) for x in geography_series.astype("string").fillna("NA").tolist()]
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        return {"status": "not_available", "reason": "Geography column has fewer than 2 unique groups."}

    group_folds = _group_holdout_folds(groups, n_folds=min(int(settings.cv_folds), len(unique_groups)))
    fold_metrics: list[dict[str, Any]] = []
    for fold_no, test_groups in enumerate(group_folds, start=1):
        test_idx = np.array([idx for idx, group in enumerate(groups) if group in test_groups], dtype="int64")
        train_idx = np.array([idx for idx, group in enumerate(groups) if group not in test_groups], dtype="int64")
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        metrics = _fit_and_score_split(
            predictor_df,
            target,
            train_idx=train_idx,
            test_idx=test_idx,
            settings=settings,
            label=f"geo_fold_{fold_no}",
        )
        metrics["holdout_groups"] = list(test_groups)
        fold_metrics.append(metrics)
    if not fold_metrics:
        return {"status": "not_available", "reason": "Geography holdout could not create non-empty train/test splits."}
    summary = _aggregate_metric_records(
        fold_metrics,
        strategy="geography_holdout",
        fold_count=len(fold_metrics),
    )
    summary["status"] = "available"
    summary["geography_column"] = settings.geography_column
    return summary


def _fit_and_score_split(predictor_df, target, *, train_idx, test_idx, settings: SpeciesTrainingSettings, label: str) -> dict[str, Any]:
    train_df = predictor_df.iloc[train_idx]
    test_df = predictor_df.iloc[test_idx]
    train_target = target.iloc[train_idx]
    test_target = target.iloc[test_idx]
    preproc = _fit_feature_preprocessor(train_df)
    X_train = _transform_predictors(train_df, preproc=preproc)
    X_test = _transform_predictors(test_df, preproc=preproc)
    coef, intercept = _fit_logistic_regression(
        X_train,
        train_target.to_numpy(dtype="float64"),
        learning_rate=float(settings.learning_rate),
        max_iter=int(settings.max_iter),
        l2_strength=float(settings.l2_strength),
    )
    probs = _predict_probability_matrix(X_test, coef=coef, intercept=intercept)
    return {
        "fold_label": label,
        "row_count": int(len(test_target)),
        "positive_count": int(train_target.sum()),
        "metrics": _binary_metrics(
            test_target.to_numpy(dtype="float64"),
            probs,
        ),
    }


def _aggregate_metric_records(records: list[dict[str, Any]], *, strategy: str, fold_count: int) -> dict[str, Any]:
    if not records:
        return {"strategy": strategy, "fold_count": 0, "metrics_mean": {}, "fold_metrics": []}
    metric_names = sorted(records[0]["metrics"].keys())
    metrics_mean: dict[str, float | None] = {}
    for metric_name in metric_names:
        values = [record["metrics"].get(metric_name) for record in records]
        numeric = [float(v) for v in values if v is not None]
        metrics_mean[metric_name] = (sum(numeric) / len(numeric)) if numeric else None
    return {
        "strategy": strategy,
        "fold_count": int(fold_count),
        "metrics_mean": metrics_mean,
        "fold_metrics": records,
    }


def _binary_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float | None]:
    clipped = np.clip(probs.astype("float64"), 1e-6, 1.0 - 1e-6)
    truth = y_true.astype("float64")
    predictions = (clipped >= 0.5).astype("float64")
    return {
        "roc_auc": _roc_auc(truth, clipped),
        "log_loss": float(-(truth * np.log(clipped) + ((1.0 - truth) * np.log(1.0 - clipped))).mean()),
        "brier_score": float(np.square(clipped - truth).mean()),
        "accuracy": float((predictions == truth).mean()),
    }


def _roc_auc(y_true: np.ndarray, probs: np.ndarray) -> float | None:
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(probs)
    ranks = np.empty_like(order, dtype="float64")
    ranks[order] = np.arange(1, len(probs) + 1, dtype="float64")
    rank_sum = float(ranks[y_true == 1].sum())
    return float((rank_sum - (positives * (positives + 1) / 2.0)) / (positives * negatives))


def _stratified_folds(y: np.ndarray, *, n_folds: int) -> list[np.ndarray]:
    n_folds = max(2, min(int(n_folds), len(y)))
    fold_lists: list[list[int]] = [[] for _ in range(n_folds)]
    for cls in (1, 0):
        idxs = np.where(y == cls)[0].tolist()
        for position, idx in enumerate(idxs):
            fold_lists[position % n_folds].append(int(idx))
    return [np.asarray(sorted(fold), dtype="int64") for fold in fold_lists if fold]


def _group_holdout_folds(groups: list[str], *, n_folds: int) -> list[list[str]]:
    unique_groups = sorted(set(groups))
    n_folds = max(2, min(int(n_folds), len(unique_groups)))
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for position, group in enumerate(unique_groups):
        folds[position % n_folds].append(str(group))
    return [fold for fold in folds if fold]


def _build_domain_of_applicability(predictor_df, *, settings: SpeciesTrainingSettings) -> dict[str, Any]:
    import pandas as pd

    predictors: dict[str, dict[str, float]] = {}
    for col in predictor_df.columns:
        numeric = pd.to_numeric(predictor_df[col], errors="coerce")
        if numeric.notna().any():
            predictors[str(col)] = {
                "p01": float(numeric.quantile(0.01)),
                "p99": float(numeric.quantile(0.99)),
                "missing_rate": float(numeric.isna().mean()),
            }
        else:
            predictors[str(col)] = {
                "p01": 0.0,
                "p99": 0.0,
                "missing_rate": 1.0,
            }
    return {
        "artifact_version": "species_domain_v1",
        "species_name": settings.species_name,
        "predictor_domain": predictors,
        "domain_score_thresholds": {"inside": 0.85, "borderline": 0.60},
        "missing_feature_fraction_thresholds": {"inside": 0.10, "borderline": 0.25},
    }


def _build_model_artifact(*, framework, settings: SpeciesTrainingSettings, preproc, coef, intercept) -> dict[str, Any]:
    return {
        "artifact_version": "species_logistic_model_v1",
        "created_date": datetime.now(timezone.utc).date().isoformat() if not settings.deterministic_output else None,
        "species_name": settings.species_name,
        "framework_name": framework.manifest.name,
        "framework_version": framework.manifest.version,
        "compatible_feature_profile_name": framework.manifest.compatible_feature_profile_name,
        "predictor_order": list(framework.feature_registry.predictor_order),
        "coefficients": {
            str(feature): float(weight)
            for feature, weight in zip(framework.feature_registry.predictor_order, coef.tolist())
        },
        "intercept": float(intercept),
        "preprocessing": preproc,
    }


def _build_model_card(
    *,
    framework,
    settings: SpeciesTrainingSettings,
    target,
    fitted_probs: np.ndarray,
    cross_validation: dict[str, Any],
    geography_holdout: dict[str, Any],
    predictor_df,
) -> dict[str, Any]:
    return {
        "artifact_version": "species_model_card_v1",
        "species_name": settings.species_name,
        "framework_name": framework.manifest.name,
        "framework_version": framework.manifest.version,
        "compatible_feature_profile_name": framework.manifest.compatible_feature_profile_name,
        "training_settings": asdict(settings),
        "training_data": {
            "row_count": int(len(target)),
            "positive_count": int(target.sum()),
            "positive_rate": float(target.mean()),
            "predictor_count": int(len(predictor_df.columns)),
        },
        "fit_metrics": _binary_metrics(target.to_numpy(dtype="float64"), fitted_probs),
        "cross_validation": cross_validation,
        "geography_holdout": geography_holdout,
        "domain_of_applicability_summary": {
            "message": "Predictions should be interpreted primarily for rows that remain inside the recorded feature domain."
        },
        "limitations": [
            "Probabilities are only as defensible as the supplied historical survey labels.",
            "Geography holdout requires a meaningful project/package/spatial grouping column.",
            "Rows outside the recorded feature domain should be treated with caution.",
        ],
    }


def _find_source_column_for_predictor(columns, predictor: str, alias_map: dict[str, str] | None = None) -> str | None:
    col_set = set(columns)
    if predictor in col_set:
        return predictor
    alias_map = alias_map or {}
    direct_alias = alias_map.get(predictor)
    if direct_alias and direct_alias in col_set:
        return direct_alias
    for alias, canonical in alias_map.items():
        if canonical == predictor and alias in col_set:
            return alias
    return None
