from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .schema import SpeciesPredictionSummary


class JsonSpeciesLogisticModel:
    def __init__(self, *, model_artifact: dict[str, Any], domain_artifact: dict[str, Any] | None = None):
        self.model_artifact = dict(model_artifact)
        self.domain_artifact = dict(domain_artifact or {})
        self.species_name = str(self.model_artifact.get("species_name", "species"))
        self.predictor_order = [str(x) for x in self.model_artifact.get("predictor_order", [])]
        self.coefficients = {
            str(k): float(v)
            for k, v in (self.model_artifact.get("coefficients") or {}).items()
        }
        self.intercept = float(self.model_artifact.get("intercept", 0.0))
        self.preprocessing = {
            str(k): {
                "median": float((v or {}).get("median", 0.0)),
                "mean": float((v or {}).get("mean", 0.0)),
                "std": float((v or {}).get("std", 1.0)) or 1.0,
            }
            for k, v in (self.model_artifact.get("preprocessing") or {}).items()
        }
        self.domain_predictors = {
            str(k): dict(v or {})
            for k, v in (self.domain_artifact.get("predictor_domain") or {}).items()
        }
        self.domain_score_thresholds = {
            "inside": float((self.domain_artifact.get("domain_score_thresholds") or {}).get("inside", 0.85)),
            "borderline": float((self.domain_artifact.get("domain_score_thresholds") or {}).get("borderline", 0.60)),
        }

    @property
    def slug(self) -> str:
        out = []
        for ch in self.species_name.strip().lower():
            if ch.isalnum():
                out.append(ch)
            else:
                out.append("_")
        slug = "".join(out).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        return slug or "species"

    def predict_probability(self, predictor_df):
        import pandas as pd

        arrays: list[np.ndarray] = []
        for predictor in self.predictor_order:
            numeric = pd.to_numeric(predictor_df.get(predictor), errors="coerce")
            cfg = self.preprocessing.get(predictor, {"median": 0.0, "mean": 0.0, "std": 1.0})
            filled = numeric.fillna(float(cfg["median"]))
            scaled = (filled.astype("float64") - float(cfg["mean"])) / float(cfg["std"])
            arrays.append(scaled.to_numpy(dtype="float64"))
        X = np.column_stack(arrays) if arrays else np.zeros((len(predictor_df), 0), dtype="float64")
        coef = np.asarray([self.coefficients.get(name, 0.0) for name in self.predictor_order], dtype="float64")
        linear = np.clip((X @ coef) + self.intercept, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-linear))

    def assess_domain(self, predictor_df):
        import pandas as pd

        score_values: list[float] = []
        labels: list[str] = []
        reason_codes: list[str] = []
        for _, row in predictor_df.iterrows():
            total = max(len(self.predictor_order), 1)
            in_range = 0
            present = 0
            row_reason_codes: list[str] = []
            for predictor in self.predictor_order:
                value = pd.to_numeric(pd.Series([row.get(predictor)]), errors="coerce").iloc[0]
                if pd.isna(value):
                    row_reason_codes.append("MISSING_PREDICTOR")
                    continue
                present += 1
                domain = self.domain_predictors.get(predictor)
                if domain is None:
                    in_range += 1
                    continue
                if float(domain.get("p01", value)) <= float(value) <= float(domain.get("p99", value)):
                    in_range += 1
                else:
                    row_reason_codes.append("OUTSIDE_FEATURE_DOMAIN")
            coverage = present / total
            in_range_frac = in_range / total
            domain_score = (0.7 * in_range_frac) + (0.3 * coverage)
            if domain_score >= self.domain_score_thresholds["inside"]:
                label = "Inside"
            elif domain_score >= self.domain_score_thresholds["borderline"]:
                label = "Borderline"
            else:
                label = "Outside"
            score_values.append(float(domain_score))
            labels.append(label)
            reason_codes.append("|".join(sorted(set(row_reason_codes))))
        return score_values, labels, reason_codes


def discover_species_model_artifacts(species_models_dir: str | Path) -> list[dict[str, str]]:
    root = Path(species_models_dir)
    if not root.exists():
        return []
    out: list[dict[str, str]] = []
    for model_path in sorted(root.glob("species_*_model.json")):
        stem = model_path.stem.removesuffix("_model")
        domain_path = model_path.with_name(f"{stem}_domain.json")
        model_card_path = model_path.with_name(f"{stem}_model_card.json")
        out.append(
            {
                "model_path": str(model_path),
                "domain_path": str(domain_path) if domain_path.exists() else "",
                "model_card_path": str(model_card_path) if model_card_path.exists() else "",
            }
        )
    return out


def load_species_models(species_models_dir: str | Path) -> dict[str, JsonSpeciesLogisticModel]:
    loaded: dict[str, JsonSpeciesLogisticModel] = {}
    for paths in discover_species_model_artifacts(species_models_dir):
        model_artifact = json.loads(Path(paths["model_path"]).read_text(encoding="utf-8"))
        domain_artifact = None
        if paths.get("domain_path"):
            domain_artifact = json.loads(Path(paths["domain_path"]).read_text(encoding="utf-8"))
        model = JsonSpeciesLogisticModel(model_artifact=model_artifact, domain_artifact=domain_artifact)
        loaded[model.species_name] = model
    return loaded


def apply_species_models(df, *, species_models: dict[str, JsonSpeciesLogisticModel], predictor_df):
    import pandas as pd

    out = df.copy()
    summary = SpeciesPredictionSummary(loaded_species=sorted(species_models))
    for species_name in sorted(species_models):
        model = species_models[species_name]
        probabilities = model.predict_probability(predictor_df)
        domain_scores, domain_labels, reason_codes = model.assess_domain(predictor_df)
        prefix = f"species_{model.slug}"
        out[f"{prefix}_probability"] = pd.Series(probabilities, index=out.index, dtype="float64").round(6)
        out[f"{prefix}_domain_score"] = pd.Series(domain_scores, index=out.index, dtype="float64").round(6)
        out[f"{prefix}_domain_status"] = pd.Series(domain_labels, index=out.index, dtype="object")
        out[f"{prefix}_reason_codes"] = pd.Series(reason_codes, index=out.index, dtype="object")
        summary.counts_by_domain_status[species_name] = _series_counts(out[f"{prefix}_domain_status"])
        summary.mean_probability_by_species[species_name] = float(out[f"{prefix}_probability"].astype("float64").mean())
    return out, summary


def _series_counts(series) -> dict[str, int]:
    counts = series.astype("string").fillna("NA").value_counts(dropna=False)
    return {str(idx): int(val) for idx, val in counts.items()}
