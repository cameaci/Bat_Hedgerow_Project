from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

from importlib import resources

from ..species import load_species_models
from .schema import ConfidenceRules, FeatureRegistry, FrameworkManifest, LoadedFramework, ThresholdConfig


class JsonLinearScoreModel:
    """Simple, versioned scoring model wrapper for framework artefacts.

    This produces a bounded ranking score in [0, 1]. It is intentionally exposed
    as a prioritisation score and not a calibrated probability.
    """

    def __init__(self, spec: dict[str, Any]):
        self.spec = spec
        self.model_type = str(spec.get("model_type", "linear_logistic_v1"))
        if self.model_type != "linear_logistic_v1":
            raise ValueError(f"Unsupported model_type '{self.model_type}' in triage model artefact.")
        self.intercept = float(spec.get("intercept", 0.0))
        self.missing_fraction_penalty = float(spec.get("missing_fraction_penalty", 0.0))
        self.output_clip_min = float(spec.get("output_clip_min", 0.0))
        self.output_clip_max = float(spec.get("output_clip_max", 1.0))
        self.terms = [dict(t) for t in (spec.get("terms") or [])]
        if not self.terms:
            raise ValueError("Triage model artefact has no scoring terms.")

    def predict_score(self, df_features):  # pragma: no cover - covered via engine tests
        import numpy as np
        import pandas as pd

        n = len(df_features)
        linear = np.full(n, self.intercept, dtype="float64")
        missing_counts = np.zeros(n, dtype="float64")
        total_terms = max(len(self.terms), 1)

        for term in self.terms:
            feature = str(term["feature"])
            weight = float(term.get("weight", 0.0))
            transform = str(term.get("transform", "zscore"))
            missing_fill = float(term.get("missing_fill", 0.0))
            s = pd.to_numeric(df_features.get(feature), errors="coerce")
            missing_mask = s.isna()
            missing_counts += missing_mask.to_numpy(dtype="float64")
            vals = self._transform_series(s, transform=transform, term=term)
            vals = vals.fillna(missing_fill)
            linear += weight * vals.to_numpy(dtype="float64")

        if self.missing_fraction_penalty:
            linear -= self.missing_fraction_penalty * (missing_counts / total_terms)

        scores = 1.0 / (1.0 + np.exp(-np.clip(linear, -20, 20)))
        return np.clip(scores, self.output_clip_min, self.output_clip_max)

    def _transform_series(self, series, *, transform: str, term: dict[str, Any]):
        import numpy as np
        import pandas as pd

        x = pd.to_numeric(series, errors="coerce")
        if transform == "distance_decay":
            scale = max(float(term.get("scale", 100.0)), 1e-6)
            x = x.clip(lower=0)
            return 1.0 / (1.0 + (x / scale))
        if transform == "unit_interval_clip":
            return x.clip(lower=0.0, upper=1.0)
        if transform == "zscore_log1p":
            center = float(term.get("center", 0.0))
            scale = float(term.get("scale", 1.0)) or 1.0
            z = (np.log1p(x.clip(lower=0)) - center) / scale
            return pd.Series(z, index=x.index).clip(
                lower=_opt_float(term.get("clip_min")),
                upper=_opt_float(term.get("clip_max")),
            )
        if transform == "zscore":
            center = float(term.get("center", 0.0))
            scale = float(term.get("scale", 1.0)) or 1.0
            z = (x - center) / scale
            return z.clip(
                lower=_opt_float(term.get("clip_min")),
                upper=_opt_float(term.get("clip_max")),
            )
        if transform == "identity":
            return x
        raise ValueError(f"Unsupported transform '{transform}' in triage model artefact.")


def get_bundled_framework_names() -> list[str]:
    try:
        root = resources.files("hedge_features.frameworks")
    except Exception:
        return []
    names: list[str] = []
    for child in root.iterdir():
        if child.is_dir():
            names.append(child.name)
    return sorted(names)


def load_framework_bundle(name: str = "bats_screening_v1", framework_dir: str | Path | None = None) -> LoadedFramework:
    root = Path(framework_dir) if framework_dir else _bundled_framework_dir(name)
    manifest = FrameworkManifest.from_dict(_read_json(root / "framework_manifest.json"))
    registry = FeatureRegistry.from_dict(_read_json(root / manifest.feature_registry_artifact))
    thresholds = ThresholdConfig.from_dict(_read_json(root / manifest.thresholds_artifact))
    confidence_rules = ConfidenceRules.from_dict(_read_json(root / manifest.confidence_rules_artifact))
    model = _load_model_artifact(root / manifest.model_artifact)
    preprocessor = None
    if manifest.preprocessor_artifact:
        pp_path = root / manifest.preprocessor_artifact
        if pp_path.exists():
            preprocessor = _load_pickle_or_joblib(pp_path)
    species_models_dir = root / manifest.species_models_dir
    species_models = load_species_models(species_models_dir)
    discovered_species = sorted(species_models)
    if discovered_species:
        manifest.species_targets_available = sorted(set(manifest.species_targets_available + discovered_species))
    return LoadedFramework(
        root_dir=root,
        manifest=manifest,
        feature_registry=registry,
        thresholds=thresholds,
        confidence_rules=confidence_rules,
        model=model,
        preprocessor=preprocessor,
        species_models=species_models,
    )


def compute_profile_hash_short(profile_name: str) -> str | None:
    try:
        base = resources.files("hedge_features.profiles")
        for suffix in (".yaml", ".yml", ".json"):
            candidate = base / f"{profile_name}{suffix}"
            if candidate.is_file():
                data = candidate.read_bytes()
                return hashlib.sha1(data).hexdigest()[:12]
    except Exception:
        return None
    return None


def build_framework_snapshot(framework: LoadedFramework) -> dict[str, Any]:
    return {
        "name": framework.manifest.name,
        "version": framework.manifest.version,
        "created_date": framework.manifest.created_date,
        "compatible_feature_profile_name": framework.manifest.compatible_feature_profile_name,
        "compatible_feature_profile_version": framework.manifest.compatible_feature_profile_version,
        "compatible_feature_profile_hash": framework.manifest.compatible_feature_profile_hash,
        "model_artifact": framework.manifest.model_artifact,
        "feature_registry_version": framework.feature_registry.version,
        "thresholds_version": framework.thresholds.version,
        "confidence_rules_version": framework.confidence_rules.version,
        "species_targets_available": sorted(framework.species_models),
    }


def _bundled_framework_dir(name: str) -> Path:
    try:
        return Path(resources.files("hedge_features.frameworks").joinpath(name))
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Bundled frameworks package is unavailable: {exc}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Framework artefact missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_model_artifact(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Triage model artefact missing: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return JsonLinearScoreModel(_read_json(path))
    if suffix in {".joblib", ".pkl", ".pickle"}:
        return _load_pickle_or_joblib(path)
    raise ValueError(f"Unsupported model artefact extension '{suffix}' for {path.name}.")


def _load_pickle_or_joblib(path: Path):
    if path.suffix.lower() == ".joblib":
        try:
            import joblib  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                f"joblib is required to load {path.name}. Install joblib or use the bundled JSON model."
            ) from exc
        return joblib.load(path)
    with path.open("rb") as f:
        return pickle.load(f)


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None
