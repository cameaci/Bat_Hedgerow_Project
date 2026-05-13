import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from hedge_features.screening import ScreeningSettings, load_framework_bundle, screen_dataframe
from hedge_features.species import SpeciesTrainingSettings, train_species_model, write_species_artifacts


def _framework_copy(tmp_path: Path) -> Path:
    src = Path(__file__).resolve().parents[1] / "hedge_features" / "frameworks" / "bats_screening_v1"
    dest = tmp_path / "bats_screening_v1"
    shutil.copytree(src, dest)
    return dest


def _synthetic_training_df(framework, *, n_rows: int = 96) -> pd.DataFrame:
    idx = np.arange(n_rows, dtype="float64")
    data: dict[str, np.ndarray] = {}
    for pos, predictor in enumerate(framework.feature_registry.predictor_order):
        base = np.sin((idx + pos) / 6.0) + ((idx % 9.0) / 9.0) + (pos * 0.005)
        if predictor.startswith("dist_"):
            values = np.abs(base) * 180.0 + 10.0
        elif predictor.startswith("buf") or predictor.startswith("roostpx_") or predictor.startswith("mhb_"):
            values = np.clip((base + 1.2) / 2.6, 0.0, 1.0)
        elif predictor.startswith("geom_") or predictor.startswith("net_") or predictor.startswith("pt_"):
            values = np.abs(base) * 3.0
        else:
            values = base
        data[predictor] = values

    predictors = framework.feature_registry.predictor_order
    signal = (
        (data[predictors[0]] * 0.9)
        + (data[predictors[1]] * 0.6)
        - (data[predictors[2]] * 0.25)
        + (data[predictors[3]] * 0.35)
    )
    threshold = float(np.quantile(signal, 0.58))
    target = (signal >= threshold).astype(int)

    df = pd.DataFrame(data)
    df.loc[df.index[::11], predictors[4]] = np.nan
    df.loc[df.index[::13], predictors[5]] = np.nan
    df["target_species"] = target
    df["project_area"] = np.array(["North", "South", "East", "West"])[(idx.astype(int) % 4)]
    return df


def test_species_training_and_runtime_scoring_pipeline(tmp_path):
    framework_dir = _framework_copy(tmp_path)
    framework = load_framework_bundle(framework_dir=framework_dir)
    df = _synthetic_training_df(framework)

    result = train_species_model(
        df,
        framework=framework,
        settings=SpeciesTrainingSettings(
            species_name="Pipistrellus pipistrellus",
            target_column="target_species",
            geography_column="project_area",
            cv_folds=4,
            min_positive_rows=5,
            max_iter=250,
        ),
    )

    assert result.model_card["cross_validation"]["fold_count"] == 4
    assert result.model_card["cross_validation"]["metrics_mean"]["roc_auc"] is not None
    assert result.model_card["geography_holdout"]["status"] == "available"

    written = write_species_artifacts(
        result,
        output_dir=framework_dir / "species_models",
        framework_dir=framework_dir,
    )
    assert Path(written["species_model"]).exists()
    assert Path(written["model_card_json"]).exists()
    assert Path(written["domain_of_applicability"]).exists()

    framework_reloaded = load_framework_bundle(framework_dir=framework_dir)
    assert "Pipistrellus pipistrellus" in framework_reloaded.species_models
    assert "Pipistrellus pipistrellus" in framework_reloaded.manifest.species_targets_available

    screening_input = df.drop(columns=["target_species"]).copy()
    run = screen_dataframe(
        screening_input,
        framework=framework_reloaded,
        settings=ScreeningSettings(species_module_enabled=True),
        feature_profile_name=framework_reloaded.manifest.compatible_feature_profile_name,
    )

    prefix = "species_pipistrellus_pipistrellus"
    assert f"{prefix}_probability" in run.results_df.columns
    assert f"{prefix}_domain_status" in run.results_df.columns
    assert f"{prefix}_reason_codes" in run.results_df.columns
    assert run.run_summary["species_models"]["loaded_species"] == ["Pipistrellus pipistrellus"]
