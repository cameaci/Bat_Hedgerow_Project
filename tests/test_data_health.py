import pandas as pd

from hedge_features.data_health import build_data_catalogue, build_feature_health
from hedge_features.datasets.registry import DatasetRegistry


def test_data_catalogue_and_feature_health_classify_sources_and_features():
    profile = {
        "name": "bankable_test",
        "datasets": {
            "viirs_nightlights": {
                "dataset_class": "authenticated_free",
                "source_type": "raster_time_series",
                "authenticated": True,
                "auto_provider": {"type": "unsupported"},
            },
            "worldcover": {
                "dataset_class": "anonymous_open",
                "source_type": "raster_categorical",
            },
        },
    }
    registry = DatasetRegistry(profile_datasets=profile["datasets"])
    catalogue = build_data_catalogue(
        profile=profile,
        registry=registry,
        used_dataset_names=["viirs_nightlights", "worldcover"],
        guidance_regime_version="bct4_ne2025_england_v1",
    )

    statuses = {item["name"]: item["status"] for item in catalogue["datasets"]}
    assert statuses["viirs_nightlights"] == "authenticated_required"
    assert statuses["worldcover"] == "missing"

    df = pd.DataFrame(
        {
            "buf100_worldcover_tree_pct": [0.5, 0.0],
            "buf100_nightlight_mean": [None, None],
            "geom_length_m": [120.0, 160.0],
        }
    )
    health = build_feature_health(
        df,
        data_catalogue=catalogue,
        guidance_regime_version="bct4_ne2025_england_v1",
    )

    support_states = {item["column_name"]: item["support_state"] for item in health["features"]}
    assert support_states["buf100_worldcover_tree_pct"] in {"measured_or_derived", "derived_internal"}
    assert support_states["buf100_nightlight_mean"] == "missing_source"
