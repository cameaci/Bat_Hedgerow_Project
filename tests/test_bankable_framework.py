from hedge_features.screening import get_bundled_framework_names, load_framework_bundle


def test_bankable_framework_bundle_loads():
    framework = load_framework_bundle("bats_bankable_england_v2")

    assert framework.manifest.name == "bats_bankable_england_v2"
    assert framework.manifest.compatible_feature_profile_name == "bats_bankable_england_v2"
    assert "mhb_roost_proxy_score" not in framework.feature_registry.predictor_order
    assert "hedge_struct_height_mean_5m" in framework.feature_registry.predictor_order


def test_bankable_framework_is_bundled():
    assert "bats_bankable_england_v2" in get_bundled_framework_names()
