from pathlib import Path

from hedge_features.profile_loader import PACKAGE_PROFILE_DIR, resolve_profile


def test_bats_profile_loads():
    profile, path = resolve_profile("bats_v1")
    assert profile["name"] == "bats_v1"
    assert path.parent == PACKAGE_PROFILE_DIR
    assert "feature_modules" in profile


def test_bankable_bats_profile_loads():
    profile, path = resolve_profile("bats_bankable_england_v2")
    assert profile["name"] == "bats_bankable_england_v2"
    assert profile["guidance_regime_version"] == "bct4_ne2025_england_v1"
    assert path.parent == PACKAGE_PROFILE_DIR
    assert "feature_modules" in profile
