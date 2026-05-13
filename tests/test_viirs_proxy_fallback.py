import pandas as pd

from hedge_features.pipeline import _apply_viirs_proxy_if_missing


def test_viirs_proxy_fills_null_columns():
    df = pd.DataFrame(
        {
            "buf100_worldcover_built_pct": [0.2, 0.0],
            "buf100_os_road_density_m_per_ha": [50.0, 0.0],
            "buf100_nightlight_mean": [None, None],
            "buf100_nightlight_median": [None, 5.0],  # second row already populated
            "buf100_nightlight_p90": [None, None],
        }
    )
    module_cfg = {
        "proxy_if_missing": "worldcover_roads",
        "buffers_m": [100],
        "stats": ["mean", "median", "p90"],
        "column_template": "buf{radius}_nightlight_{stat}",
    }
    out, notes = _apply_viirs_proxy_if_missing(df, module_cfg)
    assert out["buf100_nightlight_mean"].notna().all()
    assert out.loc[1, "buf100_nightlight_median"] == 5.0
    assert out["buf100_nightlight_p90"].notna().all()
    assert notes

