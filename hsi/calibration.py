"""Validate / calibrate the HSI against real survey data (crossing-point or activity counts).

The WSP ecologists explicitly wanted the calculated suitability compared with observed bat
activity. This module joins an activity table to the scored hedgerows and reports how well
the HSI tracks activity (Spearman rank correlation, AUC for presence), and can optionally
re-tune the SI weights to better match the observed activity (Nelder-Mead). Only scipy is
used — no scikit-learn/statsmodels.

Calibration is a *diagnostic*: it does not change scores unless the user chooses to apply
the suggested weights.
"""

from __future__ import annotations

from . import config
from .score import apply_scoring


def join_activity(scored_df, activity_df, *, activity_col: str, join_col: str | None = None, id_col: str = "hf_uid"):
    """Attach an ``activity`` column to the scored hedgerows by key join. Returns a copy."""
    out = scored_df.copy()
    if join_col and join_col in activity_df.columns:
        left_key = join_col if join_col in out.columns else id_col
        amap = dict(zip(activity_df[join_col], activity_df[activity_col]))
    elif id_col in activity_df.columns:
        left_key = id_col
        amap = dict(zip(activity_df[id_col], activity_df[activity_col]))
    else:
        out["activity"] = None
        return out
    out["activity"] = [amap.get(v) for v in out[left_key]]
    return out


def match_activity_points(scored_gdf, points_gdf, *, activity_col: str, max_dist_m: float = 50.0):
    """Spatially attach point survey activity to the nearest hedgerow within ``max_dist_m``."""
    import geopandas as gpd

    out = scored_gdf.copy()
    pts = points_gdf.to_crs(out.crs) if points_gdf.crs is not None else points_gdf
    joined = gpd.sjoin_nearest(out[["hf_uid", out.geometry.name]], pts[[activity_col, pts.geometry.name]],
                               how="left", max_distance=max_dist_m, distance_col="_d")
    agg = joined.groupby("hf_uid")[activity_col].mean()
    out["activity"] = [agg.get(u) for u in out["hf_uid"]]
    return out


def _safe_spearman(x, y) -> float | None:
    """Spearman correlation that returns None for constant/short inputs (no scipy warning)."""
    import numpy as np
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    if x.size < 3 or y.size < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return None
    r = spearmanr(x, y).statistic
    return None if r != r else float(r)


def _auc(scores, presence) -> float | None:
    """AUC of ``scores`` discriminating presence (Mann-Whitney U / (n_pos*n_neg))."""
    import numpy as np
    from scipy.stats import rankdata

    scores = np.asarray(scores, dtype="float64")
    presence = np.asarray(presence, dtype=bool)
    n_pos = int(presence.sum())
    n_neg = int((~presence).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores)
    u = ranks[presence].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def calibrate(scored_df, *, activity_col: str = "activity",
              settings: config.ScoreSettings | None = None, optimize: bool = False) -> dict:
    """Report how well the HSI tracks observed activity; optionally suggest re-tuned weights."""
    import numpy as np

    settings = settings or config.ScoreSettings()
    df = scored_df
    act = df[activity_col].astype("float64") if activity_col in df.columns else None
    if act is None:
        return {"error": f"No '{activity_col}' column; join survey data first."}
    mask = act.notna() & df["hsi_priority"].notna()
    n = int(mask.sum())
    if n < 3:
        return {"error": f"Only {n} hedgerows have both a priority and activity; need >= 3."}

    a = act[mask].to_numpy()
    priority = df.loc[mask, "hsi_priority"].astype("float64").to_numpy()
    struct = df.loc[mask, "hsi_structural_A"].astype("float64").to_numpy()

    sp_priority = _safe_spearman(priority, a)
    sp_struct = _safe_spearman(struct, a)
    presence = a > np.median(a)
    auc = _auc(priority, presence)

    report = {
        "n": n,
        "spearman_priority_vs_activity": _round(sp_priority),
        "spearman_structuralA_vs_activity": _round(sp_struct),
        "auc_priority_vs_presence": _round(auc),
        "interpretation": _interpret(sp_priority),
    }
    if optimize:
        if sp_struct is None:
            report["suggested_si_weights"] = dict(settings.si_weights)
            report["optimize_note"] = (
                "Structural scores do not vary across these sites (e.g. no LiDAR derived), so weights "
                "cannot be optimised; supply LiDAR and a spread of sites to calibrate."
            )
        else:
            report["suggested_si_weights"] = optimize_weights(df.loc[mask], a, settings)
            report["optimize_note"] = (
                "Weights re-tuned to maximise Spearman(structural A, activity) on these sites. "
                "Risk of overfitting with few sites — validate on held-out hedgerows before adopting."
            )
    return report


def optimize_weights(df_with_activity, activity, base_settings: config.ScoreSettings) -> dict:
    """Nelder-Mead over the 7 SI weights to maximise Spearman(structural A, activity)."""
    import numpy as np
    from scipy.optimize import minimize

    keys = list(config.SI_KEYS)
    x0 = np.array([base_settings.si_weights.get(k, 1.0) for k in keys], dtype="float64")

    def neg_spearman(x: "np.ndarray") -> float:
        weights = {k: max(0.0, float(xi)) for k, xi in zip(keys, x)}
        if sum(weights.values()) <= 0:
            return 1.0
        scored = apply_scoring(df_with_activity, config.ScoreSettings(
            si_weights=weights, context_weights=base_settings.context_weights, alpha=base_settings.alpha))
        rho = _safe_spearman(scored["hsi_structural_A"].astype("float64").to_numpy(), activity)
        return -(rho if rho is not None else -1.0)

    res = minimize(neg_spearman, x0, method="Nelder-Mead", options={"maxiter": 300, "xatol": 1e-2, "fatol": 1e-3})
    raw = [max(0.0, float(v)) for v in res.x]
    total = sum(raw) or 1.0
    # Rescale so the mean weight stays ~1.0 (comparable to the equal-weight default).
    scaled = [round(v * len(keys) / total, 3) for v in raw]
    return {k: w for k, w in zip(keys, scaled)}


def _round(v):
    return None if v is None or (isinstance(v, float) and v != v) else round(float(v), 4)


def _interpret(rho) -> str:
    if rho is None or rho != rho:
        return "Could not compute correlation."
    r = abs(rho)
    strength = "strong" if r >= 0.6 else "moderate" if r >= 0.4 else "weak"
    sign = "positive" if rho >= 0 else "negative"
    return f"{strength} {sign} relationship (Spearman {rho:.2f}) between HSI priority and observed activity."
