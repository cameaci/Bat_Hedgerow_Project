"""Explainability — answer "how much did each metric drive this hedgerow's priority?"

The final priority is ``alpha*A + (1-alpha)*B`` where A is the weighted mean of the
normalised SI scores and B the weighted mean of the context sub-indices. That makes each
factor's *additive* share of the priority exact:

    SI_i  contribution  = alpha     * (w_i * norm_i) / sum(w over present SIs)
    ctx_k contribution  = (1-alpha) * (v_k * val_k)  / sum(v over present ctx)

and the contributions sum to the priority. ``sensitivity`` shows how much the ranking
depends on each weight by perturbing it and measuring the change in priority.
"""

from __future__ import annotations

from . import config
from .score import _first_number, _is_missing, apply_scoring


def explain_hedgerow(scored_df, hf_uid, *, settings: config.ScoreSettings | None = None) -> dict:
    """Return a per-factor additive contribution breakdown for one hedgerow."""
    settings = settings or config.ScoreSettings()
    rows = scored_df[scored_df["hf_uid"] == hf_uid]
    if rows.empty:
        raise KeyError(f"hf_uid '{hf_uid}' not found.")
    row = rows.iloc[0]
    alpha = float(settings.alpha)

    si_terms = _terms(row, config.SI_KEYS, "hsi_{key}_norm", settings.si_weights, config.SI_LABELS)
    ctx_terms = _terms(row, config.CONTEXT_KEYS, "{key}", settings.context_weights, config.CONTEXT_LABELS)

    si_denom = sum(w for _, _, w, n in si_terms)
    ctx_denom = sum(w for _, _, w, n in ctx_terms)

    contributions = []
    for key, label, w, n in si_terms:
        share = alpha * (w * n) / si_denom if si_denom > 0 else 0.0
        contributions.append({"factor": label, "group": "structure", "value": round(n, 4),
                              "weight": w, "contribution": round(share, 4)})
    for key, label, w, n in ctx_terms:
        share = (1.0 - alpha) * (w * n) / ctx_denom if ctx_denom > 0 else 0.0
        contributions.append({"factor": label, "group": "context", "value": round(n, 4),
                              "weight": w, "contribution": round(share, 4)})

    contributions.sort(key=lambda d: d["contribution"], reverse=True)
    return {
        "hf_uid": hf_uid,
        "priority": _first_number(row, ("hsi_priority",)),
        "structural_A": _first_number(row, ("hsi_structural_A",)),
        "context_B": _first_number(row, ("hsi_context_B",)),
        "wsp_category": row.get("hsi_wsp_category"),
        "alpha": alpha,
        "contributions": contributions,
    }


def _terms(row, keys, template, weights, labels):
    out = []
    for key in keys:
        col = template.format(key=key)
        if col not in row.index:
            continue
        val = row.get(col)
        if _is_missing(val):
            continue
        w = float(weights.get(key, 0.0))
        if w <= 0:
            continue
        out.append((key, labels.get(key, key), w, float(val)))
    return out


def sensitivity(scored_df, *, settings: config.ScoreSettings | None = None, delta: float = 0.5) -> list[dict]:
    """How much does each weight move the priorities? Returns tornado data (mean |Δpriority|)."""
    settings = settings or config.ScoreSettings()
    import numpy as np

    base = apply_scoring(scored_df, settings)["hsi_priority"].astype(float)

    def impact(new_settings) -> float:
        p = apply_scoring(scored_df, new_settings)["hsi_priority"].astype(float)
        diff = (p - base).abs()
        return float(np.nanmean(diff.values)) if len(diff) else 0.0

    results: list[dict] = []
    for key in config.SI_KEYS:
        w = dict(settings.si_weights)
        lo, hi = dict(w), dict(w)
        lo[key] = max(0.0, w[key] - delta)
        hi[key] = w[key] + delta
        mag = max(
            impact(config.ScoreSettings(si_weights=lo, context_weights=settings.context_weights, alpha=settings.alpha)),
            impact(config.ScoreSettings(si_weights=hi, context_weights=settings.context_weights, alpha=settings.alpha)),
        )
        results.append({"factor": config.SI_LABELS[key], "group": "structure", "impact": round(mag, 4)})
    for key in config.CONTEXT_KEYS:
        v = dict(settings.context_weights)
        lo, hi = dict(v), dict(v)
        lo[key] = max(0.0, v[key] - delta)
        hi[key] = v[key] + delta
        mag = max(
            impact(config.ScoreSettings(si_weights=settings.si_weights, context_weights=lo, alpha=settings.alpha)),
            impact(config.ScoreSettings(si_weights=settings.si_weights, context_weights=hi, alpha=settings.alpha)),
        )
        results.append({"factor": config.CONTEXT_LABELS[key], "group": "context", "impact": round(mag, 4)})

    results.sort(key=lambda d: d["impact"], reverse=True)
    return results
