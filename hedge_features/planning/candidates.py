from __future__ import annotations

from typing import Any

from ..deps import require_geopandas
from ..utils import sha1_text


def build_candidate_points(
    hedges_gdf,
    *,
    settings,
    hedge_id_column: str = "hf_uid",
):
    gdf = hedges_gdf.copy()
    if hedge_id_column not in gdf.columns:
        raise ValueError(f"Hedge id column '{hedge_id_column}' is required for planning candidates.")

    records: list[dict[str, Any]] = []
    geometry_name = gdf.geometry.name

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        length_m = float(geom.length)
        if length_m <= 0:
            continue

        hedge_id = str(row.get(hedge_id_column))
        candidate_score = _candidate_score(row, score_column=settings.score_column)
        chainages = _candidate_chainages(
            length_m,
            spacing_m=float(settings.candidate_spacing_m),
            endpoint_offset_m=float(settings.endpoint_offset_m),
        )
        attrs = {k: v for k, v in row.items() if k != geometry_name}
        for chainage_m in chainages:
            point = geom.interpolate(chainage_m)
            candidate_id = _candidate_id(
                hedge_id=hedge_id,
                chainage_m=chainage_m,
                geom=geom,
            )
            record = dict(attrs)
            record.update(
                {
                    "candidate_id": candidate_id,
                    "source_hf_uid": hedge_id,
                    "candidate_chainage_m": round(float(chainage_m), 3),
                    "source_length_m": round(length_m, 3),
                    "candidate_score": candidate_score,
                    "selected_flag": 0,
                    "selection_rank": None,
                    "selection_phase": None,
                    "planning_status": "candidate",
                    "geometry": point,
                }
            )
            records.append(record)

    gpd = require_geopandas()
    if not records:
        return gpd.GeoDataFrame(
            {"candidate_id": [], "source_hf_uid": [], "candidate_chainage_m": [], "source_length_m": [], "candidate_score": []},
            geometry=gpd.GeoSeries([], crs=gdf.crs),
            crs=gdf.crs,
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=gdf.crs)


def _candidate_chainages(length_m: float, *, spacing_m: float, endpoint_offset_m: float) -> list[float]:
    spacing = max(float(spacing_m), 1.0)
    offset = max(float(endpoint_offset_m), 0.0)
    if length_m <= (offset * 2.0) or length_m <= spacing:
        return [length_m / 2.0]

    start = min(offset, length_m / 2.0)
    end = max(length_m - offset, start)
    if end <= start:
        return [length_m / 2.0]

    out: list[float] = []
    pos = start
    while pos <= end + 1e-9:
        out.append(float(pos))
        pos += spacing
    if out:
        last = out[-1]
        if (end - last) > (spacing * 0.5):
            out.append(float(end))
    return [round(v, 3) for v in out]


def _candidate_score(row, *, score_column: str | None) -> float:
    if not score_column:
        return 1.0
    value = row.get(score_column)
    try:
        score = float(value)
    except Exception:
        return 0.0
    if score != score:
        return 0.0
    return score


def _candidate_id(*, hedge_id: str, chainage_m: float, geom) -> str:
    payload = f"{hedge_id}|{chainage_m:.3f}|{geom.wkb_hex[:48]}"
    return f"cand_{sha1_text(payload, length=14)}"
