from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .schema import AcousticImportSettings


_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "datetime": ("datetime", "date_time", "timestamp", "start_time", "time", "event_time"),
    "species": ("species", "species_name", "scientific_name", "common_name", "class", "predicted_class", "label"),
    "guild": ("guild", "bat_guild", "species_group", "group"),
    "confidence": ("confidence", "probability", "prob", "score", "max_prob", "class_prob"),
    "activity": ("activity", "activity_count", "call_count", "calls", "detections", "count"),
    "latitude": ("latitude", "lat", "y"),
    "longitude": ("longitude", "lon", "lng", "long", "x"),
}

_BATDETECT2_ALIASES: dict[str, tuple[str, ...]] = {
    "datetime": ("datetime", "timestamp", "time_exp", "start_time"),
    "species": ("species", "class", "predicted_class", "label"),
    "confidence": ("prob", "score", "confidence", "max_prob"),
    "activity": ("call_count", "calls", "detections", "count"),
    "latitude": ("latitude", "lat"),
    "longitude": ("longitude", "lon", "lng"),
}


def import_acoustic_evidence(hedges_gdf, detections_df, *, settings: AcousticImportSettings) -> tuple[Any, dict[str, Any]]:
    """Attach aggregated bat acoustic detections to hedgerow segments.

    Detections can be linked directly via a hedgerow id column or spatially via latitude/longitude
    columns. The function is intentionally adapter-oriented so detector/model outputs such as
    BatDetect2, BattyBirdNET, or project-standard CSVs can be normalised before aggregation.
    """
    import pandas as pd

    if settings.hedge_id_column not in hedges_gdf.columns:
        raise ValueError(f"Hedge id column '{settings.hedge_id_column}' was not found in hedgerow input.")
    if hedges_gdf.crs is None:
        raise ValueError("Hedgerow input must have a CRS before acoustic spatial matching.")

    detections = _normalise_detection_columns(detections_df.copy(), settings=settings)
    initial_records = int(len(detections))
    notes: list[str] = []

    if settings.min_confidence is not None and "acoustic_confidence" in detections.columns:
        conf = pd.to_numeric(detections["acoustic_confidence"], errors="coerce")
        detections = detections.loc[conf.fillna(-1.0) >= float(settings.min_confidence)].copy()
    filtered_records = int(len(detections))

    if settings.detection_hedge_id_column:
        linked, link_notes = _link_by_hedge_id(hedges_gdf, detections, settings=settings)
    else:
        linked, link_notes = _link_by_nearest_geometry(hedges_gdf, detections, settings=settings)
    notes.extend(link_notes)

    matched_records = int(len(linked))
    summary_df = _aggregate_linked_detections(linked, hedge_id_column=settings.hedge_id_column)

    out = hedges_gdf.copy()
    out[settings.hedge_id_column] = out[settings.hedge_id_column].astype("string")
    out = out.merge(summary_df, how="left", on=settings.hedge_id_column)
    out = _fill_acoustic_defaults(out)

    summary = {
        "source_format": settings.source_format,
        "settings": asdict(settings),
        "input_detection_records": initial_records,
        "records_after_confidence_filter": filtered_records,
        "matched_detection_records": matched_records,
        "unmatched_detection_records": max(filtered_records - matched_records, 0),
        "hedgerow_count": int(len(hedges_gdf)),
        "hedgerows_with_acoustic_evidence": int((out["acoustic_detection_count"].fillna(0).astype(float) > 0).sum()),
        "notes": notes,
    }
    return out, summary


def _normalise_detection_columns(df, *, settings: AcousticImportSettings):
    aliases = dict(_COLUMN_ALIASES)
    if str(settings.source_format).lower() == "batdetect2":
        aliases.update(_BATDETECT2_ALIASES)

    explicit = {
        "datetime": settings.datetime_column,
        "species": settings.species_column,
        "guild": settings.guild_column,
        "confidence": settings.confidence_column,
        "activity": settings.activity_column,
        "latitude": settings.latitude_column,
        "longitude": settings.longitude_column,
    }
    for canonical, explicit_name in explicit.items():
        source_col = explicit_name or _find_column(df.columns, aliases.get(canonical, ()))
        if source_col and source_col in df.columns:
            df[f"acoustic_{canonical}"] = df[source_col]

    if settings.detection_hedge_id_column:
        if settings.detection_hedge_id_column not in df.columns:
            raise ValueError(
                f"Detection hedgerow id column '{settings.detection_hedge_id_column}' was not found."
            )
        df["acoustic_hedge_id"] = df[settings.detection_hedge_id_column].astype("string")

    return df


def _find_column(columns, aliases: tuple[str, ...]) -> str | None:
    lower_to_original = {str(c).strip().lower(): str(c) for c in columns}
    for alias in aliases:
        if alias.lower() in lower_to_original:
            return lower_to_original[alias.lower()]
    return None


def _link_by_hedge_id(hedges_gdf, detections, *, settings: AcousticImportSettings):
    valid_ids = set(hedges_gdf[settings.hedge_id_column].astype("string"))
    linked = detections.copy()
    linked[settings.hedge_id_column] = linked["acoustic_hedge_id"].astype("string")
    before = int(len(linked))
    linked = linked.loc[linked[settings.hedge_id_column].isin(valid_ids)].copy()
    linked["acoustic_match_distance_m"] = 0.0
    notes = [f"Linked acoustic detections by hedgerow id; dropped {before - len(linked)} unmatched record(s)."]
    return linked, notes


def _link_by_nearest_geometry(hedges_gdf, detections, *, settings: AcousticImportSettings):
    gpd = __import__("geopandas")
    from shapely.geometry import Point

    lat_col = "acoustic_latitude"
    lon_col = "acoustic_longitude"
    if lat_col not in detections.columns or lon_col not in detections.columns:
        raise ValueError(
            "Acoustic spatial matching requires latitude/longitude columns or --detection-hedge-id-col."
        )

    import pandas as pd

    lat = pd.to_numeric(detections[lat_col], errors="coerce")
    lon = pd.to_numeric(detections[lon_col], errors="coerce")
    valid = lat.notna() & lon.notna()
    records = detections.loc[valid].copy()
    points = [Point(float(x), float(y)) for x, y in zip(lon.loc[valid], lat.loc[valid])]
    points_gdf = gpd.GeoDataFrame(records, geometry=points, crs=settings.detections_crs)
    if str(points_gdf.crs) != str(hedges_gdf.crs):
        points_gdf = points_gdf.to_crs(hedges_gdf.crs)

    hedges_for_join = hedges_gdf[[settings.hedge_id_column, hedges_gdf.geometry.name]].copy()
    max_distance = None if settings.max_distance_m is None else float(settings.max_distance_m)
    joined = gpd.sjoin_nearest(
        points_gdf,
        hedges_for_join,
        how="left",
        max_distance=max_distance,
        distance_col="acoustic_match_distance_m",
    )
    joined = joined.drop(columns=[c for c in ("index_right", "geometry") if c in joined.columns])
    linked = joined.loc[joined[settings.hedge_id_column].notna()].copy()
    linked[settings.hedge_id_column] = linked[settings.hedge_id_column].astype("string")
    dropped = int(len(detections) - len(linked))
    notes = [
        "Linked acoustic detections by nearest hedgerow geometry"
        + (f" within {max_distance:g} m" if max_distance is not None else "")
        + f"; dropped {dropped} unmatched/invalid-location record(s)."
    ]
    return linked, notes


def _aggregate_linked_detections(linked, *, hedge_id_column: str):
    import pandas as pd

    if linked.empty:
        return pd.DataFrame(
            columns=[
                hedge_id_column,
                "acoustic_detection_count",
                "acoustic_species_count",
                "acoustic_species_list",
                "acoustic_guild_count",
                "acoustic_guild_list",
                "acoustic_mean_confidence",
                "acoustic_max_confidence",
                "acoustic_activity_sum",
                "acoustic_first_detection",
                "acoustic_last_detection",
                "acoustic_mean_match_distance_m",
            ]
        )

    working = linked.copy()
    working[hedge_id_column] = working[hedge_id_column].astype("string")
    if "acoustic_confidence" in working.columns:
        working["_confidence_num"] = pd.to_numeric(working["acoustic_confidence"], errors="coerce")
    else:
        working["_confidence_num"] = pd.NA
    if "acoustic_activity" in working.columns:
        working["_activity_num"] = pd.to_numeric(working["acoustic_activity"], errors="coerce").fillna(1.0)
    else:
        working["_activity_num"] = 1.0
    if "acoustic_datetime" in working.columns:
        working["_datetime"] = pd.to_datetime(working["acoustic_datetime"], errors="coerce", utc=True)
    else:
        working["_datetime"] = pd.NaT
    if "acoustic_match_distance_m" in working.columns:
        working["_distance_num"] = pd.to_numeric(working["acoustic_match_distance_m"], errors="coerce")
    else:
        working["_distance_num"] = pd.NA

    rows: list[dict[str, Any]] = []
    for hedge_id, group in working.groupby(hedge_id_column, dropna=False, sort=True):
        species = _unique_nonempty(group.get("acoustic_species"))
        guilds = _unique_nonempty(group.get("acoustic_guild"))
        first = group["_datetime"].min()
        last = group["_datetime"].max()
        rows.append(
            {
                hedge_id_column: str(hedge_id),
                "acoustic_detection_count": int(len(group)),
                "acoustic_species_count": int(len(species)),
                "acoustic_species_list": "|".join(species),
                "acoustic_guild_count": int(len(guilds)),
                "acoustic_guild_list": "|".join(guilds),
                "acoustic_mean_confidence": _round_or_none(group["_confidence_num"].mean()),
                "acoustic_max_confidence": _round_or_none(group["_confidence_num"].max()),
                "acoustic_activity_sum": _round_or_none(group["_activity_num"].sum()),
                "acoustic_first_detection": None if pd.isna(first) else first.isoformat(),
                "acoustic_last_detection": None if pd.isna(last) else last.isoformat(),
                "acoustic_mean_match_distance_m": _round_or_none(group["_distance_num"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _unique_nonempty(series) -> list[str]:
    if series is None:
        return []
    values = []
    for value in series.dropna().astype(str):
        value = value.strip()
        if value and value.lower() not in {"nan", "none", "unknown"}:
            values.append(value)
    return sorted(set(values))


def _round_or_none(value, ndigits: int = 6):
    import pandas as pd

    if pd.isna(value):
        return None
    return round(float(value), ndigits)


def _fill_acoustic_defaults(gdf):
    defaults = {
        "acoustic_detection_count": 0,
        "acoustic_species_count": 0,
        "acoustic_species_list": "",
        "acoustic_guild_count": 0,
        "acoustic_guild_list": "",
        "acoustic_mean_confidence": None,
        "acoustic_max_confidence": None,
        "acoustic_activity_sum": 0.0,
        "acoustic_first_detection": None,
        "acoustic_last_detection": None,
        "acoustic_mean_match_distance_m": None,
    }
    for col, default in defaults.items():
        if col not in gdf.columns:
            gdf[col] = default
        else:
            if default in {0, 0.0, ""}:
                gdf[col] = gdf[col].fillna(default)
    for col in ("acoustic_detection_count", "acoustic_species_count", "acoustic_guild_count"):
        gdf[col] = gdf[col].astype(int)
    return gdf
