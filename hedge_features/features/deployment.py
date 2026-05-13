from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from zoneinfo import ZoneInfo

from ..deps import require_astral, require_geopandas
from ..exceptions import OptionalDependencyError


@dataclass(slots=True)
class DeploymentRowContext:
    status: str
    start_local: datetime | None
    end_local: datetime | None
    centroid_lat: float | None
    centroid_lon: float | None
    night_intervals_local: list[tuple[datetime, datetime]]


@dataclass(slots=True)
class DeploymentContext:
    timezone_name: str
    min_night_overlap_minutes: int
    rows: dict[int, DeploymentRowContext]


@dataclass(slots=True)
class DeploymentFeatureResult:
    gdf: Any
    context: DeploymentContext | None
    notes: list[str]


def add_deployment_nights_metrics(
    hedges_gdf,
    *,
    start_column: str | None,
    end_column: str | None,
    timezone_name: str = "Europe/London",
    min_night_overlap_minutes: int = 30,
    write_local_columns: bool = False,
):
    gdf = hedges_gdf.copy()
    notes: list[str] = []

    base_cols = [
        "deploy_temporal_flag",
        "deploy_parse_status",
        "deploy_status",
        "deploy_nights_count",
        "deploy_total_night_hours",
    ]
    if write_local_columns:
        base_cols.extend(["deploy_start_dt_local", "deploy_end_dt_local"])
    for col in base_cols:
        gdf[col] = None

    if not start_column or not end_column:
        gdf["deploy_temporal_flag"] = 0
        gdf["deploy_parse_status"] = "missing_column_mapping"
        gdf["deploy_status"] = "missing_column_mapping"
        gdf["deploy_nights_count"] = 0
        gdf["deploy_total_night_hours"] = 0.0
        notes.append("Deployment temporal features skipped: deployment start/end column mapping was not provided.")
        return DeploymentFeatureResult(gdf=gdf, context=None, notes=notes)

    if start_column not in gdf.columns or end_column not in gdf.columns:
        gdf["deploy_temporal_flag"] = 0
        gdf["deploy_parse_status"] = "missing_columns"
        gdf["deploy_status"] = "missing_columns"
        gdf["deploy_nights_count"] = 0
        gdf["deploy_total_night_hours"] = 0.0
        notes.append(
            f"Deployment temporal features skipped: missing columns start='{start_column}' or end='{end_column}'."
        )
        return DeploymentFeatureResult(gdf=gdf, context=None, notes=notes)

    try:
        require_astral()
        from astral import Observer
        from astral.sun import sun
    except OptionalDependencyError:
        gdf["deploy_temporal_flag"] = 0
        gdf["deploy_parse_status"] = "astral_missing"
        gdf["deploy_status"] = "astral_missing"
        gdf["deploy_nights_count"] = 0
        gdf["deploy_total_night_hours"] = 0.0
        notes.append("Astral is not installed; deployment night windows could not be computed.")
        return DeploymentFeatureResult(gdf=gdf, context=None, notes=notes)

    import pandas as pd

    tz = ZoneInfo(timezone_name)
    gpd = require_geopandas()
    centroids = gdf.geometry.centroid
    centroids_wgs84 = gpd.GeoSeries(centroids, crs=gdf.crs).to_crs("EPSG:4326")

    sun_cache: dict[tuple[float, float, date, str], datetime] = {}
    row_ctxs: dict[int, DeploymentRowContext] = {}

    for row_pos, (idx, row) in enumerate(gdf.iterrows()):
        start_ts = _parse_dt_local(row.get(start_column), tz)
        end_ts = _parse_dt_local(row.get(end_column), tz)
        if start_ts is None or end_ts is None:
            status = "parse_error"
            _set_deploy_row(gdf, row_pos, status=status, flag=0, count=0, total_hours=0.0)
            row_ctxs[row_pos] = DeploymentRowContext(
                status=status,
                start_local=start_ts,
                end_local=end_ts,
                centroid_lat=None,
                centroid_lon=None,
                night_intervals_local=[],
            )
            continue
        if end_ts <= start_ts:
            status = "invalid_interval"
            _set_deploy_row(gdf, row_pos, status=status, flag=0, count=0, total_hours=0.0)
            row_ctxs[row_pos] = DeploymentRowContext(
                status=status,
                start_local=start_ts,
                end_local=end_ts,
                centroid_lat=None,
                centroid_lon=None,
                night_intervals_local=[],
            )
            continue

        pt = centroids_wgs84.iloc[row_pos]
        if pt is None or pt.is_empty:
            status = "missing_geometry"
            _set_deploy_row(gdf, row_pos, status=status, flag=0, count=0, total_hours=0.0)
            row_ctxs[row_pos] = DeploymentRowContext(
                status=status,
                start_local=start_ts,
                end_local=end_ts,
                centroid_lat=None,
                centroid_lon=None,
                night_intervals_local=[],
            )
            continue

        lat = float(pt.y)
        lon = float(pt.x)
        observer = Observer(latitude=lat, longitude=lon)
        night_intervals = _deployment_night_intervals(
            observer=observer,
            start_local=start_ts,
            end_local=end_ts,
            tz=tz,
            min_overlap_minutes=min_night_overlap_minutes,
            sun_fn=sun,
            sun_cache=sun_cache,
        )
        if not night_intervals:
            status = "no_night_overlap"
            _set_deploy_row(gdf, row_pos, status=status, flag=0, count=0, total_hours=0.0)
        else:
            total_hours = sum((b - a).total_seconds() for a, b in night_intervals) / 3600.0
            status = "ok"
            _set_deploy_row(
                gdf,
                row_pos,
                status=status,
                flag=1,
                count=len(night_intervals),
                total_hours=float(total_hours),
            )
        if write_local_columns:
            gdf.iat[row_pos, gdf.columns.get_loc("deploy_start_dt_local")] = start_ts.isoformat()
            gdf.iat[row_pos, gdf.columns.get_loc("deploy_end_dt_local")] = end_ts.isoformat()

        row_ctxs[row_pos] = DeploymentRowContext(
            status=status,
            start_local=start_ts,
            end_local=end_ts,
            centroid_lat=lat,
            centroid_lon=lon,
            night_intervals_local=night_intervals,
        )

    notes.append(
        f"Deployment nights computed from '{start_column}'/'{end_column}' using timezone {timezone_name}."
    )
    return DeploymentFeatureResult(
        gdf=gdf,
        context=DeploymentContext(
            timezone_name=timezone_name,
            min_night_overlap_minutes=int(min_night_overlap_minutes),
            rows=row_ctxs,
        ),
        notes=notes,
    )


def _set_deploy_row(gdf, row_pos: int, *, status: str, flag: int, count: int, total_hours: float) -> None:
    gdf.iat[row_pos, gdf.columns.get_loc("deploy_temporal_flag")] = int(flag)
    gdf.iat[row_pos, gdf.columns.get_loc("deploy_parse_status")] = status
    gdf.iat[row_pos, gdf.columns.get_loc("deploy_status")] = status
    gdf.iat[row_pos, gdf.columns.get_loc("deploy_nights_count")] = int(count)
    gdf.iat[row_pos, gdf.columns.get_loc("deploy_total_night_hours")] = float(total_hours)


def _parse_dt_local(value: Any, tz: ZoneInfo) -> datetime | None:
    import pandas as pd

    if value is None:
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if ts is pd.NaT or pd.isna(ts):
        return None
    try:
        if ts.tzinfo is None:
            ts = ts.tz_localize(str(tz), ambiguous="NaT", nonexistent="shift_forward")
            if ts is pd.NaT or pd.isna(ts):
                return None
        else:
            ts = ts.tz_convert(str(tz))
    except Exception:
        return None
    dt_obj = ts.to_pydatetime()
    return dt_obj


def _deployment_night_intervals(
    *,
    observer,
    start_local: datetime,
    end_local: datetime,
    tz: ZoneInfo,
    min_overlap_minutes: int,
    sun_fn,
    sun_cache: dict[tuple[float, float, date, str], datetime],
) -> list[tuple[datetime, datetime]]:
    out: list[tuple[datetime, datetime]] = []
    min_overlap = timedelta(minutes=max(1, int(min_overlap_minutes)))

    d = start_local.date() - timedelta(days=1)
    end_d = end_local.date()
    while d <= end_d:
        sunset_d = _sun_event(observer, d, "sunset", tz, sun_fn, sun_cache)
        sunrise_next = _sun_event(observer, d + timedelta(days=1), "sunrise", tz, sun_fn, sun_cache)
        if sunset_d is None or sunrise_next is None or sunrise_next <= sunset_d:
            d += timedelta(days=1)
            continue
        ov_start = max(start_local, sunset_d)
        ov_end = min(end_local, sunrise_next)
        if ov_end > ov_start and (ov_end - ov_start) >= min_overlap:
            out.append((ov_start, ov_end))
        d += timedelta(days=1)
    return out


def _sun_event(observer, d: date, key: str, tz: ZoneInfo, sun_fn, cache) -> datetime | None:
    lat_r = round(float(observer.latitude), 4)
    lon_r = round(float(observer.longitude), 4)
    ck = (lat_r, lon_r, d, key)
    if ck in cache:
        return cache[ck]
    try:
        vals = sun_fn(observer, date=d, tzinfo=tz)
        ev = vals.get(key)
    except Exception:
        ev = None
    cache[ck] = ev
    return ev

