from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..deps import require_astral
from ..exceptions import OptionalDependencyError
from .deployment import DeploymentContext
from .weather import WeatherContext


SYNODIC_MONTH_DAYS = 29.53058867


@dataclass(slots=True)
class MoonFeatureResult:
    gdf: Any
    notes: list[str]


def add_moonlight_nightly_aggregate(
    hedges_gdf,
    *,
    deployment_context: DeploymentContext | None,
    weather_context: WeatherContext | None,
    use_cloud_adjustment: bool = True,
):
    gdf = hedges_gdf.copy()
    notes: list[str] = []
    cols = [
        "moon_phase_mean_frac",
        "moon_illum_mean_pct",
        "moon_altitude_mean_deg",
        "moon_visible_hours_est",
        "moon_cloud_mean_pct",
        "moonlight_proxy_mean",
        "moonlight_proxy_p90",
        "moonlight_nights_sampled",
        "moon_status",
    ]
    for col in cols:
        gdf[col] = None

    if not deployment_context:
        gdf["moon_status"] = "deployment_missing"
        notes.append("Moonlight proxy skipped: deployment context missing.")
        return MoonFeatureResult(gdf=gdf, notes=notes)

    try:
        require_astral()
        from astral import Observer
        from astral.moon import elevation as moon_elevation
        from astral.moon import phase as moon_phase_days
    except OptionalDependencyError:
        gdf["moon_status"] = "astral_missing"
        notes.append("Astral is not installed; moonlight proxy could not be computed.")
        return MoonFeatureResult(gdf=gdf, notes=notes)

    import numpy as np
    import pandas as pd

    for row_pos in range(len(gdf)):
        row_ctx = deployment_context.rows.get(row_pos)
        if row_ctx is None or row_ctx.status != "ok":
            gdf.iat[row_pos, gdf.columns.get_loc("moon_status")] = "no_deployment"
            continue
        sampled = weather_context.row_night_hourly.get(row_pos) if weather_context else None
        if sampled is None or getattr(sampled, "empty", True):
            gdf.iat[row_pos, gdf.columns.get_loc("moon_status")] = "no_weather_samples"
            continue
        if row_ctx.centroid_lat is None or row_ctx.centroid_lon is None:
            gdf.iat[row_pos, gdf.columns.get_loc("moon_status")] = "missing_geometry"
            continue

        observer = Observer(latitude=row_ctx.centroid_lat, longitude=row_ctx.centroid_lon)
        times = pd.to_datetime(sampled["time_local"], errors="coerce").dropna()
        if times.empty:
            gdf.iat[row_pos, gdf.columns.get_loc("moon_status")] = "no_samples"
            continue

        altitudes: list[float] = []
        phases_frac: list[float] = []
        illum_pct: list[float] = []
        clouds_pct: list[float] = []
        proxies: list[float] = []
        visible_hours = 0.0

        cloud_series = pd.to_numeric(sampled.get("cloud_cover"), errors="coerce") if "cloud_cover" in sampled.columns else None
        for i, ts in enumerate(times):
            dt_obj = ts.to_pydatetime()
            alt = float(moon_elevation(observer, dt_obj))
            phase_days = float(moon_phase_days(dt_obj.date()))
            phase_frac = (phase_days % SYNODIC_MONTH_DAYS) / SYNODIC_MONTH_DAYS
            illum_frac = 0.5 * (1.0 - math.cos(2.0 * math.pi * phase_frac))
            illum = float(max(0.0, min(1.0, illum_frac)) * 100.0)
            cloud_pct = float(cloud_series.iloc[i]) if cloud_series is not None and i < len(cloud_series) and pd.notna(cloud_series.iloc[i]) else 0.0
            cloud_frac = max(0.0, min(1.0, cloud_pct / 100.0))
            if alt > 0:
                visible_factor = max(0.0, math.sin(math.radians(alt)))
                visible_hours += 1.0
            else:
                visible_factor = 0.0
            proxy = float((illum / 100.0) * visible_factor * (1.0 - cloud_frac if use_cloud_adjustment else 1.0))

            altitudes.append(alt)
            phases_frac.append(float(phase_frac))
            illum_pct.append(illum)
            clouds_pct.append(cloud_pct)
            proxies.append(proxy)

        if not proxies:
            gdf.iat[row_pos, gdf.columns.get_loc("moon_status")] = "no_samples"
            continue

        gdf.iat[row_pos, gdf.columns.get_loc("moon_phase_mean_frac")] = float(np.mean(phases_frac))
        gdf.iat[row_pos, gdf.columns.get_loc("moon_illum_mean_pct")] = float(np.mean(illum_pct))
        gdf.iat[row_pos, gdf.columns.get_loc("moon_altitude_mean_deg")] = float(np.mean(altitudes))
        gdf.iat[row_pos, gdf.columns.get_loc("moon_visible_hours_est")] = float(visible_hours)
        gdf.iat[row_pos, gdf.columns.get_loc("moon_cloud_mean_pct")] = float(np.mean(clouds_pct))
        gdf.iat[row_pos, gdf.columns.get_loc("moonlight_proxy_mean")] = float(np.mean(proxies))
        gdf.iat[row_pos, gdf.columns.get_loc("moonlight_proxy_p90")] = float(np.percentile(proxies, 90))
        gdf.iat[row_pos, gdf.columns.get_loc("moonlight_nights_sampled")] = int(
            _count_intervals_with_samples(sampled, row_ctx.night_intervals_local)
        )
        gdf.iat[row_pos, gdf.columns.get_loc("moon_status")] = "ok"

    notes.append(
        "Moonlight proxy computed from Astral moon geometry and Open-Meteo cloud cover (weather-adjusted proxy, not direct illuminance)."
    )
    return MoonFeatureResult(gdf=gdf, notes=notes)


def _count_intervals_with_samples(sampled, intervals_local) -> int:
    if sampled is None or getattr(sampled, "empty", True) or "time_local" not in sampled.columns:
        return 0
    count = 0
    for start_local, end_local in intervals_local:
        if ((sampled["time_local"] >= start_local) & (sampled["time_local"] < end_local)).any():
            count += 1
    return count
