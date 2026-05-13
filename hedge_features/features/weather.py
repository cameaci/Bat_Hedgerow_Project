from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from ..utils import sha1_text
from .deployment import DeploymentContext


OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


@dataclass(slots=True)
class WeatherContext:
    timezone_name: str
    # row position -> sampled hourly dataframe (night-only) including cloud_cover/time_local
    row_night_hourly: dict[int, Any] = field(default_factory=dict)
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WeatherFeatureResult:
    gdf: Any
    context: WeatherContext | None
    notes: list[str]


class OpenMeteoArchiveClient:
    def __init__(self, *, cache_dir: str | Path | None, timezone_name: str = "Europe/London", coord_precision: int = 4):
        self.timezone_name = timezone_name
        self.coord_precision = int(coord_precision)
        self.cache_root = (Path(cache_dir) if cache_dir else Path.cwd() / ".hedge_features_cache") / "open_meteo_archive"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.request_count = 0
        self.cache_hits = 0

    def fetch_hourly(
        self,
        *,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
        hourly_variables: list[str],
    ):
        import pandas as pd

        lat_r = round(float(latitude), self.coord_precision)
        lon_r = round(float(longitude), self.coord_precision)
        vars_key = ",".join(sorted(hourly_variables))
        key = sha1_text(
            f"{lat_r}|{lon_r}|{start_date.isoformat()}|{end_date.isoformat()}|{vars_key}|{self.timezone_name}",
            length=24,
        )
        cache_path = self.cache_root / f"{key}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            self.cache_hits += 1
        else:
            query = {
                "latitude": f"{lat_r:.{self.coord_precision}f}",
                "longitude": f"{lon_r:.{self.coord_precision}f}",
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "hourly": ",".join(hourly_variables),
                "timezone": self.timezone_name,
            }
            url = f"{OPEN_METEO_ARCHIVE_URL}?{urllib.parse.urlencode(query)}"
            req = urllib.request.Request(url, headers={"User-Agent": "hedge-features/0.1"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.load(resp)
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            self.request_count += 1

        hourly = payload.get("hourly") or {}
        if not hourly or "time" not in hourly:
            return pd.DataFrame()
        df = pd.DataFrame(hourly)
        if df.empty:
            return df
        df["time_local"] = pd.to_datetime(df["time"], errors="coerce")
        # Open-Meteo returns local naive strings when timezone param is passed.
        df["time_local"] = df["time_local"].dt.tz_localize(
            self.timezone_name, ambiguous="NaT", nonexistent="shift_forward"
        )
        df = df.dropna(subset=["time_local"]).reset_index(drop=True)
        return df


def add_weather_nightly_aggregate(
    hedges_gdf,
    *,
    deployment_context: DeploymentContext | None,
    cache_dir: str | Path | None,
    timezone_name: str = "Europe/London",
    backend: str = "open_meteo",
):
    gdf = hedges_gdf.copy()
    notes: list[str] = []
    cols = [
        "depwx_night_temp_mean_c",
        "depwx_night_temp_min_c",
        "depwx_night_temp_max_c",
        "depwx_night_temp_p90_c",
        "depwx_wind_mean_ms",
        "depwx_wind_max_ms",
        "depwx_gust_max_ms",
        "depwx_rain_sum_mm",
        "depwx_rain_mean_mmph",
        "depwx_hours_sampled",
        "depwx_nights_with_weather",
        "depwx_status",
    ]
    for col in cols:
        gdf[col] = None

    if not deployment_context:
        gdf["depwx_status"] = "deployment_missing"
        notes.append("Weather nightly aggregation skipped: deployment context missing.")
        return WeatherFeatureResult(gdf=gdf, context=None, notes=notes)

    if backend != "open_meteo":
        gdf["depwx_status"] = "unsupported_backend"
        notes.append(f"Weather backend '{backend}' not implemented.")
        return WeatherFeatureResult(gdf=gdf, context=None, notes=notes)

    import numpy as np

    client = OpenMeteoArchiveClient(cache_dir=cache_dir, timezone_name=timezone_name)
    weather_ctx = WeatherContext(timezone_name=timezone_name)
    hourly_vars = [
        "temperature_2m",
        "precipitation",
        "wind_speed_10m",
        "wind_gusts_10m",
        "cloud_cover",
    ]

    for row_pos in range(len(gdf)):
        row_ctx = deployment_context.rows.get(row_pos)
        if row_ctx is None or row_ctx.status != "ok":
            gdf.iat[row_pos, gdf.columns.get_loc("depwx_status")] = "no_deployment"
            continue
        if not row_ctx.night_intervals_local or row_ctx.centroid_lat is None or row_ctx.centroid_lon is None:
            gdf.iat[row_pos, gdf.columns.get_loc("depwx_status")] = "no_night_overlap"
            continue

        start_d = (row_ctx.night_intervals_local[0][0] - timedelta(days=1)).date()
        end_d = (row_ctx.night_intervals_local[-1][1] + timedelta(days=1)).date()
        try:
            hourly_df = client.fetch_hourly(
                latitude=row_ctx.centroid_lat,
                longitude=row_ctx.centroid_lon,
                start_date=start_d,
                end_date=end_d,
                hourly_variables=hourly_vars,
            )
        except Exception as exc:
            gdf.iat[row_pos, gdf.columns.get_loc("depwx_status")] = "api_error"
            if not notes:
                notes.append(f"Open-Meteo request failed (first error): {exc}")
            continue
        if hourly_df.empty:
            gdf.iat[row_pos, gdf.columns.get_loc("depwx_status")] = "no_samples"
            continue

        sampled = _filter_to_night_intervals(hourly_df, row_ctx.night_intervals_local)
        if sampled.empty:
            gdf.iat[row_pos, gdf.columns.get_loc("depwx_status")] = "no_samples"
            continue

        weather_ctx.row_night_hourly[row_pos] = sampled.copy()
        nights_with_samples = _count_intervals_with_samples(sampled, row_ctx.night_intervals_local)
        _fill_depwx_metrics(gdf, row_pos, sampled, nights_with_samples=nights_with_samples)
        gdf.iat[row_pos, gdf.columns.get_loc("depwx_status")] = "ok"

    weather_ctx.provider_metadata = {
        "provider": "open_meteo_archive",
        "url": OPEN_METEO_ARCHIVE_URL,
        "timezone": timezone_name,
        "request_count": client.request_count,
        "cache_hits": client.cache_hits,
        "cache_dir": str(client.cache_root),
    }
    notes.append(
        f"Weather nightly features computed with Open-Meteo archive (requests={client.request_count}, cache_hits={client.cache_hits})."
    )
    return WeatherFeatureResult(gdf=gdf, context=weather_ctx, notes=notes)


def _filter_to_night_intervals(hourly_df, intervals_local: list[tuple[datetime, datetime]]):
    import pandas as pd

    if hourly_df.empty or not intervals_local:
        return hourly_df.iloc[0:0].copy()
    mask = pd.Series(False, index=hourly_df.index)
    for start_local, end_local in intervals_local:
        mask |= (hourly_df["time_local"] >= start_local) & (hourly_df["time_local"] < end_local)
    sampled = hourly_df.loc[mask].copy()
    return sampled.reset_index(drop=True)


def _fill_depwx_metrics(gdf, row_pos: int, sampled, *, nights_with_samples: int | None = None) -> None:
    import numpy as np
    import pandas as pd

    def num(col):
        if col not in sampled.columns:
            return np.array([], dtype=float)
        return pd.to_numeric(sampled[col], errors="coerce").dropna().to_numpy(dtype=float)

    temp = num("temperature_2m")
    wind = num("wind_speed_10m")
    gust = num("wind_gusts_10m")
    rain = num("precipitation")

    col_ix = gdf.columns.get_loc
    if temp.size:
        gdf.iat[row_pos, col_ix("depwx_night_temp_mean_c")] = float(np.mean(temp))
        gdf.iat[row_pos, col_ix("depwx_night_temp_min_c")] = float(np.min(temp))
        gdf.iat[row_pos, col_ix("depwx_night_temp_max_c")] = float(np.max(temp))
        gdf.iat[row_pos, col_ix("depwx_night_temp_p90_c")] = float(np.percentile(temp, 90))
    if wind.size:
        gdf.iat[row_pos, col_ix("depwx_wind_mean_ms")] = float(np.mean(wind))
        gdf.iat[row_pos, col_ix("depwx_wind_max_ms")] = float(np.max(wind))
    if gust.size:
        gdf.iat[row_pos, col_ix("depwx_gust_max_ms")] = float(np.max(gust))
    if rain.size:
        gdf.iat[row_pos, col_ix("depwx_rain_sum_mm")] = float(np.sum(rain))
        gdf.iat[row_pos, col_ix("depwx_rain_mean_mmph")] = float(np.mean(rain))

    gdf.iat[row_pos, col_ix("depwx_hours_sampled")] = int(len(sampled))
    if nights_with_samples is not None:
        gdf.iat[row_pos, col_ix("depwx_nights_with_weather")] = int(nights_with_samples)
    elif "time_local" in sampled.columns and not sampled.empty:
        gdf.iat[row_pos, col_ix("depwx_nights_with_weather")] = int(sampled["time_local"].dt.date.nunique())
    else:
        gdf.iat[row_pos, col_ix("depwx_nights_with_weather")] = 0


def _count_intervals_with_samples(sampled, intervals_local: list[tuple[datetime, datetime]]) -> int:
    if sampled.empty or "time_local" not in sampled.columns:
        return 0
    count = 0
    for start_local, end_local in intervals_local:
        if ((sampled["time_local"] >= start_local) & (sampled["time_local"] < end_local)).any():
            count += 1
    return count
