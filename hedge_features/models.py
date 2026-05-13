from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils import sha1_text, stable_json_dumps


@dataclass(slots=True)
class DatasetRef:
    name: str
    path: str | None = None
    license: str | None = None
    attribution: str | None = None
    version: str | None = None
    mode: str = "local_cache"
    optional: bool = True
    dataset_class: str | None = None
    snapshot_id: str | None = None
    coverage_scope: str | None = None
    source_resolution: str | None = None
    source_type: str | None = None
    proxy_used: bool = False
    fallback_used: bool = False
    outside_coverage: bool = False
    guidance_regime_version: str | None = None
    authenticated: bool = False
    status: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunOptions:
    input_path: Path
    output_path: Path
    profile_name: str
    profile_path: Path | None = None
    working_crs: str = "EPSG:27700"
    input_crs: str | None = None
    export_crs: str | None = None
    cache_dir: Path | None = None
    auto_fetch: bool = True
    credentials: dict[str, str] = field(default_factory=dict)
    drop_all_null_feature_columns: bool = True
    deployment_start_column: str | None = None
    deployment_end_column: str | None = None
    deployment_timezone: str = "Europe/London"
    weather_backend: str = "open_meteo"
    min_night_overlap_minutes: int = 30
    enable_temporal_features: bool = True
    enable_roost_microhabitat_proxies: bool = True
    dataset_overrides: dict[str, str] = field(default_factory=dict)
    batch_size: int = 1000
    write_shapefile_zip: bool = False
    write_csv: bool = False
    deterministic_output: bool = True
    frozen_datasets_only: bool = False


@dataclass(slots=True)
class RunMetadata:
    tool_name: str
    tool_version: str
    run_id: str
    run_timestamp_utc: str | None
    input_path: str
    output_path: str
    working_crs: str
    export_crs: str | None
    profile_name: str
    profile_path: str | None
    datasets: list[dict[str, Any]]
    guidance_regime_version: str | None = None
    deterministic_output: bool = True
    frozen_datasets_only: bool = False
    notes: list[str] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        *,
        tool_version: str,
        input_path: Path,
        output_path: Path,
        working_crs: str,
        export_crs: str | None,
        profile_name: str,
        profile_path: Path | None,
        datasets: list[dict[str, Any]],
        guidance_regime_version: str | None = None,
        deterministic_output: bool = True,
        frozen_datasets_only: bool = False,
        notes: list[str] | None = None,
    ) -> "RunMetadata":
        payload = stable_json_dumps(
            {
                "tool_name": "hedge-features",
                "tool_version": tool_version,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "working_crs": working_crs,
                "export_crs": export_crs,
                "profile_name": profile_name,
                "profile_path": str(profile_path) if profile_path else None,
                "datasets": datasets,
                "guidance_regime_version": guidance_regime_version,
                "deterministic_output": bool(deterministic_output),
                "frozen_datasets_only": bool(frozen_datasets_only),
            }
        )
        return cls(
            tool_name="hedge-features",
            tool_version=tool_version,
            run_id=f"run_{sha1_text(payload, length=12)}",
            run_timestamp_utc=None if deterministic_output else datetime.now(timezone.utc).isoformat(),
            input_path=str(input_path),
            output_path=str(output_path),
            working_crs=working_crs,
            export_crs=export_crs,
            profile_name=profile_name,
            profile_path=str(profile_path) if profile_path else None,
            datasets=datasets,
            guidance_regime_version=guidance_regime_version,
            deterministic_output=bool(deterministic_output),
            frozen_datasets_only=bool(frozen_datasets_only),
            notes=notes or [],
        )
