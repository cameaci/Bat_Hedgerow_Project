from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..exceptions import DatasetResolutionError
from ..models import DatasetRef


class DatasetRegistry:
    """Resolves dataset paths from profile config and CLI overrides."""

    def __init__(
        self,
        profile_datasets: dict[str, Any] | None,
        overrides: dict[str, str] | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.profile_datasets = profile_datasets or {}
        self.overrides = overrides or {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.runtime_paths: dict[str, str] = {}
        self.runtime_fields: dict[str, dict[str, Any]] = {}

    def resolve(self, name: str) -> DatasetRef:
        cfg = self.profile_datasets.get(name, {}) or {}
        runtime = self.runtime_fields.get(name, {})
        ref = DatasetRef(
            name=name,
            path=self.runtime_paths.get(name) or self.overrides.get(name) or cfg.get("path"),
            license=runtime.get("license", cfg.get("license")),
            attribution=runtime.get("attribution", cfg.get("attribution")),
            version=runtime.get("version", cfg.get("version")),
            mode=runtime.get("mode", cfg.get("mode", "local_cache")),
            optional=bool(runtime.get("optional", cfg.get("optional", True))),
            dataset_class=runtime.get("dataset_class", cfg.get("dataset_class")),
            snapshot_id=runtime.get("snapshot_id", cfg.get("snapshot_id")),
            coverage_scope=runtime.get("coverage_scope", cfg.get("coverage_scope")),
            source_resolution=runtime.get("source_resolution", cfg.get("source_resolution")),
            source_type=runtime.get("source_type", cfg.get("source_type")),
            proxy_used=bool(runtime.get("proxy_used", cfg.get("proxy_used", False))),
            fallback_used=bool(runtime.get("fallback_used", cfg.get("fallback_used", False))),
            outside_coverage=bool(runtime.get("outside_coverage", cfg.get("outside_coverage", False))),
            guidance_regime_version=runtime.get(
                "guidance_regime_version",
                cfg.get("guidance_regime_version"),
            ),
            authenticated=bool(runtime.get("authenticated", cfg.get("authenticated", False))),
            status=runtime.get("status", cfg.get("status")),
            metadata=dict(cfg.get("metadata", {}) or {}),
        )
        ref.metadata.update(dict(runtime.get("metadata", {}) or {}))
        if ref.path:
            p = Path(ref.path)
            ref.metadata["exists"] = p.exists()
            if p.exists():
                stat = p.stat()
                ref.metadata["filesize_bytes"] = stat.st_size
                ref.metadata["mtime_utc"] = datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat()
                if ref.version is None:
                    ref.version = p.name
                if ref.snapshot_id is None:
                    ref.snapshot_id = ref.version or p.name
                if ref.status is None:
                    ref.status = "available"
            else:
                ref.metadata["exists"] = False
                if ref.status is None:
                    ref.status = "path_missing"
        else:
            ref.metadata["exists"] = False
            if ref.status is None:
                ref.status = "missing"
        return ref

    def set_runtime_resolution(
        self,
        name: str,
        *,
        path: str,
        mode: str = "on_demand",
        license: str | None = None,
        attribution: str | None = None,
        version: str | None = None,
        dataset_class: str | None = None,
        snapshot_id: str | None = None,
        coverage_scope: str | None = None,
        source_resolution: str | None = None,
        source_type: str | None = None,
        proxy_used: bool | None = None,
        fallback_used: bool | None = None,
        outside_coverage: bool | None = None,
        guidance_regime_version: str | None = None,
        authenticated: bool | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.runtime_paths[name] = path
        fields = self.runtime_fields.setdefault(name, {})
        fields["mode"] = mode
        if license is not None:
            fields["license"] = license
        if attribution is not None:
            fields["attribution"] = attribution
        if version is not None:
            fields["version"] = version
        if dataset_class is not None:
            fields["dataset_class"] = dataset_class
        if snapshot_id is not None:
            fields["snapshot_id"] = snapshot_id
        if coverage_scope is not None:
            fields["coverage_scope"] = coverage_scope
        if source_resolution is not None:
            fields["source_resolution"] = source_resolution
        if source_type is not None:
            fields["source_type"] = source_type
        if proxy_used is not None:
            fields["proxy_used"] = bool(proxy_used)
        if fallback_used is not None:
            fields["fallback_used"] = bool(fallback_used)
        if outside_coverage is not None:
            fields["outside_coverage"] = bool(outside_coverage)
        if guidance_regime_version is not None:
            fields["guidance_regime_version"] = guidance_regime_version
        if authenticated is not None:
            fields["authenticated"] = bool(authenticated)
        if status is not None:
            fields["status"] = status
        if metadata:
            existing = dict(fields.get("metadata", {}) or {})
            existing.update(metadata)
            fields["metadata"] = existing

    def require_path(self, name: str) -> str:
        ref = self.resolve(name)
        if not ref.path:
            raise DatasetResolutionError(
                f"Dataset '{name}' path not provided. Supply --dataset {name}=<path>."
            )
        if not Path(ref.path).exists():
            raise DatasetResolutionError(f"Dataset '{name}' not found at: {ref.path}")
        return ref.path

    def to_metadata_records(self, used_names: list[str]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for name in used_names:
            records.append(asdict(self.resolve(name)))
        return records
