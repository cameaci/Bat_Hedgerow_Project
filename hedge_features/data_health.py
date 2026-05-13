from __future__ import annotations

from dataclasses import asdict
from typing import Any


DATA_CATALOGUE_VERSION = "data_catalogue_v1"
FEATURE_HEALTH_VERSION = "feature_health_v1"


def build_data_catalogue(
    *,
    profile: dict[str, Any],
    registry,
    used_dataset_names: list[str],
    guidance_regime_version: str | None,
) -> dict[str, Any]:
    profile_datasets = profile.get("datasets") or {}
    used = set(str(name) for name in used_dataset_names)
    records: list[dict[str, Any]] = []

    for dataset_name in sorted(profile_datasets):
        cfg = profile_datasets.get(dataset_name, {}) or {}
        ref = registry.resolve(dataset_name)
        record = asdict(ref)
        provider = (cfg.get("auto_provider") or {}).get("type")
        exists = bool(ref.metadata.get("exists"))
        authenticated = bool(ref.authenticated)
        dataset_class = str(ref.dataset_class or cfg.get("dataset_class") or "anonymous_open")
        enabled = dataset_name in used
        status = _dataset_status(
            enabled=enabled,
            exists=exists,
            authenticated=authenticated,
            dataset_class=dataset_class,
            path=ref.path,
            provider_type=str(provider or ""),
        )
        if status == "missing" and str(ref.status or "").strip() not in {"", "missing"}:
            status = str(ref.status)
        record.update(
            {
                "enabled_in_run": enabled,
                "provider_type": provider,
                "dataset_class": dataset_class,
                "status": status,
                "guidance_regime_version": guidance_regime_version,
            }
        )
        records.append(record)

    return {
        "artifact_version": DATA_CATALOGUE_VERSION,
        "profile_name": str(profile.get("name", "")),
        "guidance_regime_version": guidance_regime_version,
        "datasets": records,
        "summary": {
            "dataset_count": int(len(records)),
            "enabled_dataset_count": int(sum(1 for record in records if record["enabled_in_run"])),
            "counts_by_status": _count_values(record["status"] for record in records),
            "counts_by_class": _count_values(record["dataset_class"] for record in records),
        },
    }


def build_feature_health(
    gdf,
    *,
    data_catalogue: dict[str, Any],
    guidance_regime_version: str | None,
) -> dict[str, Any]:
    geometry_name = getattr(getattr(gdf, "geometry", None), "name", None)
    dataset_by_name = {
        str(record.get("name")): record
        for record in (data_catalogue.get("datasets") or [])
    }
    feature_records: list[dict[str, Any]] = []

    for col in gdf.columns:
        if col == geometry_name:
            continue
        family = _feature_family(col)
        source_dataset = _feature_dataset_name(col)
        series = gdf[col]
        null_fraction = float(series.isna().mean()) if len(series) else 0.0
        numeric_summary = _numeric_summary(series)
        dataset_record = dataset_by_name.get(source_dataset)
        support_state = _feature_support_state(
            column_name=col,
            dataset_record=dataset_record,
            null_fraction=null_fraction,
        )
        feature_records.append(
            {
                "column_name": str(col),
                "feature_family": family,
                "source_dataset_name": source_dataset,
                "source_type": (dataset_record or {}).get("source_type"),
                "dataset_class": (dataset_record or {}).get("dataset_class"),
                "support_state": support_state,
                "proxy_used": bool((dataset_record or {}).get("proxy_used", False)),
                "fallback_used": bool((dataset_record or {}).get("fallback_used", False)),
                "outside_coverage": bool((dataset_record or {}).get("outside_coverage", False)),
                "null_fraction": round(null_fraction, 6),
                "non_null_count": int(series.notna().sum()),
                "null_count": int(series.isna().sum()),
                "numeric_summary": numeric_summary,
            }
        )

    return {
        "artifact_version": FEATURE_HEALTH_VERSION,
        "guidance_regime_version": guidance_regime_version,
        "row_count": int(len(gdf)),
        "feature_count": int(len(feature_records)),
        "features": feature_records,
        "summary": {
            "counts_by_support_state": _count_values(record["support_state"] for record in feature_records),
            "counts_by_feature_family": _count_values(record["feature_family"] for record in feature_records),
            "high_null_features": [
                record["column_name"]
                for record in feature_records
                if float(record["null_fraction"]) >= 0.95
            ],
        },
    }


def _dataset_status(
    *,
    enabled: bool,
    exists: bool,
    authenticated: bool,
    dataset_class: str,
    path: str | None,
    provider_type: str,
) -> str:
    if enabled and exists:
        return "available"
    if enabled and authenticated and not exists:
        return "authenticated_required"
    if enabled and dataset_class == "client_project" and not exists:
        return "project_input_required"
    if enabled and path:
        return "path_missing"
    if enabled and provider_type == "unsupported":
        return "manual_or_authenticated_source_required"
    if enabled:
        return "missing"
    return "configured_not_used"


def _feature_support_state(*, column_name: str, dataset_record: dict[str, Any] | None, null_fraction: float) -> str:
    if column_name.endswith("_status"):
        return "status_flag"
    if column_name.endswith("_coverage_flag"):
        return "coverage_flag"
    if dataset_record is None:
        return "derived_internal"
    if bool(dataset_record.get("outside_coverage", False)):
        return "outside_coverage"
    if bool(dataset_record.get("fallback_used", False)):
        return "fallback"
    if bool(dataset_record.get("proxy_used", False)):
        return "proxy"
    if float(null_fraction) >= 0.95 and str(dataset_record.get("status")) not in {"available", "configured_not_used"}:
        return "missing_source"
    return "measured_or_derived"


def _feature_family(column_name: str) -> str:
    col = str(column_name)
    if col.startswith("geom_"):
        return "geometry"
    if col.startswith("net_"):
        return "network"
    if "os_road" in col:
        return "roads"
    if "os_river" in col or "water" in col:
        return "water"
    if "worldcover" in col:
        return "landcover"
    if "nightlight" in col:
        return "lighting"
    if "copdem" in col:
        return "terrain"
    if col.startswith("buf") and "_phi_" in col:
        return "priority_habitat"
    if "awi" in col:
        return "ancient_woodland"
    if col.startswith("roostpx_"):
        return "roost_proxy"
    if col.startswith("mhb_"):
        return "microhabitat"
    if col.startswith("hedge_struct_"):
        return "hedgerow_structure"
    if col.startswith("living_"):
        return "living_england"
    if col.startswith("dep") or col.startswith("moon_"):
        return "temporal"
    return "other"


def _feature_dataset_name(column_name: str) -> str | None:
    col = str(column_name)
    if "os_road" in col:
        return "os_open_roads"
    if "os_river" in col or col == "mhb_water_dist_m":
        return "os_open_rivers"
    if "worldcover" in col or col.startswith("mhb_corridor10_"):
        return "worldcover"
    if "nightlight" in col:
        return "viirs_nightlights"
    if "copdem" in col or col.startswith("mhb_dem_"):
        return "copdem"
    if "_phi_" in col:
        return "ne_phi"
    if "awi" in col:
        return "ne_awi"
    if col.startswith("roostpx_"):
        return "osm_structures_roost"
    if col.startswith("hedge_struct_"):
        return "ea_lidar_dsm"
    if col.startswith("living_"):
        return "living_england_habitat"
    return None


def _numeric_summary(series) -> dict[str, float | None] | None:
    try:
        numeric = series.astype("float64")
    except Exception:
        return None
    numeric = numeric.dropna()
    if numeric.empty:
        return None
    return {
        "min": float(numeric.min()),
        "mean": float(numeric.mean()),
        "max": float(numeric.max()),
        "zero_fraction": float((numeric == 0).mean()),
    }


def _count_values(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
