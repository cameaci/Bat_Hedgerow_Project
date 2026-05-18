from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import __version__
from .data_health import build_data_catalogue, build_feature_health
from .datasets.auto_fetch import AutoDataFetcher
from .datasets.registry import DatasetRegistry
from .deps import require_geopandas
from .exceptions import DatasetResolutionError, InputValidationError
from .features.deployment import add_deployment_nights_metrics
from .features.geometry import add_geometry_metrics
from .features.lidar_structure import add_lidar_hedgerow_structure_features
from .features.moon import add_moonlight_nightly_aggregate
from .features.network import add_network_metrics
from .features.proxies_microhabitat import add_microhabitat_proxy_features
from .features.proxies_roost import add_roost_proxy_features
from .features.raster import (
    add_raster_categorical_proportions_in_buffers,
    add_raster_zonal_stats_in_buffers,
)
from .features.vector import (
    add_vector_distance,
    add_vector_line_density_in_buffers,
    add_vector_polygon_composition_in_buffers,
)
from .features.weather import add_weather_nightly_aggregate
from .io import prepare_working_gdf, read_input_geodata, write_geodata, write_metadata_readme
from .models import RunMetadata, RunOptions
from .profile_loader import resolve_profile
from .utils import dedupe_strings


def _parse_buffers(module_cfg: dict[str, Any], profile_cfg: dict[str, Any]) -> list[int]:
    values = module_cfg.get("buffers_m", profile_cfg.get("buffers_m", []))
    out: list[int] = []
    for v in values:
        out.append(int(v))
    return out


def _load_vector_dataset(path: str, working_crs: str):
    gpd = require_geopandas()
    gdf = gpd.read_file(path)
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        raise InputValidationError(f"Auxiliary vector dataset has no CRS: {path}")
    if str(gdf.crs) != str(working_crs):
        gdf = gdf.to_crs(working_crs)
    return gdf


def _resolve_dataset_path_for_module(
    module_cfg: dict[str, Any],
    registry: DatasetRegistry,
    notes: list[str],
    fetcher: AutoDataFetcher | None = None,
) -> tuple[str | None, str | None]:
    dataset_name = module_cfg.get("dataset")
    if not dataset_name:
        return None, None
    ref = registry.resolve(dataset_name)
    if ref.path and Path(ref.path).exists():
        return dataset_name, ref.path
    if fetcher is not None:
        auto_path, auto_notes = fetcher.ensure_dataset(dataset_name)
        notes.extend(auto_notes)
        if auto_path and Path(auto_path).exists():
            return dataset_name, auto_path
    if not ref.optional:
        raise DatasetResolutionError(
            f"Required dataset '{dataset_name}' is missing. Supply --dataset {dataset_name}=<path>."
        )
    notes.append(
        f"Dataset '{dataset_name}' unavailable; module '{module_cfg.get('type')}' will output nulls/flags."
    )
    return dataset_name, None


def _resolve_named_dataset_path(
    dataset_name: str,
    *,
    registry: DatasetRegistry,
    notes: list[str],
    fetcher: AutoDataFetcher | None = None,
) -> str | None:
    ref = registry.resolve(dataset_name)
    if ref.path and Path(ref.path).exists():
        return ref.path
    if fetcher is not None:
        auto_path, auto_notes = fetcher.ensure_dataset(dataset_name)
        notes.extend(auto_notes)
        if auto_path and Path(auto_path).exists():
            return auto_path
    if not ref.optional:
        raise DatasetResolutionError(
            f"Required dataset '{dataset_name}' is missing. Supply --dataset {dataset_name}=<path>."
        )
    notes.append(f"Dataset '{dataset_name}' unavailable.")
    return None


def _resolve_temporal_settings(options: RunOptions, profile: dict[str, Any]) -> dict[str, Any]:
    temporal_cfg = dict(profile.get("temporal", {}) or {})
    deployment_cols = dict(temporal_cfg.get("deployment_columns", {}) or {})
    return {
        "start_column": options.deployment_start_column or deployment_cols.get("start"),
        "end_column": options.deployment_end_column or deployment_cols.get("end"),
        "timezone": options.deployment_timezone or str(temporal_cfg.get("timezone", "Europe/London")),
        "weather_backend": options.weather_backend or str(temporal_cfg.get("weather_backend", "open_meteo")),
        "min_night_overlap_minutes": int(
            options.min_night_overlap_minutes
            if options.min_night_overlap_minutes is not None
            else temporal_cfg.get("min_night_overlap_minutes", 30)
        ),
        "night_window_rule": str(temporal_cfg.get("night_window_rule", "sunset_to_sunrise")),
    }


def _qa_sanity_checks(gdf) -> list[str]:
    notes: list[str] = []
    for col in gdf.columns:
        if col.startswith("dist_"):
            series = gdf[col].dropna()
            if not series.empty and (series < 0).any():
                notes.append(f"Sanity check: negative values found in {col}.")
        if col.endswith("_pct"):
            series = gdf[col].dropna()
            if not series.empty:
                if col.startswith("moon_"):
                    if (((series < 0) | (series > 100)).any()):
                        notes.append(f"Sanity check: out-of-range percent values found in {col}.")
                elif (((series < 0) | (series > 1)).any()):
                    notes.append(f"Sanity check: out-of-range proportion values found in {col}.")
    return notes


def _qa_line_geometry_output(gdf) -> list[str]:
    notes: list[str] = []
    geom_types = {str(t) for t in gdf.geometry.geom_type.dropna().unique()}
    invalid = geom_types - {"LineString", "MultiLineString"}
    if invalid:
        raise InputValidationError(
            "Output geometry type changed from line geometry unexpectedly. "
            f"Found non-line types: {sorted(invalid)}"
        )
    notes.append(f"Output geometry QA passed (line types only): {sorted(geom_types)}")
    return notes


def _max_profile_buffer_m(profile: dict[str, Any]) -> int:
    max_val = 0
    for v in profile.get("buffers_m", []) or []:
        try:
            max_val = max(max_val, int(v))
        except Exception:
            continue
    for module_cfg in profile.get("feature_modules", []) or []:
        for v in module_cfg.get("buffers_m", []) or []:
            try:
                max_val = max(max_val, int(v))
            except Exception:
                continue
    return max_val or 1000


def _apply_viirs_proxy_if_missing(hedges_gdf, module_cfg: dict[str, Any]) -> tuple[Any, list[str]]:
    """Populate nightlight columns with an open-data proxy if VIIRS is unavailable."""
    if module_cfg.get("proxy_if_missing") != "worldcover_roads":
        return hedges_gdf, []

    gdf = hedges_gdf.copy()
    pd = None
    radii = [int(r) for r in module_cfg.get("buffers_m", [])]
    stats = [str(s).lower() for s in module_cfg.get("stats", [])]
    col_template = module_cfg["column_template"]
    notes: list[str] = []
    proxy_used_any = False

    for radius in radii:
        built_col = f"buf{radius}_worldcover_built_pct"
        road_col = f"buf{radius}_os_road_density_m_per_ha"
        if built_col not in gdf.columns and road_col not in gdf.columns:
            continue

        if built_col in gdf.columns:
            built = gdf[built_col].astype("float64", copy=False)
        else:
            if pd is None:
                import pandas as pd  # local import to avoid hard dependency at module import time
            built = pd.Series(0.0, index=gdf.index)
        if road_col in gdf.columns:
            road = gdf[road_col].astype("float64", copy=False)
        else:
            if pd is None:
                import pandas as pd
            road = pd.Series(0.0, index=gdf.index)
        # Simple disturbance proxy: built-up proportion + scaled road density.
        proxy = (built.fillna(0).astype(float) * 80.0) + (road.fillna(0).astype(float).clip(lower=0) * 0.03)
        proxy = proxy.clip(lower=0.0, upper=100.0)

        for stat in stats:
            col = col_template.format(radius=radius, stat=stat)
            if col not in gdf.columns:
                gdf[col] = None
            if stat in {"mean", "median"}:
                values = proxy
            elif stat == "p90":
                values = (proxy * 1.15 + 1.0).clip(lower=0.0, upper=100.0)
            elif stat == "min":
                values = (proxy * 0.8).clip(lower=0.0, upper=100.0)
            elif stat == "max":
                values = (proxy * 1.25 + 2.0).clip(lower=0.0, upper=100.0)
            else:
                continue
            null_mask = gdf[col].isna()
            if null_mask.any():
                gdf.loc[null_mask, col] = values[null_mask]
                proxy_used_any = True

    if proxy_used_any:
        notes.append(
            "VIIRS nightlights were unavailable for anonymous auto-download; filled nightlight columns with an open-data proxy derived from WorldCover built-up proportion and road density."
        )
    return gdf, notes


def _drop_all_null_feature_columns(gdf, *, original_columns: list[str]) -> tuple[Any, list[str]]:
    original = set(original_columns)
    geom_col_name = getattr(getattr(gdf, "geometry", None), "name", None)
    feature_cols = [c for c in gdf.columns if c not in original and c != geom_col_name]
    dropped: list[str] = []
    for col in feature_cols:
        series = gdf[col]
        if series.isna().all():
            dropped.append(col)
    if not dropped:
        return gdf, []
    gdf = gdf.drop(columns=dropped)
    note = f"Dropped {len(dropped)} derived columns that were all null: {', '.join(dropped)}"
    return gdf, [note]


def run_enrichment(options: RunOptions) -> dict[str, Any]:
    profile, profile_path = resolve_profile(options.profile_name, options.profile_path)
    working_crs = options.working_crs or str(profile.get("working_crs", "EPSG:27700"))
    id_column = ((profile.get("output") or {}).get("id_column")) or "hf_uid"
    guidance_regime_version = str(profile.get("guidance_regime_version", "")).strip() or None
    temporal_settings = _resolve_temporal_settings(options, profile)

    raw_gdf, read_notes = read_input_geodata(options.input_path, input_crs=options.input_crs)
    hedges_gdf, input_crs, repaired_count = prepare_working_gdf(
        raw_gdf, working_crs=working_crs, id_column=id_column
    )
    original_output_columns = list(hedges_gdf.columns)
    notes: list[str] = list(read_notes)
    if repaired_count:
        notes.append(f"Repaired {repaired_count} invalid geometries.")

    registry = DatasetRegistry(
        profile_datasets=profile.get("datasets"),
        overrides=options.dataset_overrides,
        cache_dir=options.cache_dir,
    )
    auto_fetcher = (
        AutoDataFetcher(
            registry=registry,
            profile_datasets=profile.get("datasets"),
            hedges_gdf=hedges_gdf,
            working_crs=working_crs,
            cache_dir=options.cache_dir,
            max_buffer_m=float(_max_profile_buffer_m(profile)),
            credentials=options.credentials,
            allow_live_fetch=not options.frozen_datasets_only,
        )
        if options.auto_fetch
        else None
    )
    loaded_vectors: dict[str, Any] = {}
    runtime_ctx: dict[str, Any] = {
        "deployment": None,
        "weather": None,
    }
    used_dataset_names: list[str] = []

    for module_cfg in profile.get("feature_modules", []):
        if not module_cfg.get("enabled", True):
            continue
        mod_type = module_cfg.get("type")
        if not mod_type:
            continue

        if mod_type == "geometry_metrics":
            hedges_gdf = add_geometry_metrics(hedges_gdf)
            continue

        if mod_type == "network_metrics":
            tolerance_m = float(module_cfg.get("tolerance_m", 1.0))
            hedges_gdf, mod_notes = add_network_metrics(hedges_gdf, tolerance_m=tolerance_m)
            notes.extend(mod_notes)
            continue

        if mod_type == "deployment_nights_metrics":
            if not options.enable_temporal_features:
                notes.append("Temporal features disabled; deployment_nights_metrics skipped.")
                continue
            result = add_deployment_nights_metrics(
                hedges_gdf,
                start_column=module_cfg.get("start_column") or temporal_settings["start_column"],
                end_column=module_cfg.get("end_column") or temporal_settings["end_column"],
                timezone_name=module_cfg.get("timezone") or temporal_settings["timezone"],
                min_night_overlap_minutes=int(
                    module_cfg.get("min_night_overlap_minutes", temporal_settings["min_night_overlap_minutes"])
                ),
                write_local_columns=bool(module_cfg.get("write_local_columns", False)),
            )
            hedges_gdf = result.gdf
            runtime_ctx["deployment"] = result.context
            notes.extend(result.notes)
            continue

        if mod_type == "weather_nightly_aggregate":
            if not options.enable_temporal_features:
                notes.append("Temporal features disabled; weather_nightly_aggregate skipped.")
                continue
            result = add_weather_nightly_aggregate(
                hedges_gdf,
                deployment_context=runtime_ctx.get("deployment"),
                cache_dir=options.cache_dir,
                timezone_name=module_cfg.get("timezone") or temporal_settings["timezone"],
                backend=module_cfg.get("backend") or temporal_settings["weather_backend"],
            )
            hedges_gdf = result.gdf
            runtime_ctx["weather"] = result.context
            notes.extend(result.notes)
            weather_dataset_name = module_cfg.get("dataset") or "open_meteo_archive"
            used_dataset_names.append(weather_dataset_name)
            if result.context and result.context.provider_metadata:
                cache_root = result.context.provider_metadata.get("cache_dir")
                if cache_root:
                    registry.set_runtime_resolution(
                        weather_dataset_name,
                        path=str(Path(cache_root)),
                        mode="on_demand_api",
                        version="live-api",
                        metadata=result.context.provider_metadata,
                    )
            continue

        if mod_type == "moonlight_nightly_aggregate":
            if not options.enable_temporal_features:
                notes.append("Temporal features disabled; moonlight_nightly_aggregate skipped.")
                continue
            result = add_moonlight_nightly_aggregate(
                hedges_gdf,
                deployment_context=runtime_ctx.get("deployment"),
                weather_context=runtime_ctx.get("weather"),
                use_cloud_adjustment=bool(module_cfg.get("use_cloud_adjustment", True)),
            )
            hedges_gdf = result.gdf
            notes.extend(result.notes)
            continue

        if mod_type == "osm_roost_proxies":
            if not options.enable_roost_microhabitat_proxies:
                notes.append("Roost/microhabitat proxies disabled; osm_roost_proxies skipped.")
                continue
            buildings_ds = str(module_cfg.get("buildings_dataset", "osm_buildings"))
            structs_ds = str(module_cfg.get("structures_dataset", "osm_structures_roost"))
            buildings_path = _resolve_named_dataset_path(
                buildings_ds, registry=registry, notes=notes, fetcher=auto_fetcher
            )
            structures_path = _resolve_named_dataset_path(
                structs_ds, registry=registry, notes=notes, fetcher=auto_fetcher
            )
            used_dataset_names.extend([buildings_ds, structs_ds])
            buildings_gdf = None
            if buildings_path:
                buildings_gdf = loaded_vectors.get(buildings_ds)
                if buildings_gdf is None:
                    buildings_gdf = _load_vector_dataset(buildings_path, working_crs)
                    loaded_vectors[buildings_ds] = buildings_gdf
            structures_gdf = None
            if structures_path:
                structures_gdf = loaded_vectors.get(structs_ds)
                if structures_gdf is None:
                    structures_gdf = _load_vector_dataset(structures_path, working_crs)
                    loaded_vectors[structs_ds] = structures_gdf
            result = add_roost_proxy_features(
                hedges_gdf,
                buildings_gdf=buildings_gdf,
                structures_gdf=structures_gdf,
            )
            hedges_gdf = result.gdf
            notes.extend(result.notes)
            continue

        if mod_type == "microhabitat_proxies":
            if not options.enable_roost_microhabitat_proxies:
                notes.append("Roost/microhabitat proxies disabled; microhabitat_proxies skipped.")
                continue
            worldcover_ds = str(module_cfg.get("worldcover_dataset", "worldcover"))
            copdem_ds = str(module_cfg.get("copdem_dataset", "copdem"))
            worldcover_path = _resolve_named_dataset_path(
                worldcover_ds, registry=registry, notes=notes, fetcher=auto_fetcher
            )
            copdem_path = _resolve_named_dataset_path(
                copdem_ds, registry=registry, notes=notes, fetcher=auto_fetcher
            )
            used_dataset_names.extend([worldcover_ds, copdem_ds])
            result = add_microhabitat_proxy_features(
                hedges_gdf,
                worldcover_path=worldcover_path,
                copdem_path=copdem_path,
                include_roost_proxy_alias=bool(module_cfg.get("include_roost_proxy_alias", True)),
            )
            hedges_gdf = result.gdf
            notes.extend(result.notes)
            continue

        if mod_type == "lidar_hedgerow_structure":
            dtm_ds = str(module_cfg.get("dtm_dataset", "ea_lidar_dtm"))
            dsm_ds = str(module_cfg.get("dsm_dataset", "ea_lidar_dsm"))
            dtm_path = _resolve_named_dataset_path(
                dtm_ds, registry=registry, notes=notes, fetcher=auto_fetcher
            )
            dsm_path = _resolve_named_dataset_path(
                dsm_ds, registry=registry, notes=notes, fetcher=auto_fetcher
            )
            used_dataset_names.extend([dtm_ds, dsm_ds])
            result = add_lidar_hedgerow_structure_features(
                hedges_gdf,
                dtm_path=dtm_path,
                dsm_path=dsm_path,
                height_buffer_m=float(module_cfg.get("height_buffer_m", 5.0)),
                continuity_buffer_m=float(module_cfg.get("continuity_buffer_m", 10.0)),
            )
            hedges_gdf = result.gdf
            notes.extend(result.notes)
            continue

        dataset_name, dataset_path = _resolve_dataset_path_for_module(
            module_cfg, registry, notes, fetcher=auto_fetcher
        )
        if dataset_name:
            used_dataset_names.append(dataset_name)

        if mod_type == "vector_distance":
            target = None
            if dataset_path:
                target = loaded_vectors.get(dataset_name)
                if target is None:
                    target = _load_vector_dataset(dataset_path, working_crs)
                    loaded_vectors[dataset_name] = target
            result = add_vector_distance(
                hedges_gdf,
                target,
                distance_column=module_cfg["distance_column"],
                geometry_kinds=module_cfg.get("geometry_kinds"),
                coverage_flag_column=module_cfg.get("coverage_flag_column"),
            )
            hedges_gdf = result.gdf
            notes.extend(result.notes)
            continue

        if mod_type == "vector_line_density":
            target = None
            if dataset_path:
                target = loaded_vectors.get(dataset_name)
                if target is None:
                    target = _load_vector_dataset(dataset_path, working_crs)
                    loaded_vectors[dataset_name] = target
            result = add_vector_line_density_in_buffers(
                hedges_gdf,
                target,
                radii_m=_parse_buffers(module_cfg, profile),
                density_column_template=module_cfg["density_column_template"],
                output_metric=module_cfg.get("output_metric", "m_per_ha"),
            )
            hedges_gdf = result.gdf
            notes.extend(result.notes)
            continue

        if mod_type == "vector_polygon_composition":
            target = None
            if dataset_path:
                target = loaded_vectors.get(dataset_name)
                if target is None:
                    target = _load_vector_dataset(dataset_path, working_crs)
                    loaded_vectors[dataset_name] = target
            result = add_vector_polygon_composition_in_buffers(
                hedges_gdf,
                target,
                radii_m=_parse_buffers(module_cfg, profile),
                class_field=module_cfg["class_field"],
                selected_classes=module_cfg.get("selected_classes", {}),
                column_template=module_cfg["column_template"],
                coverage_flag_column=module_cfg.get("coverage_flag_column"),
            )
            hedges_gdf = result.gdf
            notes.extend(result.notes)
            continue

        if mod_type == "raster_zonal_stats":
            result = add_raster_zonal_stats_in_buffers(
                hedges_gdf,
                dataset_path,
                radii_m=_parse_buffers(module_cfg, profile),
                stats=[str(s) for s in module_cfg.get("stats", [])],
                column_template=module_cfg["column_template"],
                centroid_sample_column=module_cfg.get("centroid_sample_column"),
            )
            hedges_gdf = result.gdf
            notes.extend(result.notes)
            if dataset_name == "viirs_nightlights" and not dataset_path:
                hedges_gdf, proxy_notes = _apply_viirs_proxy_if_missing(hedges_gdf, module_cfg)
                notes.extend(proxy_notes)
                if proxy_notes:
                    viirs_runtime = registry.runtime_fields.setdefault("viirs_nightlights", {})
                    viirs_runtime["fallback_used"] = True
                    viirs_runtime["status"] = "fallback_proxy"
            continue

        if mod_type == "raster_categorical_proportions":
            class_map = {
                str(k): [int(v) for v in vals]
                for k, vals in (module_cfg.get("class_map") or {}).items()
            }
            result = add_raster_categorical_proportions_in_buffers(
                hedges_gdf,
                dataset_path,
                radii_m=_parse_buffers(module_cfg, profile),
                class_map=class_map,
                column_template=module_cfg["column_template"],
                patch_richness_column_template=module_cfg.get("patch_richness_column_template"),
                landscape_class_names=[str(c) for c in module_cfg.get("landscape_class_names", [])],
                landscape_metrics=[str(m) for m in module_cfg.get("landscape_metrics", [])],
                landscape_column_templates=module_cfg.get("landscape_column_templates") or {},
            )
            hedges_gdf = result.gdf
            notes.extend(result.notes)
            continue

        notes.append(f"Unknown feature module type '{mod_type}' skipped.")

    notes.extend(_qa_sanity_checks(hedges_gdf))
    notes.extend(_qa_line_geometry_output(hedges_gdf))
    if options.drop_all_null_feature_columns:
        hedges_gdf, drop_notes = _drop_all_null_feature_columns(
            hedges_gdf, original_columns=original_output_columns
        )
        notes.extend(drop_notes)
    used_dataset_names = dedupe_strings(used_dataset_names)
    data_catalogue = build_data_catalogue(
        profile=profile,
        registry=registry,
        used_dataset_names=used_dataset_names,
        guidance_regime_version=guidance_regime_version,
    )
    feature_health = build_feature_health(
        hedges_gdf,
        data_catalogue=data_catalogue,
        guidance_regime_version=guidance_regime_version,
    )

    export_crs = options.export_crs
    if export_crs and export_crs.lower() == "input":
        export_crs = str(input_crs) if input_crs else None

    metadata = asdict(
        RunMetadata.build(
            tool_version=__version__,
            input_path=Path(options.input_path),
            output_path=Path(options.output_path),
            working_crs=working_crs,
            export_crs=export_crs,
            profile_name=options.profile_name,
            profile_path=profile_path,
            datasets=registry.to_metadata_records(used_dataset_names),
            guidance_regime_version=guidance_regime_version,
            deterministic_output=options.deterministic_output,
            frozen_datasets_only=options.frozen_datasets_only,
            notes=notes,
        )
    )
    metadata["profile"] = {
        "name": profile.get("name", options.profile_name),
        "description": profile.get("description"),
        "buffers_m": profile.get("buffers_m", []),
    }
    metadata["guidance_regime_version"] = guidance_regime_version
    metadata["temporal"] = temporal_settings
    metadata["credentials"] = {
        "earthdata_token_provided": bool(options.credentials.get("earthdata_token")),
        "eog_credentials_provided": bool(
            options.credentials.get("eog_username") and options.credentials.get("eog_password")
        ),
    }
    metadata["data_catalogue"] = data_catalogue
    metadata["feature_health"] = feature_health

    written = write_geodata(
        hedges_gdf,
        options.output_path,
        metadata=metadata,
        export_crs=export_crs,
        write_csv=options.write_csv,
        write_shapefile_zip=options.write_shapefile_zip,
    )
    readme_output = write_metadata_readme(options.output_path, metadata)
    written["output_readme"] = str(readme_output)
    data_catalogue_path = Path(options.output_path).with_name("DATA_CATALOGUE.json")
    feature_health_path = Path(options.output_path).with_name("FEATURE_HEALTH.json")
    import json

    data_catalogue_path.write_text(json.dumps(data_catalogue, indent=2), encoding="utf-8")
    feature_health_path.write_text(json.dumps(feature_health, indent=2), encoding="utf-8")
    written["data_catalogue"] = str(data_catalogue_path)
    written["feature_health"] = str(feature_health_path)

    return {
        "rows": int(len(hedges_gdf)),
        "columns": int(len(hedges_gdf.columns)),
        "working_crs": working_crs,
        "input_crs": str(input_crs) if input_crs else None,
        "export_crs": export_crs,
        "written": written,
        "guidance_regime_version": guidance_regime_version,
        "data_catalogue": data_catalogue,
        "feature_health": feature_health,
        "notes": notes,
    }
