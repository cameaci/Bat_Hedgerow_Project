from __future__ import annotations

import json
from pathlib import Path

import click

from .acoustics import (
    AcousticImportSettings,
    AcousticValidationSettings,
    import_acoustic_evidence,
    read_acoustic_table,
    validate_acoustic_evidence,
)
from .deps import require_geopandas
from .io import prepare_working_gdf, read_input_geodata, write_geodata
from .models import RunOptions
from .planning import PlanningSettings, plan_static_detectors, write_planning_outputs
from .pipeline import run_enrichment
from .profile_loader import PACKAGE_PROFILE_DIR
from .screening import load_framework_bundle
from .screening.io import read_attribute_table
from .species import SpeciesTrainingSettings, train_species_model, write_species_artifacts


def _parse_dataset_overrides(entries: tuple[str, ...]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in entries:
        if "=" not in item:
            raise click.BadParameter(
                f"Invalid --dataset '{item}'. Expected NAME=PATH, e.g. --dataset worldcover=C:\\data\\worldcover.tif"
            )
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise click.BadParameter(f"Invalid --dataset '{item}'. NAME and PATH are required.")
        overrides[name] = path
    return overrides


def _parse_key_value_int_entries(entries: tuple[str, ...], *, option_name: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in entries:
        if "=" not in item:
            raise click.BadParameter(f"Invalid {option_name} '{item}'. Expected NAME=INT.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise click.BadParameter(f"Invalid {option_name} '{item}'. NAME and INT are required.")
        try:
            out[key] = int(value)
        except ValueError as exc:
            raise click.BadParameter(f"Invalid {option_name} '{item}'. INT is required.") from exc
    return out


def _parse_key_value_float_entries(entries: tuple[str, ...], *, option_name: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in entries:
        if "=" not in item:
            raise click.BadParameter(f"Invalid {option_name} '{item}'. Expected NAME=FLOAT.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise click.BadParameter(f"Invalid {option_name} '{item}'. NAME and FLOAT are required.")
        try:
            out[key] = float(value)
        except ValueError as exc:
            raise click.BadParameter(f"Invalid {option_name} '{item}'. FLOAT is required.") from exc
    return out


def _planning_weight_kwargs(entries: tuple[str, ...]) -> dict[str, float]:
    aliases = {
        "base_score": "objective_weight_base_score",
        "habitat_representation": "objective_weight_habitat_representation",
        "route_coverage": "objective_weight_route_coverage",
        "corridor_coverage": "objective_weight_corridor_coverage",
        "high_risk_coverage": "objective_weight_high_risk_coverage",
        "uncertainty_reduction": "objective_weight_uncertainty_reduction",
        "redundancy_penalty": "objective_weight_redundancy_penalty",
    }
    parsed = _parse_key_value_float_entries(entries, option_name="--objective-weight")
    out: dict[str, float] = {}
    for key, value in parsed.items():
        canonical = aliases.get(key)
        if canonical is None:
            raise click.BadParameter(
                f"Invalid --objective-weight key '{key}'. Supported: {', '.join(sorted(aliases))}."
            )
        out[canonical] = value
    return out


def _read_optional_geodata(path: Path | None):
    if path is None:
        return None
    gpd = require_geopandas()
    gdf = gpd.read_file(path)
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        raise click.BadParameter(f"Optional geodata has no CRS: {path}")
    return gdf


def _read_input_metadata(input_path: Path) -> dict[str, object] | None:
    candidates = [
        input_path.with_name("METADATA.json"),
        input_path.with_name(f"{input_path.stem}_METADATA.json"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            return {
                "status": "metadata_present_but_unreadable",
                "metadata_path": str(candidate),
            }
    return None


def _read_table_or_geodata(path: Path):
    suffix = path.suffix.lower()
    if suffix in {".csv", ".xlsx"}:
        return read_attribute_table(path)
    gpd = require_geopandas()
    return gpd.read_file(path)


def _write_dataframe_csv(df, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = df.copy()
    geom_name = getattr(getattr(out_df, "geometry", None), "name", None)
    if geom_name and geom_name in out_df.columns:
        out_df = out_df.drop(columns=[geom_name])
    out_df.to_csv(output_path, index=False)
    return str(output_path)


@click.group()
def main():
    """Hedgerow feature enrichment CLI."""


@main.command("list-profiles")
def list_profiles():
    """List bundled profiles."""
    for p in sorted(PACKAGE_PROFILE_DIR.glob("*.*")):
        if p.suffix.lower() in {".yaml", ".yml", ".json"}:
            click.echo(p.stem)


@main.command("enrich")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "output_path", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--profile", "profile_name", default="bats_v1", show_default=True)
@click.option("--profile-file", "profile_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--working-crs", default="EPSG:27700", show_default=True)
@click.option("--input-crs", default=None, help="Required if input dataset has no CRS metadata.")
@click.option("--export-crs", default=None, help="CRS for output geometry, or 'input' to restore input CRS.")
@click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--auto-fetch/--no-auto-fetch", default=True, show_default=True, help="Automatically fetch open datasets for the input AOI when local paths are not provided.")
@click.option("--earthdata-token", default=None, help="NASA Earthdata bearer token (used for authenticated providers when enabled).")
@click.option("--eog-username", default=None, help="EOG username for VIIRS/nightlights providers.")
@click.option("--eog-password", default=None, help="EOG password for VIIRS/nightlights providers.")
@click.option("--deployment-start-col", default=None, help="Hedgerow attribute column containing deployment start datetime.")
@click.option("--deployment-end-col", default=None, help="Hedgerow attribute column containing deployment end datetime.")
@click.option("--deployment-timezone", default="Europe/London", show_default=True, help="Timezone used for naive deployment datetimes and night windows.")
@click.option("--weather-backend", default="open_meteo", show_default=True, help="Temporal weather backend.")
@click.option("--min-night-overlap-minutes", default=30, show_default=True, type=int, help="Minimum overlap with a night window to count as a deployment night.")
@click.option("--temporal-features/--no-temporal-features", default=True, show_default=True)
@click.option("--roost-microhabitat-proxies/--no-roost-microhabitat-proxies", default=True, show_default=True)
@click.option("--dataset", "dataset_entries", multiple=True, help="Dataset override NAME=PATH (repeatable).")
@click.option("--batch-size", default=1000, show_default=True, type=int)
@click.option("--write-csv/--no-write-csv", default=False, show_default=True)
@click.option("--write-shp-zip/--no-write-shp-zip", default=False, show_default=True)
@click.option("--drop-all-null-features/--keep-all-null-features", default=True, show_default=True, help="Drop derived feature columns that are entirely null in this run.")
@click.option("--deterministic-output/--non-deterministic-output", default=True, show_default=True)
@click.option("--frozen-datasets-only/--allow-live-datasets", default=False, show_default=True, help="When enabled, reuse only local paths or cached snapshots and do not perform live downloads.")
@click.option("--json-summary/--no-json-summary", default=False, show_default=True)
def enrich(
    input_path: Path,
    output_path: Path,
    profile_name: str,
    profile_file: Path | None,
    working_crs: str,
    input_crs: str | None,
    export_crs: str | None,
    cache_dir: Path | None,
    auto_fetch: bool,
    earthdata_token: str | None,
    eog_username: str | None,
    eog_password: str | None,
    deployment_start_col: str | None,
    deployment_end_col: str | None,
    deployment_timezone: str,
    weather_backend: str,
    min_night_overlap_minutes: int,
    temporal_features: bool,
    roost_microhabitat_proxies: bool,
    dataset_entries: tuple[str, ...],
    batch_size: int,
    write_csv: bool,
    write_shp_zip: bool,
    drop_all_null_features: bool,
    deterministic_output: bool,
    frozen_datasets_only: bool,
    json_summary: bool,
):
    """Run profile-driven enrichment on a hedgerow dataset."""
    options = RunOptions(
        input_path=input_path,
        output_path=output_path,
        profile_name=profile_name,
        profile_path=profile_file,
        working_crs=working_crs,
        input_crs=input_crs,
        export_crs=export_crs,
        cache_dir=cache_dir,
        auto_fetch=auto_fetch,
        credentials={
            k: v
            for k, v in {
                "earthdata_token": earthdata_token,
                "eog_username": eog_username,
                "eog_password": eog_password,
            }.items()
            if v
        },
        drop_all_null_feature_columns=drop_all_null_features,
        deployment_start_column=deployment_start_col,
        deployment_end_column=deployment_end_col,
        deployment_timezone=deployment_timezone,
        weather_backend=weather_backend,
        min_night_overlap_minutes=min_night_overlap_minutes,
        enable_temporal_features=temporal_features,
        enable_roost_microhabitat_proxies=roost_microhabitat_proxies,
        dataset_overrides=_parse_dataset_overrides(dataset_entries),
        batch_size=batch_size,
        write_shapefile_zip=write_shp_zip,
        write_csv=write_csv,
        deterministic_output=deterministic_output,
        frozen_datasets_only=frozen_datasets_only,
    )
    result = run_enrichment(options)
    if json_summary:
        click.echo(json.dumps(result, indent=2))
        return
    click.echo(f"Rows: {result['rows']}  Columns: {result['columns']}")
    click.echo(f"Working CRS: {result['working_crs']}")
    for key, value in result["written"].items():
        click.echo(f"{key}: {value}")
    if result["notes"]:
        click.echo("Notes:")
        for note in result["notes"]:
            click.echo(f"- {note}")


@main.command("import-acoustics")
@click.option("--hedges", "hedges_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Hedgerow geospatial dataset to annotate with acoustic evidence.")
@click.option("--detections", "detections_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Acoustic detections table (.csv or .xlsx).")
@click.option("--output", "output_path", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--format", "source_format", default="generic", show_default=True, type=click.Choice(["generic", "batdetect2"], case_sensitive=False), help="Detection output format adapter.")
@click.option("--hedge-id-col", default="hf_uid", show_default=True, help="Hedgerow id column in the geospatial input.")
@click.option("--detection-hedge-id-col", default=None, help="Detection table column that directly references hedgerow ids. If omitted, latitude/longitude spatial matching is used.")
@click.option("--lat-col", default=None, help="Detection latitude column for spatial matching. Auto-detected when omitted.")
@click.option("--lon-col", default=None, help="Detection longitude column for spatial matching. Auto-detected when omitted.")
@click.option("--detections-crs", default="EPSG:4326", show_default=True, help="CRS for detection coordinates.")
@click.option("--max-distance-m", default=50.0, show_default=True, type=float, help="Maximum point-to-hedgerow match distance for spatial linking.")
@click.option("--datetime-col", default=None, help="Detection datetime/timestamp column. Auto-detected when omitted.")
@click.option("--species-col", default=None, help="Species/class column. Auto-detected when omitted.")
@click.option("--guild-col", default=None, help="Species guild/group column. Auto-detected when omitted.")
@click.option("--confidence-col", default=None, help="Confidence/probability column. Auto-detected when omitted.")
@click.option("--activity-col", default=None, help="Activity/call-count column. Auto-detected when omitted; each row counts as 1.")
@click.option("--min-confidence", default=None, type=float, help="Drop detections below this confidence before aggregation.")
@click.option("--acoustic-timezone", default="UTC", show_default=True, help="Timezone used when assigning detections to acoustic survey nights.")
@click.option("--night-rollover-hour", default=12, show_default=True, type=click.IntRange(0, 23), help="Local hour before which detections are assigned to the previous acoustic night.")
@click.option("--working-crs", default="EPSG:27700", show_default=True)
@click.option("--input-crs", default=None, help="Required if hedgerow input has no CRS metadata.")
@click.option("--export-crs", default=None, help="CRS for output geometry, or omit to keep working CRS.")
@click.option("--write-csv/--no-write-csv", default=False, show_default=True)
@click.option("--json-summary/--no-json-summary", default=False, show_default=True)
def import_acoustics_cmd(
    hedges_path: Path,
    detections_path: Path,
    output_path: Path,
    source_format: str,
    hedge_id_col: str,
    detection_hedge_id_col: str | None,
    lat_col: str | None,
    lon_col: str | None,
    detections_crs: str,
    max_distance_m: float,
    datetime_col: str | None,
    species_col: str | None,
    guild_col: str | None,
    confidence_col: str | None,
    activity_col: str | None,
    min_confidence: float | None,
    acoustic_timezone: str,
    night_rollover_hour: int,
    working_crs: str,
    input_crs: str | None,
    export_crs: str | None,
    write_csv: bool,
    json_summary: bool,
):
    """Attach aggregated acoustic bat detections to hedgerow segments."""
    raw_gdf, read_notes = read_input_geodata(hedges_path, input_crs=input_crs)
    hedges_gdf, _, repaired_count = prepare_working_gdf(raw_gdf, working_crs=working_crs, id_column=hedge_id_col)
    detections_df = read_acoustic_table(detections_path)
    settings = AcousticImportSettings(
        source_format=source_format.lower(),
        hedge_id_column=hedge_id_col,
        detection_hedge_id_column=detection_hedge_id_col,
        latitude_column=lat_col,
        longitude_column=lon_col,
        detections_crs=detections_crs,
        max_distance_m=max_distance_m,
        datetime_column=datetime_col,
        species_column=species_col,
        guild_column=guild_col,
        confidence_column=confidence_col,
        activity_column=activity_col,
        min_confidence=min_confidence,
        acoustic_timezone=acoustic_timezone,
        night_rollover_hour=night_rollover_hour,
    )
    out_gdf, summary = import_acoustic_evidence(hedges_gdf, detections_df, settings=settings)
    notes = list(read_notes)
    if repaired_count:
        notes.append(f"Repaired {repaired_count} invalid geometries.")
    notes.extend(summary.get("notes", []))
    metadata = {
        "tool": "hedge-features",
        "command": "import-acoustics",
        "source_hedges": str(hedges_path),
        "source_detections": str(detections_path),
        "summary": summary,
        "notes": notes,
    }
    written = write_geodata(out_gdf, output_path, metadata=metadata, export_crs=export_crs, write_csv=write_csv)
    payload = dict(summary)
    payload["written"] = written
    payload["notes"] = notes
    if json_summary:
        click.echo(json.dumps(payload, indent=2))
        return
    click.echo(f"Detections: {summary['records_after_confidence_filter']} after filtering")
    click.echo(f"Matched detections: {summary['matched_detection_records']}")
    click.echo(f"Hedgerows with acoustic evidence: {summary['hedgerows_with_acoustic_evidence']}")
    for key, value in written.items():
        click.echo(f"{key}: {value}")
    if notes:
        click.echo("Notes:")
        for note in notes:
            click.echo(f"- {note}")


@main.command("validate-acoustics")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Table or geodata already containing acoustic summary columns and a GIS score column.")
@click.option("--score-column", default="survey_priority_score", show_default=True, help="Numeric GIS screening/planning score column to compare against acoustic evidence.")
@click.option("--acoustic-presence-column", default="acoustic_detection_count", show_default=True, help="Numeric acoustic count/activity column used to define evidence presence.")
@click.option("--presence-threshold", default=1.0, show_default=True, type=float, help="Minimum acoustic count/activity value treated as acoustic presence.")
@click.option("--low-score-threshold", default=0.33, show_default=True, type=float, help="Scores at or below this threshold are treated as low GIS priority.")
@click.option("--high-score-threshold", default=0.67, show_default=True, type=float, help="Scores at or above this threshold are treated as high GIS priority.")
@click.option("--id-column", default="hf_uid", show_default=True, help="Optional row id column included in validation examples when present.")
@click.option("--species-list-column", default="acoustic_species_list", show_default=True)
@click.option("--guild-list-column", default="acoustic_guild_list", show_default=True)
@click.option("--max-examples", default=20, show_default=True, type=int)
@click.option("--output-json", default=None, type=click.Path(dir_okay=False, path_type=Path), help="Optional path for validation summary JSON.")
@click.option("--output-csv", default=None, type=click.Path(dir_okay=False, path_type=Path), help="Optional path for annotated validation rows CSV.")
@click.option("--json-summary/--no-json-summary", default=False, show_default=True)
def validate_acoustics_cmd(
    input_path: Path,
    score_column: str,
    acoustic_presence_column: str,
    presence_threshold: float,
    low_score_threshold: float,
    high_score_threshold: float,
    id_column: str | None,
    species_list_column: str,
    guild_list_column: str,
    max_examples: int,
    output_json: Path | None,
    output_csv: Path | None,
    json_summary: bool,
):
    """Compare GIS scores with imported acoustic evidence."""
    df = _read_table_or_geodata(input_path)
    settings = AcousticValidationSettings(
        score_column=score_column,
        acoustic_presence_column=acoustic_presence_column,
        acoustic_presence_threshold=presence_threshold,
        low_score_threshold=low_score_threshold,
        high_score_threshold=high_score_threshold,
        id_column=id_column,
        species_list_column=species_list_column,
        guild_list_column=guild_list_column,
        max_examples=max_examples,
    )
    annotated, summary = validate_acoustic_evidence(df, settings=settings)
    written: dict[str, str] = {}
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        written["json"] = str(output_json)
    if output_csv is not None:
        written["csv"] = _write_dataframe_csv(annotated, output_csv)
    payload = dict(summary)
    payload["written"] = written
    if json_summary:
        click.echo(json.dumps(payload, indent=2))
        return
    click.echo(f"Rows: {summary['row_count']}")
    click.echo(f"Acoustic presence rows: {summary['acoustic_presence_count']}")
    cases = summary["validation_case_counts"]
    click.echo(f"High score without acoustic evidence: {cases.get('high_score_no_acoustic_evidence', 0)}")
    click.echo(f"Low score with acoustic evidence: {cases.get('low_score_with_acoustic_evidence', 0)}")
    for key, value in written.items():
        click.echo(f"{key}: {value}")


@main.command("plan-statics")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "output_path", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--working-crs", default="EPSG:27700", show_default=True)
@click.option("--input-crs", default=None, help="Required if input dataset has no CRS metadata.")
@click.option("--detector-budget", required=True, type=int)
@click.option("--candidate-spacing", default=100.0, show_default=True, type=float)
@click.option("--endpoint-offset", default=20.0, show_default=True, type=float)
@click.option("--min-detector-spacing", default=150.0, show_default=True, type=float)
@click.option("--evidence-engine/--no-evidence-engine", default=True, show_default=True, help="Compute guild-based ecological evidence scores before planning.")
@click.option("--optimizer", "optimizer_strategy", default="greedy", show_default=True, type=click.Choice(["greedy", "exact"], case_sensitive=False), help="Detector-selection optimizer strategy. Greedy is the deterministic v1 strategy; exact searches small candidate sets exhaustively.")
@click.option("--objective-weight", "objective_weight_entries", multiple=True, help="Override planner objective weight NAME=FLOAT. Supported names: base_score, route_coverage, corridor_coverage, habitat_representation, high_risk_coverage, uncertainty_reduction, redundancy_penalty.")
@click.option("--score-column", default=None, help="Optional numeric source column used to override the computed planning priority score.")
@click.option("--target-scenario", default="all_bats", show_default=True, help="Planner evidence target scenario, e.g. all_bats, edge_commuter, common_pipistrelle, barbastelle.")
@click.option("--min-score", default=None, type=float, help="Minimum candidate score required for eligibility.")
@click.option("--access-flag-column", default=None, help="Boolean/flag column indicating access suitability.")
@click.option("--section-column", default=None, help="Optional source column used for section-based minimum quotas.")
@click.option("--section-min", "section_min_entries", multiple=True, help="Section minimum quota NAME=INT (repeatable).")
@click.option("--include-area", "include_area_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Optional polygon/area dataset used as an inclusion mask.")
@click.option("--exclude-area", "exclude_area_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Optional polygon/area dataset used as an exclusion mask.")
@click.option("--reject-overlit-candidates/--allow-overlit-candidates", default=True, show_default=True, help="Reject candidates explicitly flagged as over-lit by the planning evidence layer.")
@click.option("--reject-low-confidence-candidates/--allow-low-confidence-candidates", default=False, show_default=True, help="Reject candidates with low evidence confidence or outside-domain status.")
@click.option("--deterministic-output/--non-deterministic-output", default=True, show_default=True)
@click.option("--json-summary/--no-json-summary", default=False, show_default=True)
def plan_statics(
    input_path: Path,
    output_path: Path,
    working_crs: str,
    input_crs: str | None,
    detector_budget: int,
    candidate_spacing: float,
    endpoint_offset: float,
    min_detector_spacing: float,
    evidence_engine: bool,
    optimizer_strategy: str,
    objective_weight_entries: tuple[str, ...],
    score_column: str | None,
    target_scenario: str,
    min_score: float | None,
    access_flag_column: str | None,
    section_column: str | None,
    section_min_entries: tuple[str, ...],
    include_area_path: Path | None,
    exclude_area_path: Path | None,
    reject_overlit_candidates: bool,
    reject_low_confidence_candidates: bool,
    deterministic_output: bool,
    json_summary: bool,
):
    """Generate deterministic static-detector candidates and select a detector set."""
    raw_gdf, _ = read_input_geodata(input_path, input_crs=input_crs)
    hedges_gdf, _, _ = prepare_working_gdf(raw_gdf, working_crs=working_crs, id_column="hf_uid")
    include_area_gdf = _read_optional_geodata(include_area_path)
    exclude_area_gdf = _read_optional_geodata(exclude_area_path)
    source_metadata = _read_input_metadata(input_path)

    objective_weight_kwargs = _planning_weight_kwargs(objective_weight_entries)
    settings = PlanningSettings(
        detector_budget=detector_budget,
        optimizer_strategy=optimizer_strategy.lower(),
        candidate_spacing_m=candidate_spacing,
        endpoint_offset_m=endpoint_offset,
        min_detector_spacing_m=min_detector_spacing,
        use_evidence_engine=evidence_engine,
        score_column=score_column,
        target_scenario=target_scenario,
        min_score=min_score,
        access_flag_column=access_flag_column,
        section_column=section_column,
        section_minimum_counts=_parse_key_value_int_entries(section_min_entries, option_name="--section-min"),
        reject_overlit_candidates=reject_overlit_candidates,
        reject_low_confidence_candidates=reject_low_confidence_candidates,
        deterministic_output=deterministic_output,
        **objective_weight_kwargs,
    )
    result = plan_static_detectors(
        hedges_gdf,
        settings=settings,
        include_area_gdf=include_area_gdf,
        exclude_area_gdf=exclude_area_gdf,
        hedge_id_column="hf_uid",
    )
    written = write_planning_outputs(
        result,
        output_path,
        source_name=input_path.name,
        source_metadata=source_metadata,
    )
    if json_summary:
        payload = dict(result.run_summary)
        payload["written"] = written
        click.echo(json.dumps(payload, indent=2))
        return
    click.echo(f"Candidates: {len(result.candidates_gdf)}")
    click.echo(f"Selected: {len(result.selected_gdf)}")
    for key, value in written.items():
        click.echo(f"{key}: {value}")


@main.command("train-species-model")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--species-name", required=True, help="Species/guild target name used in artefact filenames and reports.")
@click.option("--target-column", required=True, help="Binary 0/1 training label column.")
@click.option("--framework-name", default="bats_screening_v1", show_default=True)
@click.option("--framework-dir", default=None, type=click.Path(exists=True, file_okay=False, path_type=Path), help="Optional framework directory to write artefacts into.")
@click.option("--output-dir", default=None, type=click.Path(file_okay=False, path_type=Path), help="Optional output directory for species artefacts. Defaults to <framework>/species_models.")
@click.option("--geography-column", default=None, help="Optional grouping column used for geography holdout evaluation.")
@click.option("--cv-folds", default=5, show_default=True, type=int)
@click.option("--min-positive-rows", default=10, show_default=True, type=int)
@click.option("--max-iter", default=400, show_default=True, type=int)
@click.option("--learning-rate", default=0.2, show_default=True, type=float)
@click.option("--l2-strength", default=0.01, show_default=True, type=float)
@click.option("--json-summary/--no-json-summary", default=False, show_default=True)
def train_species_model_cmd(
    input_path: Path,
    species_name: str,
    target_column: str,
    framework_name: str,
    framework_dir: Path | None,
    output_dir: Path | None,
    geography_column: str | None,
    cv_folds: int,
    min_positive_rows: int,
    max_iter: int,
    learning_rate: float,
    l2_strength: float,
    json_summary: bool,
):
    """Train a calibrated species-target model and write framework artefacts."""
    framework = load_framework_bundle(framework_name, framework_dir=framework_dir)
    training_df = read_attribute_table(input_path)
    settings = SpeciesTrainingSettings(
        species_name=species_name,
        target_column=target_column,
        geography_column=geography_column,
        cv_folds=cv_folds,
        min_positive_rows=min_positive_rows,
        max_iter=max_iter,
        learning_rate=learning_rate,
        l2_strength=l2_strength,
    )
    result = train_species_model(training_df, framework=framework, settings=settings)
    target_output_dir = output_dir or (framework.root_dir / framework.manifest.species_models_dir)
    written = write_species_artifacts(
        result,
        output_dir=target_output_dir,
        framework_dir=framework.root_dir,
    )
    if json_summary:
        payload = dict(result.summary)
        payload["written"] = written
        click.echo(json.dumps(payload, indent=2))
        return
    click.echo(f"Species target: {species_name}")
    click.echo(f"Rows: {result.summary['row_count']}")
    click.echo(f"Positive rows: {result.summary['positive_count']}")
    for key, value in written.items():
        click.echo(f"{key}: {value}")


if __name__ == "__main__":  # pragma: no cover
    main()
