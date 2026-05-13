# Hedge Features

`hedge-features` is a file-in, file-out GIS enrichment tool for hedgerow segments. It reads a hedgerow layer, computes ecological and landscape-context features from open data, appends them as new columns, and writes an enriched output dataset plus run metadata.

It also includes a Streamlit-based **GIS-only bat survey screening / prioritisation** workflow for unsurveyed hedgerows. This screening uses only GIS-derived features from enriched outputs (or uploaded pre-enriched tables) and returns a prioritisation score, confidence, reason codes, and recommended action.

It also includes a **static detector planner** in both CLI and Streamlit UI form. The planner can generate candidate detector points from enriched hedgerows, compute guild-based ecological evidence scores, apply inclusion/exclusion/access constraints, select a deterministic detector set, support expert review with audit-trailed manual overrides, and write a bankability-oriented evidence pack.

It also now includes a **species / calibration layer** for framework-specific bat targets. This layer does not ship with bundled calibrated species models by default, but it can train, validate, and package versioned species artefacts from historical static survey data when those labels are available.

By default, the app now attempts to auto-fetch open datasets for the input AOI (OSM/ArcGIS/Planetary Computer-backed sources) when local dataset paths are not supplied.

## Supported Input

- Zipped shapefile package (`.zip`) containing at least `.shp`, `.shx`, `.dbf` (and ideally `.prj`)
- GeoPackage (`.gpkg`)
- GeoJSON (`.geojson`, `.json`)

`*.shx` alone is not a valid input. The tool rejects it with a clear error.

## Output

- Primary: GeoPackage (`.gpkg`)
- Optional: zipped ESRI Shapefile (`.zip`) and CSV attributes (`.csv`)
- `METADATA.json` is written alongside the output

## Screening (GIS-only Bat Prioritisation)

The Streamlit app now has a second workflow for screening unsurveyed hedgerows using GIS-derived features only.

Two entry routes are supported:

- Upload a **pre-enriched** table (`.csv` or `.xlsx`) containing GIS feature columns (for example `geom_*`, `net_*`, `dist_*`, `buf*_*`, `pt_*`, `roostpx_*`, `mhb_*`)
- Run the in-app GIS enrichment workflow first, then apply screening to the enriched output in the same Streamlit session

Screening outputs append columns such as:

- `survey_priority_score` (ranking score; **not** a guaranteed probability)
- `survey_priority_band` (`Low`, `Medium`, `High`)
- `confidence_level`
- `reason_codes`
- `recommended_action`

Default screening policy:

- `Recall-first`

Confidence is first-class:

- low-confidence rows are flagged for review and are **not** auto-deprioritised
- reason codes explain common issues (missing required predictors, low GIS coverage, status gaps, outliers, etc.)

## Quick Start

```powershell
pip install -e ".[gis,dev]"
hedge-features enrich `
  --input hedges.zip `
  --output enriched.gpkg `
  --profile bats_v1 `
  --json-summary
```

Optional local path overrides (advanced):

```powershell
hedge-features enrich `
  --input hedges.zip `
  --output enriched.gpkg `
  --profile bats_v1 `
  --dataset worldcover=C:\data\worldcover_10m.tif `
  --dataset copdem=C:\data\copdem_glo30.tif
```

Planning a first detector set:

```powershell
hedge-features plan-statics `
  --input enriched.gpkg `
  --output static_plan.gpkg `
  --detector-budget 12 `
  --candidate-spacing 100 `
  --min-detector-spacing 150 `
  --json-summary
```

The planner now writes an evidence pack alongside the requested screened output path:

- screened hedgerow GPKG
- candidate point GPKG
- chosen detector set GPKG
- run manifest JSON
- evidence report Markdown

Training a species-target calibration model:

```powershell
hedge-features train-species-model `
  --input historical_static_surveys.csv `
  --species-name "Pipistrellus pipistrellus" `
  --target-column pip_pip_present `
  --framework-name bats_screening_v1 `
  --geography-column project_package `
  --json-summary
```

## Streamlit UI

Run the app:

```powershell
streamlit run .\hedge_features\ui_streamlit.py
```

Tabs/workflows in the UI:

- `GIS Enrichment`: existing file-in, file-out enrichment flow (now also offers CSV download)
- `GIS-only Bat Screening`: upload pre-enriched CSV/XLSX or reuse in-app enrichment output
- `Static Detector Planner`: upload enriched geospatial hedgerows or reuse in-app enrichment output, run deterministic detector planning, review candidates on a map, apply expert overrides, and export reviewed GPKG/CSV/XLSX/JSON outputs

Screening UI modes:

- `Default`: strict GIS-only predictors, packaged thresholds, `Recall-first` policy, confidence override enabled
- `Advanced`: policy, confidence strictness, thresholds, profile mismatch handling, and other controls

## Notes

- Working CRS defaults to `EPSG:27700` (British National Grid) for distance/area calculations.
- If the input CRS is missing, supply `--input-crs` or the run will fail.
- England-only datasets (e.g. PHI/AWI) should return nulls outside coverage; profile flags support this.
- VIIRS nightlights may require authenticated sources in some environments; when anonymous auto-download is unavailable, the default profile fills nightlight columns using a documented open-data proxy (WorldCover built-up + road density) and records this in `METADATA.json`.
- For Shapefile export, long field names are truncated and a field-map JSON is written.
- Screening is a **decision-support framework**, not a replacement for ecological judgement.
- The screening score (`survey_priority_score`) is a prioritisation/ranking score, not a calibrated bat occurrence probability unless a calibrated framework artefact is explicitly packaged and enabled.
- Planning uses a transparent guild-based evidence engine (`edge_open`, `clutter_linear`, `woodland_sensitive`) and writes `eco_suitability_score`, `survey_utility_score`, `planning_priority_score`, `evidence_confidence_level`, and `evidence_reason_codes`.
- The planner ignores `mhb_roost_proxy_score` in evidence scoring to avoid double-counting the same roost signal already represented by `roostpx_struct_proxy_score`.
- Detector selection now uses a deterministic greedy coverage optimizer rather than simple top-K score ranking. The optimizer balances habitat representation, route coverage, high-risk corridor coverage, uncertainty reduction, and redundancy minimisation.
- The Streamlit planner adds a map-first review workflow with `Project Setup`, `Candidate Map`, `Optimisation`, `Expert Review`, and `Exports` surfaces, plus a review audit trail and reviewed GeoPackage export layers (`source_hedges`, `candidates`, `selected_auto`, `selected_final`).
- The planner evidence pack is intended for defensible review: the candidate and screened exports now carry explicit `why selected`, `why not selected`, missing-data summaries, confidence summaries, dataset provenance, and framework version metadata.

## Project Layout

- `hedge_features/io.py`: input validation, ingestion, export, metadata
- `hedge_features/pipeline.py`: profile-driven enrichment pipeline
- `hedge_features/screening/`: reusable GIS-only screening engine (framework loading, column governance, confidence, I/O)
- `hedge_features/planning/`: static detector planning engine (candidate generation, ecological evidence scoring, constraints, optimisation, reporting)
- `hedge_features/species/`: species-model training, artefact writing, runtime inference, and domain-of-applicability logic
- `hedge_features/ui_planner.py`: Streamlit planner workflow (setup, map review, override audit trail, exports)
- `hedge_features/frameworks/bats_screening_v1/`: bundled versioned screening artefacts (manifest, registry, thresholds, confidence rules, model spec)
- `hedge_features/features/`: geometry, vector, raster, network feature calculators
- `hedge_features/datasets/`: dataset registry and local cache path resolution
- `hedge_features/cli.py`: command line interface
- `hedge_features/ui_streamlit.py`: Streamlit UI for enrichment + GIS-only screening
- `tests/`: unit and integration tests

## Framework Artefacts (Screening)

Bundled screening frameworks live under `hedge_features/frameworks/<framework_name>/` and are loaded at runtime.

Current bundled package:

- `hedge_features/frameworks/bats_screening_v1/`

Key artefacts:

- `framework_manifest.json`
- `feature_registry.json`
- `triage_model.json` (versioned scoring model spec; ranking score output)
- `triage_thresholds.json`
- `confidence_rules.json`
- `species_models/` (optional, currently placeholder)

Species calibration artefacts created by `train-species-model` include:

- `species_<target>_model.json`
- `species_<target>_model_card.json`
- `species_<target>_model_card.md`
- `species_<target>_domain.json`
- `species_<target>_training_summary.json`

When trained species models are present in the framework bundle and the screening species module is enabled, the screening engine appends:

- `species_<target>_probability`
- `species_<target>_domain_score`
- `species_<target>_domain_status`
- `species_<target>_reason_codes`
- `README_framework.md`

This design keeps screening logic versioned and separate from Streamlit UI code.

streamlit run .\hedge_features\ui_streamlit.py
