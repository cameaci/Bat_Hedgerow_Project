# Bat Hedgerow Suitability Index (HSI) — England

A decision-support tool for ecologists: **upload a hedgerow shapefile → get a ranked list of
which hedgerows to prioritise for static bat surveys**, with a map and downloadable outputs.

It is a transparent, weighted implementation of the WSP / HyNet (Sarah Proctor) hedgerow
suitability assessment — seven suitability indices (SI1–SI7) — derived remotely for England
from open GIS data, plus a separate landscape-context layer to refine the survey priority.

> The HSI measures *habitat suitability*, not bat presence. A high score means a hedgerow is a
> higher priority to survey, not that bats are definitely present. Interpret it alongside survey
> data and professional judgement.

## What it does

1. Reads a hedgerow line layer (zipped shapefile / GeoPackage / GeoJSON), reprojects to EPSG:27700.
2. Derives the seven WSP suitability indices remotely (see below), each with a confidence flag.
3. Computes the WSP suitability category (**Poor / Good / Excellent**) and a 0–1 survey-priority rank.
4. Lets you fine-tune sub-index weights (sensible defaults) and filter/sort results by any sub-index.
5. Exports a scored GeoPackage / Shapefile / CSV plus a method statement.

## The seven indices (WSP / HyNet)

| Index | Measures | England remote proxy | Confidence |
|------|----------|----------------------|------------|
| SI1 Height | hedgerow height | EA 1 m LiDAR canopy-height model | Medium |
| SI2 Width | hedgerow width | EA 1 m LiDAR canopy width | Medium |
| SI3 Gappiness | canopy gaps | EA 1 m LiDAR gap fraction | Medium |
| SI4 Arable margin | adjacent arable / margin | CROME / ESA WorldCover cropland | Low |
| SI5 Trees present | trees per 50 m | EA 1 m LiDAR canopy / WorldCover | Medium |
| SI6 Woody species diversity | woody species count | **precautionary default** (not remotely verifiable) | Low |
| SI7 Wet ditch | wet ditch present | watercourse proximity (OS / OSM) | Low |

The structural category is the unweighted arithmetic mean of SI1–SI7 (`<1.70` Poor, `1.70–2.39`
Good, `≥2.40` Excellent). A **landscape-context** layer (woodland, water and roost proximity,
darkness, connectivity, road quietness) is combined with the structural score
(`priority = α·structure + (1−α)·context`, default α = 0.65) to give the final survey-priority rank.
If you measured any index in the field, add it as a column (e.g. `height_m`, `woody_species_count_20m`)
and it is used at high confidence instead of the proxy.

## Quickstart

```bash
pip install -e .                 # installs the GIS + Streamlit stack
streamlit run app.py             # open the app, upload a hedgerow layer
```

Try it with the bundled sample:

```bash
python scripts/check_sources.py --aoi sample_data/sample_hedgerows_england.gpkg
```

`sample_data/sample_hedgerows_england.gpkg` is a small England hedgerow network you can upload directly.

## Data (England, optional pre-download)

Live open data (OpenStreetMap, ESA WorldCover, Natural England Priority Habitats / Ancient
Woodland) is fetched automatically and clipped to your area. For the **structural** indices
(SI1–SI3, SI5) you need EA 1 m LiDAR, which is downloaded once into `data/`. See
[`data/README.md`](data/README.md) for the folder layout and exact download links, and run
`python scripts/check_sources.py` to verify which sources return data for your area.

Without LiDAR the tool still ranks hedgerows from land-cover, water, woodland, roost and darkness
context, and honestly flags the WSP category as **Incomplete** (field verification required).

## Outputs

- **Ranked table** — priority rank, WSP category & score, SI1–SI7 scores, confidence, recommended survey effort.
- **Map** — hedgerows coloured by category, with the SI breakdown on hover.
- **Downloads** — GeoPackage, zipped Shapefile, CSV, run metadata (`METADATA.json`) and a Markdown method statement.

## Development

```bash
pip install -e ".[dev]"
pytest                           # 26 tests (scoring maths incl. the WSP worked example, LiDAR, context, offline degradation)
```

### Layout

- `hsi/` — the tool: `config` (weights, thresholds, England dataset registry), `ingest`, `datasets`
  (local discovery + AOI clip + live fallback), `indices` (SI proxies), `context` (landscape layer),
  `score` (the scoring engine), `pipeline` (orchestration), `report` (outputs + method statement).
- `app.py` — the single Streamlit UI.
- `scripts/check_sources.py` — probe each data source one-by-one for a test AOI.
- `hedge_features/` — retained, proven GIS plumbing (I/O, dataset fetchers, feature calculators) reused by `hsi/`.
