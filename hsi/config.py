"""Central configuration for the England Bat Hedgerow Suitability Index tool.

Everything a user might want to tune lives here as a documented default: the WSP
suitability-index band thresholds, the default sub-index weights, the landscape
context settings, and the England data-source registry. Defaults are chosen so the
tool works out-of-the-box; the UI exposes the weights/alpha for fine-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------------------
# Spatial defaults
# --------------------------------------------------------------------------------------

WORKING_CRS = "EPSG:27700"  # British National Grid — required for metric distances/areas.
EXPORT_CRS = "EPSG:27700"

# Buffer (m) added around the hedgerow extent when fetching/clipping auxiliary data.
AOI_BUFFER_M = 2000.0


# --------------------------------------------------------------------------------------
# WSP / HyNet hedgerow suitability indices (SI1-SI7)
# --------------------------------------------------------------------------------------

SI_KEYS = ("si1", "si2", "si3", "si4", "si5", "si6", "si7")

SI_LABELS = {
    "si1": "Height",
    "si2": "Width",
    "si3": "Gappiness",
    "si4": "Arable field margin",
    "si5": "Trees present",
    "si6": "Woody species diversity",
    "si7": "Wet ditch",
}

# Maximum band value for each SI (used to normalise to 0-1). SI4/SI5 top out at 4.
SI_MAX = {
    "si1": 3,
    "si2": 3,
    "si3": 3,
    "si4": 4,
    "si5": 4,
    "si6": 3,
    "si7": 2,
}

# Default per-SI weights for the structural score A. WSP's worked example used a simple
# (equal-weighted) mean, so all default to 1.0. The UI lets the user change these.
DEFAULT_SI_WEIGHTS = {key: 1.0 for key in SI_KEYS}

# SI6 (woody species diversity) cannot be derived from remote data. Per the WSP guidance
# it gets a precautionary neutral default and always-Low confidence unless field-supplied.
SI6_DEFAULT_BAND = 2  # mid band on the 1-3 SI6 scale

# WSP suitability categories, evaluated on the *unweighted arithmetic mean* of the raw
# SI scores (NOT the normalised values), to stay faithful to the published method:
#   1.00-1.69 = Poor, 1.70-2.39 = Good, >=2.40 = Excellent.
WSP_CATEGORY_GOOD_LOWER = 1.70
WSP_CATEGORY_EXCELLENT_LOWER = 2.40

# Survey effort implied by each category (decision-support output). These follow the
# England guidance regime used by the original project and can be edited here.
GUIDANCE_REGIME_VERSION = "bct4_ne2025_england_v1"
SURVEY_REQUIREMENTS = {
    "Poor": "No further survey",
    "Good": "Seasonal automated static detector survey",
    "Excellent": "Seasonal automated static plus monthly modified DEFRA local-level surveys",
    "Incomplete": "Field verification required before survey effort can be reduced",
}

# Numeric weight per confidence level, used for the aggregate confidence score (0-1).
CONFIDENCE_WEIGHTS = {"High": 1.0, "Medium": 0.6, "Low": 0.3, "Missing": 0.0}


# --------------------------------------------------------------------------------------
# Landscape context layer (stage B). Each sub-index is 0-1 where higher = better for bats.
# These DO NOT change the WSP category (which is purely structural); they refine the
# final survey-priority ranking.
# --------------------------------------------------------------------------------------

CONTEXT_KEYS = (
    "ctx_woodland",
    "ctx_water",
    "ctx_connectivity",
    "ctx_roost",
    "ctx_darkness",
    "ctx_road_severance",
)

CONTEXT_LABELS = {
    "ctx_woodland": "Woodland proximity",
    "ctx_water": "Water proximity",
    "ctx_connectivity": "Network connectivity",
    "ctx_roost": "Roost potential",
    "ctx_darkness": "Darkness (low light)",
    "ctx_road_severance": "Road quietness",
}

DEFAULT_CONTEXT_WEIGHTS = {
    "ctx_woodland": 0.20,
    "ctx_water": 0.20,
    "ctx_connectivity": 0.15,
    "ctx_roost": 0.20,
    "ctx_darkness": 0.15,
    "ctx_road_severance": 0.10,
}

# Distance-decay scales (m) for the context sub-indices. invdist(d, s) = s / (s + d).
CONTEXT_SCALES_M = {
    "water": 150.0,
    "woodland": 250.0,
    "roost_building": 100.0,
    "roost_structure": 200.0,
    "road": 100.0,        # "far from road is good" — uses 1 - invdist
    "darkness_built": 200.0,  # "far from built-up is darker" — uses 1 - invdist
}

# Blend of structural A and context B into the final priority score:
#   priority = alpha * A + (1 - alpha) * B
DEFAULT_ALPHA = 0.65  # structure-led by default


# --------------------------------------------------------------------------------------
# Data directory conventions. The user drops pre-downloaded England national datasets
# into these folders (see data/README.md); the resolver auto-discovers them and clips to
# the area of interest. Anything absent falls back to a live API or a documented proxy.
# --------------------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache"

# dataset name -> subfolder under data/ that may hold a pre-downloaded local copy.
LOCAL_DATA_SUBDIRS = {
    "ea_lidar_dtm": "lidar/dtm",
    "ea_lidar_dsm": "lidar/dsm",
    "crome": "crome",
    "os_open_rivers": "os_open_rivers",
    "os_open_roads": "os_open_roads",
    "nightlights": "nightlights",
    "ancient_trees": "ancient_trees",
    "ne_phi": "ne_phi",
    "ne_awi": "ne_awi",
    "worldcover": "worldcover",
    "living_england": "living_england",
}

RASTER_EXTS = (".tif", ".tiff", ".asc", ".vrt")
VECTOR_EXTS = (".gpkg", ".shp", ".geojson", ".json")


# --------------------------------------------------------------------------------------
# England dataset registry (auto_provider entries mirror the proven configuration). Only
# sources that actually return data are wired live; large national rasters/vectors are
# pre-downloaded locally. Verify reachability any time with scripts/check_sources.py.
# --------------------------------------------------------------------------------------

ENGLAND_DATASETS = {
    # --- live, anonymous open APIs -------------------------------------------------
    "os_open_roads": {
        "license": "OGL v3.0 (or OSM ODbL fallback)",
        "attribution": "Contains OS data © Crown copyright (OGL), or OpenStreetMap contributors (ODbL).",
        "optional": True,
        "source_type": "vector_lines",
        "coverage_scope": "england_or_better",
        "auto_provider": {"type": "osm_overpass_lines", "query_kind": "roads", "aoi_buffer_m": 3000},
    },
    "os_open_rivers": {
        "license": "OGL v3.0 (or OSM ODbL fallback)",
        "attribution": "Contains OS data © Crown copyright (OGL), or OpenStreetMap contributors (ODbL).",
        "optional": True,
        "source_type": "vector_lines",
        "coverage_scope": "england_or_better",
        "auto_provider": {"type": "osm_overpass_lines", "query_kind": "waterways", "aoi_buffer_m": 3000},
    },
    "worldcover": {
        "license": "CC BY 4.0",
        "attribution": "ESA WorldCover © ESA and partners (CC BY 4.0).",
        "optional": True,
        "source_type": "raster_categorical",
        "coverage_scope": "global",
        "auto_provider": {"type": "pc_stac_raster", "collection": "esa-worldcover", "asset": "map", "aoi_buffer_m": 1500},
    },
    "ne_phi": {
        "license": "OGL v3.0 / DEFRA terms",
        "attribution": "Natural England Priority Habitats Inventory (OGL).",
        "optional": True,
        "source_type": "vector_polygons",
        "coverage_scope": "england",
        "auto_provider": {
            "type": "arcgis_feature_layer",
            "service_url": "https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/Priority_Habitats_Inventory_England/FeatureServer/0",
            "aoi_buffer_m": 2000,
        },
    },
    "ne_awi": {
        "license": "OGL v3.0 / DEFRA terms",
        "attribution": "Natural England Ancient Woodland Inventory (OGL).",
        "optional": True,
        "source_type": "vector_polygons",
        "coverage_scope": "england",
        "auto_provider": {
            "type": "arcgis_feature_layer",
            "service_url": "https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/Ancient_Woodland_England/FeatureServer/0",
            "aoi_buffer_m": 5000,
        },
    },
    "osm_buildings": {
        "license": "ODbL 1.0",
        "attribution": "Contains OpenStreetMap data © OpenStreetMap contributors (ODbL).",
        "optional": True,
        "source_type": "vector_features",
        "coverage_scope": "england_or_better",
        "auto_provider": {"type": "osm_overpass_features", "query_kind": "buildings", "aoi_buffer_m": 2000},
    },
    "osm_structures_roost": {
        "license": "ODbL 1.0",
        "attribution": "Contains OpenStreetMap data © OpenStreetMap contributors (ODbL).",
        "optional": True,
        "source_type": "vector_features",
        "coverage_scope": "england_or_better",
        "auto_provider": {"type": "osm_overpass_features", "query_kind": "structures_roost", "aoi_buffer_m": 3000},
    },
    # --- local pre-download only (no clean anonymous API) --------------------------
    "ea_lidar_dtm": {
        "license": "OGL v3.0",
        "attribution": "Environment Agency National LiDAR Programme DTM (OGL).",
        "optional": True,
        "source_type": "raster_dtm",
        "coverage_scope": "england",
        "auto_provider": {
            "type": "ea_wcs_raster",
            "wcs_url": "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs",
        },
        "download_hint": "Auto-fetched via EA WCS; or drop tiles in data/lidar/dtm from https://environment.data.gov.uk/survey (LIDAR Composite DTM 1m).",
    },
    "ea_lidar_dsm": {
        "license": "OGL v3.0",
        "attribution": "Environment Agency National LiDAR Programme DSM (OGL).",
        "optional": True,
        "source_type": "raster_dsm",
        "coverage_scope": "england",
        "auto_provider": {
            "type": "ea_wcs_raster",
            "wcs_url": "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-surface-model-first-return-dsm-1m/wcs",
        },
        "download_hint": "Auto-fetched via EA WCS; or drop tiles in data/lidar/dsm from https://environment.data.gov.uk/survey (LIDAR Composite DSM 1m).",
    },
    "crome": {
        "license": "OGL v3.0",
        "attribution": "RPA Crop Map of England (CROME) (OGL).",
        "optional": True,
        "source_type": "vector_polygons",
        "coverage_scope": "england",
        "local_only": True,
        "download_hint": "https://environment.data.gov.uk (Crop Map of England) — county shapefile(s) for your AOI.",
    },
    "nightlights": {
        "license": "Various (e.g. VIIRS VNL / Falchi 2016)",
        "attribution": "Night-time lights raster (e.g. EOG VIIRS VNL V2, or Falchi et al. 2016 World Atlas).",
        "optional": True,
        "source_type": "raster_continuous",
        "coverage_scope": "global",
        "local_only": True,
        "download_hint": "https://eogdata.mines.edu (VIIRS VNL annual, Europe tile) — optional; without it a distance proxy is used.",
    },
    "ancient_trees": {
        "license": "Woodland Trust open data terms",
        "attribution": "Woodland Trust Ancient Tree Inventory.",
        "optional": True,
        "source_type": "vector_points",
        "coverage_scope": "uk",
        "local_only": True,
        "download_hint": "https://ati.woodlandtrust.org.uk (export for your AOI) — optional roost refinement.",
    },
    "living_england": {
        "license": "OGL v3.0",
        "attribution": "Natural England Living England Habitat Map (OGL).",
        "optional": True,
        "source_type": "vector_polygons",
        "coverage_scope": "england",
        "local_only": True,
        "download_hint": "https://naturalengland-defra.opendata.arcgis.com (Living England Habitat Map) — optional; drop in data/living_england to enrich woodland context.",
    },
}

# WorldCover class codes (10 m) used for cropland (SI4) and land-cover context.
WORLDCOVER_CLASS_MAP = {
    "tree": [10],
    "shrubland": [20],
    "grassland": [30],
    "cropland": [40],
    "built": [50],
    "bare_sparse": [60],
    "water": [80],
    "wetland": [90],
}

# Candidate field names / value tokens for detecting arable polygons in CROME.
CROME_CLASS_FIELD_CANDIDATES = ("lucode", "LUCODE", "crop_name", "primary_crop", "land_use", "Land_Use")
CROME_ARABLE_TOKENS = (
    "wheat", "barley", "oat", "cereal", "maize", "oilseed", "rape", "bean",
    "pea", "potato", "sugar", "linseed", "arable", "fallow", "AC",
)

# Living England habitat map: candidate class-field names and woodland tokens.
LIVING_ENGLAND_FIELD_CANDIDATES = ("A_pred", "Primary_Hab", "primary_habitat", "habitat", "Habitat", "HABITAT", "main_habit")
LIVING_ENGLAND_WOODLAND_TOKENS = ("wood", "broadleaf", "broadleaved", "forest")


@dataclass(slots=True)
class ScoreSettings:
    """User-tunable scoring parameters (defaults from this module)."""

    si_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SI_WEIGHTS))
    context_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CONTEXT_WEIGHTS))
    alpha: float = DEFAULT_ALPHA
    si6_default_band: int = SI6_DEFAULT_BAND
    guidance_regime_version: str = GUIDANCE_REGIME_VERSION
