from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from .deps import require_geopandas
from .exceptions import InputValidationError
from .utils import ensure_parent_dir, sha1_text


SUPPORTED_INPUT_EXTENSIONS = {".zip", ".gpkg", ".geojson", ".json", ".shp"}
LINE_GEOM_TYPES = {"LineString", "MultiLineString"}


def _find_shapefile_components(directory: Path) -> dict[str, list[Path]]:
    files = list(directory.rglob("*"))
    grouped: dict[str, list[Path]] = {".shp": [], ".shx": [], ".dbf": [], ".prj": []}
    for f in files:
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext in grouped:
            grouped[ext].append(f)
    return grouped


def _validate_shapefile_package(directory: Path) -> Path:
    grouped = _find_shapefile_components(directory)
    if not grouped[".shp"]:
        raise InputValidationError("Zip does not contain a .shp file.")
    if len(grouped[".shp"]) > 1:
        raise InputValidationError(
            "Zip contains multiple .shp files. Provide a single dataset per zip."
        )
    shp = grouped[".shp"][0]
    stem = shp.stem.lower()
    names = {ext: {p.stem.lower() for p in paths} for ext, paths in grouped.items()}
    missing = [ext for ext in (".shx", ".dbf") if stem not in names[ext]]
    if missing:
        raise InputValidationError(
            "Invalid shapefile package. Missing required component(s) for "
            f"{shp.name}: {', '.join(missing)}"
        )
    return shp


def _read_gdf(path: Path, input_crs: str | None = None):
    gpd = require_geopandas()
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise InputValidationError("Input dataset contains no features.")
    if gdf.geometry is None:
        raise InputValidationError("Input dataset has no geometry column.")
    if gdf.crs is None:
        if not input_crs:
            raise InputValidationError(
                "Input CRS is missing. Supply --input-crs to avoid incorrect distance metrics."
            )
        gdf = gdf.set_crs(input_crs)
    return gdf


def read_input_geodata(input_path: str | Path, input_crs: str | None = None):
    """Read user input and return GeoDataFrame plus notes."""
    path = Path(input_path)
    if not path.exists():
        raise InputValidationError(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".shx":
        raise InputValidationError(
            "A .shx index file alone is not a usable dataset. Upload a zipped shapefile "
            "containing at least .shp, .shx, and .dbf (ideally .prj), or use .gpkg/.geojson."
        )
    if suffix not in SUPPORTED_INPUT_EXTENSIONS:
        raise InputValidationError(
            f"Unsupported input format '{suffix}'. Supported: {sorted(SUPPORTED_INPUT_EXTENSIONS)}"
        )

    notes: list[str] = []
    if suffix == ".zip":
        with tempfile.TemporaryDirectory(prefix="hedge_features_in_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(tmp_path)
            shp_path = _validate_shapefile_package(tmp_path)
            if not shp_path.with_suffix(".prj").exists():
                notes.append(
                    "Input shapefile package did not contain a .prj file; CRS came from user input."
                    if input_crs
                    else "Input shapefile package did not contain a .prj file."
                )
            gdf = _read_gdf(shp_path, input_crs=input_crs)
            return gdf, notes

    if suffix == ".shp":
        for ext in (".shx", ".dbf"):
            if not path.with_suffix(ext).exists():
                raise InputValidationError(
                    f"Missing required shapefile component: {path.with_suffix(ext).name}"
                )
        if not path.with_suffix(".prj").exists():
            notes.append("Input shapefile missing .prj file.")
        return _read_gdf(path, input_crs=input_crs), notes

    return _read_gdf(path, input_crs=input_crs), notes


def validate_hedgerow_geometry_types(gdf) -> None:
    geom_types = {str(t) for t in gdf.geometry.geom_type.dropna().unique()}
    if not geom_types:
        raise InputValidationError("No valid geometries found.")
    invalid = geom_types - LINE_GEOM_TYPES
    if invalid:
        raise InputValidationError(
            "Expected LineString or MultiLineString geometries for hedgerows. "
            f"Found: {sorted(invalid)}"
        )


def repair_invalid_geometries(gdf):
    invalid_mask = ~gdf.geometry.is_valid
    if not invalid_mask.any():
        return gdf, 0
    gdf = gdf.copy()
    repaired_count = 0
    try:
        from shapely import make_valid  # shapely>=2
    except Exception:
        make_valid = None

    for idx in gdf.index[invalid_mask]:
        geom = gdf.at[idx, gdf.geometry.name]
        fixed = None
        if make_valid is not None:
            try:
                fixed = make_valid(geom)
            except Exception:
                fixed = None
        # Only accept repaired geometry if it remains lineal.
        if fixed is not None and getattr(fixed, "geom_type", None) in LINE_GEOM_TYPES and fixed.is_valid:
            gdf.at[idx, gdf.geometry.name] = fixed
            repaired_count += 1
    return gdf, repaired_count


def assign_internal_id(gdf, id_column: str = "hf_uid"):
    if id_column in gdf.columns and gdf[id_column].notna().all():
        return gdf
    gdf = gdf.copy()
    ids: list[str] = []
    seen: dict[str, int] = {}
    for idx, geom in zip(gdf.index, gdf.geometry):
        wkb_hex = geom.wkb_hex if geom is not None else f"null-{idx}"
        base = f"hf_{sha1_text(wkb_hex, length=12)}"
        dup_no = seen.get(base, 0)
        seen[base] = dup_no + 1
        value = base if dup_no == 0 else f"{base}_{dup_no}"
        ids.append(value)
    gdf[id_column] = ids
    return gdf


def prepare_working_gdf(
    gdf,
    *,
    working_crs: str,
    id_column: str = "hf_uid",
):
    validate_hedgerow_geometry_types(gdf)
    gdf, repaired_count = repair_invalid_geometries(gdf)
    gdf = gdf[gdf.geometry.notna()].copy()
    validate_hedgerow_geometry_types(gdf)
    gdf = assign_internal_id(gdf, id_column=id_column)
    input_crs = gdf.crs
    if str(gdf.crs) != str(working_crs):
        gdf = gdf.to_crs(working_crs)
    return gdf, input_crs, repaired_count


def _truncate_shapefile_columns(columns: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    used: set[str] = set()
    original_to_short: dict[str, str] = {}
    short_to_original: dict[str, str] = {}
    for col in columns:
        if col == "geometry":
            original_to_short[col] = col
            continue
        base = col[:10]
        candidate = base
        i = 1
        while candidate.lower() in used:
            suffix = str(i)
            candidate = f"{base[:10 - len(suffix)]}{suffix}"
            i += 1
        used.add(candidate.lower())
        original_to_short[col] = candidate
        short_to_original[candidate] = col
    return original_to_short, short_to_original


def write_geodata(
    gdf,
    output_path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    export_crs: str | None = None,
    write_csv: bool = False,
    write_shapefile_zip: bool = False,
) -> dict[str, str]:
    gpd = require_geopandas()
    path = Path(output_path)
    ensure_parent_dir(path)

    gdf_out = gdf.copy()
    if export_crs:
        gdf_out = gdf_out.to_crs(export_crs)

    suffix = path.suffix.lower()
    written: dict[str, str] = {}
    if suffix == ".gpkg":
        gdf_out.to_file(path, driver="GPKG")
        written["gpkg"] = str(path)
    elif suffix in {".geojson", ".json"}:
        gdf_out.to_file(path, driver="GeoJSON")
        written["geojson"] = str(path)
    elif suffix == ".shp":
        gdf_out.to_file(path, driver="ESRI Shapefile")
        written["shp"] = str(path)
    else:
        raise InputValidationError(
            f"Unsupported output extension '{suffix}'. Use .gpkg, .geojson, or .shp."
        )

    metadata_path = path.with_name("METADATA.json")
    if metadata is not None:
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        written["metadata"] = str(metadata_path)

    if write_csv:
        csv_path = path.with_suffix(".csv")
        attr_df = gdf_out.drop(columns=gdf_out.geometry.name).copy()
        attr_df.to_csv(csv_path, index=False)
        written["csv"] = str(csv_path)

    if write_shapefile_zip:
        zip_path, fmap_path = _write_zipped_shapefile(gdf_out, path)
        written["shapefile_zip"] = str(zip_path)
        written["shapefile_field_map"] = str(fmap_path)

    return written


def _write_zipped_shapefile(gdf, base_output_path: Path) -> tuple[Path, Path]:
    gpd = require_geopandas()
    stem = base_output_path.stem
    zip_path = base_output_path.with_name(f"{stem}_shapefile.zip")
    fmap_path = base_output_path.with_name(f"{stem}_shapefile_field_map.json")
    with tempfile.TemporaryDirectory(prefix="hedge_features_shp_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        shp_dir = tmp_dir_path / "shp"
        shp_dir.mkdir(parents=True, exist_ok=True)
        shp_path = shp_dir / f"{stem}.shp"
        orig_to_short, short_to_orig = _truncate_shapefile_columns(list(gdf.columns))
        gdf_shp = gdf.rename(columns={k: v for k, v in orig_to_short.items() if k != "geometry"})
        gdf_shp.to_file(shp_path, driver="ESRI Shapefile")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in shp_dir.iterdir():
                zf.write(f, arcname=f.name)
        fmap_path.write_text(json.dumps(short_to_orig, indent=2), encoding="utf-8")
    return zip_path, fmap_path


def write_metadata_readme(output_path: str | Path, metadata: dict[str, Any]) -> Path:
    path = Path(output_path).with_name("README_OUTPUT.txt")
    lines = [
        "hedge-features output package",
        "",
        "Dataset attributions:",
    ]
    for dataset in metadata.get("datasets", []):
        name = dataset.get("name", "unknown")
        attribution = dataset.get("attribution") or "No attribution string provided."
        license_name = dataset.get("license") or "unknown"
        lines.append(f"- {name}: {attribution} (License: {license_name})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def copy_to_cache(src_path: str | Path, cache_dir: str | Path, key: str | None = None) -> Path:
    src = Path(src_path)
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix
    name = key or src.stem
    dst = cache_root / f"{name}{suffix}"
    shutil.copy2(src, dst)
    return dst
