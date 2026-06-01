"""Streamlit upload handling for geospatial input layers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from hedge_features.exceptions import InputValidationError
from hedge_features.utils import sha1_bytes


STANDALONE_UPLOAD_EXTENSIONS = {".zip", ".gpkg", ".geojson", ".json"}
SHAPEFILE_COMPONENT_EXTENSIONS = {".shp", ".shx", ".dbf", ".prj", ".cpg"}
ACCEPTED_UPLOAD_EXTENSIONS = sorted(
    ext.removeprefix(".")
    for ext in STANDALONE_UPLOAD_EXTENSIONS | SHAPEFILE_COMPONENT_EXTENSIONS
)


class UploadedFileLike(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


def normalize_uploaded_files(
    uploaded_files: UploadedFileLike | Iterable[UploadedFileLike] | None,
) -> list[UploadedFileLike]:
    """Return Streamlit's single- or multi-file upload value as a list."""
    if uploaded_files is None:
        return []
    if hasattr(uploaded_files, "getvalue"):
        return [uploaded_files]
    return list(uploaded_files)


def uploaded_files_fingerprint(
    uploaded_files: UploadedFileLike | Iterable[UploadedFileLike],
    *,
    length: int = 20,
) -> str:
    """Hash uploaded names and content without depending on picker order."""
    files = normalize_uploaded_files(uploaded_files)
    parts: list[bytes] = []
    for upload in sorted(files, key=lambda item: Path(item.name).name.casefold()):
        name = Path(upload.name).name.casefold().encode("utf-8")
        raw = upload.getvalue()
        parts.extend(
            [
                len(name).to_bytes(8, "big"),
                name,
                len(raw).to_bytes(8, "big"),
                raw,
            ]
        )
    return sha1_bytes(b"".join(parts), length=length)


def save_uploaded_geodata(
    uploaded_files: UploadedFileLike | Iterable[UploadedFileLike],
    directory: str | Path,
) -> Path:
    """Persist one uploaded geospatial dataset and return its entry-point path."""
    files = normalize_uploaded_files(uploaded_files)
    if not files:
        raise InputValidationError("Upload a hedgerow layer to begin.")

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    named_files = [(upload, Path(upload.name).name) for upload in files]
    names = [name.casefold() for _, name in named_files]
    if len(names) != len(set(names)):
        raise InputValidationError("Upload contains duplicate filenames.")

    if len(named_files) == 1:
        upload, name = named_files[0]
        if Path(name).suffix.lower() in STANDALONE_UPLOAD_EXTENSIONS:
            path = destination / name
            path.write_bytes(upload.getvalue())
            return path

    suffixes = {Path(name).suffix.lower() for _, name in named_files}
    standalone = suffixes & STANDALONE_UPLOAD_EXTENSIONS
    if standalone:
        raise InputValidationError(
            "Upload one GeoPackage, GeoJSON, or zipped shapefile at a time. "
            "Only direct shapefile components should be selected together."
        )

    unsupported = suffixes - SHAPEFILE_COMPONENT_EXTENSIONS
    if unsupported:
        raise InputValidationError(
            f"Unsupported uploaded file type(s): {', '.join(sorted(unsupported))}"
        )

    shp_files = [(upload, name) for upload, name in named_files if Path(name).suffix.lower() == ".shp"]
    if len(shp_files) != 1:
        raise InputValidationError(
            "Direct shapefile upload requires exactly one .shp file together with its "
            "matching .shx and .dbf files. Alternatively, upload a zipped shapefile."
        )

    _, shp_name = shp_files[0]
    stem = Path(shp_name).stem
    mismatched = [name for _, name in named_files if Path(name).stem.casefold() != stem.casefold()]
    if mismatched:
        raise InputValidationError(
            "Direct shapefile components must have the same filename stem. "
            f"Files not matching {shp_name}: {', '.join(sorted(mismatched))}"
        )

    by_suffix = {Path(name).suffix.lower(): upload for upload, name in named_files}
    missing = [ext for ext in (".shx", ".dbf") if ext not in by_suffix]
    if missing:
        raise InputValidationError(
            "Direct shapefile upload is incomplete. "
            f"Missing for {shp_name}: {', '.join(missing)}. "
            "Select the .shp, .shx, and .dbf files together, or upload a zipped shapefile."
        )

    for ext, upload in by_suffix.items():
        (destination / f"{stem}{ext}").write_bytes(upload.getvalue())
    return destination / f"{stem}.shp"
