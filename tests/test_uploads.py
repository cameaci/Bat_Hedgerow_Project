from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from shapely.geometry import LineString

from hedge_features.exceptions import InputValidationError
from hedge_features.io import read_input_geodata
from hsi.uploads import save_uploaded_geodata, uploaded_files_fingerprint


@dataclass
class FakeUpload:
    name: str
    data: bytes

    def getvalue(self) -> bytes:
        return self.data


def test_saves_direct_shapefile_components_with_shared_stem(tmp_path: Path):
    uploads = [
        FakeUpload("hedges.dbf", b"dbf"),
        FakeUpload("hedges.shp", b"shp"),
        FakeUpload("hedges.prj", b"prj"),
        FakeUpload("hedges.shx", b"shx"),
    ]

    path = save_uploaded_geodata(uploads, tmp_path)

    assert path == tmp_path / "hedges.shp"
    assert path.read_bytes() == b"shp"
    assert (tmp_path / "hedges.shx").read_bytes() == b"shx"
    assert (tmp_path / "hedges.dbf").read_bytes() == b"dbf"
    assert (tmp_path / "hedges.prj").read_bytes() == b"prj"


def test_rejects_incomplete_direct_shapefile_upload(tmp_path: Path):
    with pytest.raises(InputValidationError) as exc:
        save_uploaded_geodata([FakeUpload("hedges.shp", b"shp")], tmp_path)

    assert "Missing for hedges.shp: .shx, .dbf" in str(exc.value)


def test_saved_direct_shapefile_components_are_readable(tmp_path: Path):
    import geopandas as gpd

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_path = source_dir / "hedges.shp"
    gpd.GeoDataFrame(
        {"name": ["test hedge"]},
        geometry=[LineString([(400000, 300000), (400100, 300000)])],
        crs="EPSG:27700",
    ).to_file(source_path)
    uploads = [FakeUpload(path.name, path.read_bytes()) for path in source_dir.iterdir()]

    saved_path = save_uploaded_geodata(uploads, tmp_path / "uploaded")
    gdf, _ = read_input_geodata(saved_path)

    assert gdf["name"].tolist() == ["test hedge"]


def test_upload_fingerprint_does_not_depend_on_picker_order():
    uploads = [FakeUpload("hedges.shp", b"shp"), FakeUpload("hedges.dbf", b"dbf")]

    assert uploaded_files_fingerprint(uploads) == uploaded_files_fingerprint(list(reversed(uploads)))
