"""Data resolution layer for England sources.

Resolution order for every dataset:
  1. a user pre-downloaded local file under ``data/<subdir>/`` (clipped to the AOI), then
  2. a live anonymous API (OSM Overpass / ArcGIS / Planetary Computer STAC), then
  3. nothing -> the caller degrades gracefully (precautionary default + low confidence).

Large national rasters (EA LiDAR, night-lights) and GB-wide vectors (OS Open Rivers/Roads,
CROME) are never loaded whole: they are windowed/clipped to the hedgerow area of interest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hedge_features.deps import require_geopandas, require_rasterio
from hedge_features.datasets.auto_fetch import AutoDataFetcher
from hedge_features.datasets.registry import DatasetRegistry
from hedge_features.utils import sha1_text

from . import config


class DataResolver:
    """Resolve England datasets to local clipped files or live-fetched copies."""

    def __init__(
        self,
        hedges_gdf,
        *,
        data_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        allow_live_fetch: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir else config.DATA_DIR
        self.cache_dir = Path(cache_dir) if cache_dir else config.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.allow_live_fetch = bool(allow_live_fetch)
        self.notes: list[str] = []
        self.status: dict[str, str] = {}

        self.registry = DatasetRegistry(config.ENGLAND_DATASETS, cache_dir=self.cache_dir)
        self.fetcher = AutoDataFetcher(
            registry=self.registry,
            profile_datasets=config.ENGLAND_DATASETS,
            hedges_gdf=hedges_gdf,
            working_crs=config.WORKING_CRS,
            cache_dir=self.cache_dir,
            max_buffer_m=config.AOI_BUFFER_M,
            allow_live_fetch=self.allow_live_fetch,
        )
        self._aoi_box = self._compute_aoi_box(hedges_gdf)
        self._vector_cache: dict[str, Any] = {}
        self._raster_cache: dict[str, str | None] = {}

    # -- AOI -------------------------------------------------------------------------
    def _compute_aoi_box(self, hedges_gdf):
        from shapely.geometry import box

        geom = hedges_gdf.geometry
        union = geom.union_all() if hasattr(geom, "union_all") else geom.unary_union
        minx, miny, maxx, maxy = union.buffer(config.AOI_BUFFER_M).bounds
        return box(minx, miny, maxx, maxy)

    @property
    def aoi_box(self):
        return self._aoi_box

    def _local_files(self, name: str, exts: tuple[str, ...]) -> list[Path]:
        subdir = config.LOCAL_DATA_SUBDIRS.get(name)
        if not subdir:
            return []
        folder = self.data_dir / subdir
        if not folder.exists():
            return []
        files: list[Path] = []
        for ext in exts:
            files.extend(sorted(folder.rglob(f"*{ext}")))
        # Avoid picking up shapefile sidecars or our own cache outputs.
        return [f for f in files if f.is_file()]

    # -- Vector ----------------------------------------------------------------------
    def get_vector(self, name: str):
        """Return a GeoDataFrame (working CRS, clipped to AOI) or None."""
        if name in self._vector_cache:
            return self._vector_cache[name]
        gdf = self._resolve_vector(name)
        self._vector_cache[name] = gdf
        return gdf

    def _resolve_vector(self, name: str):
        gpd = require_geopandas()
        # 1) local pre-download
        local = self._local_files(name, config.VECTOR_EXTS)
        if local:
            frames = []
            for path in local:
                try:
                    part = self._read_vector_clipped(path)
                    if part is not None and not part.empty:
                        frames.append(part)
                except Exception as exc:  # noqa: BLE001
                    self.notes.append(f"Could not read local '{name}' file {path.name}: {exc}")
            if frames:
                merged = gpd.GeoDataFrame(
                    __import__("pandas").concat(frames, ignore_index=True), crs=config.WORKING_CRS
                )
                self.status[name] = f"local ({len(merged)} features)"
                return merged
            self.status[name] = "local (empty in AOI)"
            return None
        # 2) live API
        cfg = config.ENGLAND_DATASETS.get(name, {})
        if cfg.get("local_only") or not self.allow_live_fetch:
            self.status[name] = "absent (local-only, not provided)" if cfg.get("local_only") else "absent (live disabled)"
            return None
        path, notes = self.fetcher.ensure_dataset(name)
        self.notes.extend(notes)
        if not path:
            self.status[name] = "live: no data"
            return None
        try:
            gdf = gpd.read_file(path)
            if gdf.crs is not None and str(gdf.crs) != str(config.WORKING_CRS):
                gdf = gdf.to_crs(config.WORKING_CRS)
            self.status[name] = f"live ({len(gdf)} features)"
            return gdf if not gdf.empty else None
        except Exception as exc:  # noqa: BLE001
            self.notes.append(f"Failed to read fetched '{name}': {exc}")
            self.status[name] = "live: read error"
            return None

    def _read_vector_clipped(self, path: Path):
        gpd = require_geopandas()
        aoi = gpd.GeoSeries([self._aoi_box], crs=config.WORKING_CRS)
        try:
            gdf = gpd.read_file(path, bbox=aoi)
        except Exception:
            # Fallback: read whole then spatial filter (slower, but robust).
            gdf = gpd.read_file(path)
            if gdf.crs is not None and str(gdf.crs) != str(config.WORKING_CRS):
                gdf = gdf.to_crs(config.WORKING_CRS)
            gdf = gdf[gdf.geometry.intersects(self._aoi_box)]
            return gdf
        if gdf.crs is not None and str(gdf.crs) != str(config.WORKING_CRS):
            gdf = gdf.to_crs(config.WORKING_CRS)
        return gdf

    # -- Raster ----------------------------------------------------------------------
    def get_raster_path(self, name: str) -> str | None:
        """Return a path to a single AOI-clipped raster (working CRS) or None."""
        if name in self._raster_cache:
            return self._raster_cache[name]
        path = self._resolve_raster(name)
        self._raster_cache[name] = path
        return path

    def _resolve_raster(self, name: str) -> str | None:
        # 1) local mosaic
        local = self._local_files(name, config.RASTER_EXTS)
        if local:
            try:
                out = self._mosaic_clip_rasters(name, local)
                if out is not None:
                    self.status[name] = f"local ({len(local)} tile(s) clipped)"
                    return str(out)
            except Exception as exc:  # noqa: BLE001
                self.notes.append(f"Could not mosaic local '{name}' rasters: {exc}")
            self.status[name] = "local (no AOI overlap)"
            return None
        # 2) live: EA WCS (LiDAR) or STAC raster
        cfg = config.ENGLAND_DATASETS.get(name, {})
        provider = cfg.get("auto_provider") or {}
        if not self.allow_live_fetch:
            self.status[name] = "absent (live disabled)"
            return None
        if provider.get("type") == "ea_wcs_raster":
            out = self._fetch_wcs_raster(name, provider)
            if out is not None:
                self.status[name] = "live (WCS)"
                return str(out)
            self.status[name] = "WCS: no data"
            return None
        if cfg.get("local_only"):
            self.status[name] = "absent (local-only, not provided)"
            return None
        path, notes = self.fetcher.ensure_dataset(name)
        self.notes.extend(notes)
        if path:
            self.status[name] = "live raster"
            return path
        self.status[name] = "live: no data"
        return None

    def _fetch_wcs_raster(self, name: str, provider: dict) -> Path | None:
        """Fetch an AOI subset from an OGC WCS 2.0.1 endpoint (EA LiDAR DTM/DSM)."""
        import urllib.request

        rasterio = require_rasterio()
        wcs_url = provider.get("wcs_url")
        if not wcs_url:
            return None
        key = sha1_text(f"{name}|{self._aoi_box.bounds}", length=16)
        out_path = self.cache_dir / f"{name}_wcs_{key}.tif"
        if out_path.exists():
            return out_path
        try:
            coverage_id = self._wcs_coverage_id(wcs_url, provider.get("coverage_id"))
        except Exception as exc:  # noqa: BLE001
            self.notes.append(f"WCS GetCapabilities failed for '{name}': {exc}")
            return None
        if not coverage_id:
            self.notes.append(f"No WCS coverage id discovered for '{name}'.")
            return None
        bounds = self._aoi_box.bounds  # EPSG:27700
        # Axis labels vary by server; try the common conventions until one yields a raster.
        for axis_e, axis_n in (("E", "N"), ("x", "y"), ("Long", "Lat"), ("i", "j")):
            url = build_wcs_getcoverage_url(wcs_url, coverage_id, bounds, axis_e, axis_n)
            tmp = out_path.with_suffix(".part")
            try:
                urllib.request.urlretrieve(url, tmp)
                with rasterio.open(tmp):
                    pass  # validates it is a readable raster, not an XML error
                tmp.replace(out_path)
                return out_path
            except Exception:  # noqa: BLE001
                try:
                    tmp.unlink()
                except Exception:  # noqa: BLE001
                    pass
                continue
        self.notes.append(f"WCS GetCoverage failed for '{name}' (all axis conventions tried).")
        return None

    def _wcs_coverage_id(self, wcs_url: str, configured: str | None = None) -> str | None:
        import urllib.request
        from xml.etree import ElementTree as ET

        if configured:
            return configured
        cap_url = f"{wcs_url}?service=WCS&version=2.0.1&request=GetCapabilities"
        with urllib.request.urlopen(cap_url, timeout=60) as resp:
            root = ET.fromstring(resp.read())
        for el in root.iter():
            if el.tag.endswith("CoverageId") and el.text and el.text.strip():
                return el.text.strip()
        return None

    def _mosaic_clip_rasters(self, name: str, files: list[Path]) -> Path | None:
        rasterio = require_rasterio()
        from rasterio.merge import merge
        from rasterio.warp import transform_bounds

        key = sha1_text("|".join(sorted(str(f) for f in files)) + str(self._aoi_box.bounds), length=16)
        out_path = self.cache_dir / f"{name}_{key}.tif"
        if out_path.exists():
            return out_path

        aoi_bounds_wgs = self._aoi_box.bounds  # working CRS bounds
        sources = []
        try:
            for path in files:
                try:
                    src = rasterio.open(path)
                except Exception:
                    continue
                # Compute AOI bounds in this raster's CRS to test overlap.
                try:
                    if src.crs is None or str(src.crs) == str(config.WORKING_CRS):
                        clip_bounds = aoi_bounds_wgs
                    else:
                        clip_bounds = transform_bounds(
                            config.WORKING_CRS, src.crs, *aoi_bounds_wgs, densify_pts=21
                        )
                except Exception:
                    clip_bounds = None
                if clip_bounds is not None and not _bounds_intersect(src.bounds, clip_bounds):
                    src.close()
                    continue
                sources.append(src)
            if not sources:
                return None
            first = sources[0]
            if first.crs is None or str(first.crs) == str(config.WORKING_CRS):
                clip_bounds = aoi_bounds_wgs
            else:
                clip_bounds = transform_bounds(config.WORKING_CRS, first.crs, *aoi_bounds_wgs, densify_pts=21)
            mosaic, out_transform = merge(sources, bounds=clip_bounds)
            profile = first.profile.copy()
            profile.update(
                driver="GTiff",
                height=mosaic.shape[1],
                width=mosaic.shape[2],
                transform=out_transform,
                count=mosaic.shape[0],
                compress="LZW",
            )
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(mosaic)
            return out_path
        finally:
            for src in sources:
                try:
                    src.close()
                except Exception:
                    pass

    # -- Metadata --------------------------------------------------------------------
    def metadata_records(self, used_names: list[str]) -> list[dict[str, Any]]:
        return self.registry.to_metadata_records(used_names)


def _bounds_intersect(a, b) -> bool:
    """Return True if two (left, bottom, right, top) bounds overlap."""
    al, ab, ar, at = a
    bl, bb, br, bt = b
    return not (ar < bl or br < al or at < bb or bt < ab)


def build_wcs_getcoverage_url(
    wcs_url: str,
    coverage_id: str,
    bounds,
    axis_e: str = "E",
    axis_n: str = "N",
) -> str:
    """Build an OGC WCS 2.0.1 GetCoverage URL for an AOI subset (native CRS, e.g. EPSG:27700).

    Pure string builder (no network) so it can be unit-tested.
    """
    import urllib.parse

    minx, miny, maxx, maxy = bounds
    params = urllib.parse.urlencode({
        "service": "WCS",
        "version": "2.0.1",
        "request": "GetCoverage",
        "coverageId": coverage_id,
        "format": "image/tiff",
    })
    subset_e = f"subset={axis_e}({minx:.2f},{maxx:.2f})"
    subset_n = f"subset={axis_n}({miny:.2f},{maxy:.2f})"
    return f"{wcs_url}?{params}&{subset_e}&{subset_n}"
