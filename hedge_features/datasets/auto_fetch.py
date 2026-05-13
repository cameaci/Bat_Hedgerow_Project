from __future__ import annotations

import json
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

from ..deps import require_geopandas, require_rasterio
from ..utils import sha1_text
from .registry import DatasetRegistry


DEFAULT_CACHE_DIRNAME = ".hedge_features_cache"
PC_STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
OSM_ODBL_ATTRIBUTION = "Contains OpenStreetMap data © OpenStreetMap contributors (ODbL)."
DEFAULT_OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


class AutoDataFetcher:
    """AOI-aware automatic acquisition of open datasets for enrichment."""

    def __init__(
        self,
        *,
        registry: DatasetRegistry,
        profile_datasets: dict[str, Any] | None,
        hedges_gdf,
        working_crs: str,
        cache_dir: str | Path | None = None,
        max_buffer_m: float = 1000.0,
        credentials: dict[str, str] | None = None,
        allow_live_fetch: bool = True,
    ) -> None:
        self.registry = registry
        self.profile_datasets = profile_datasets or {}
        self.hedges_gdf = hedges_gdf
        self.working_crs = working_crs
        self.max_buffer_m = float(max_buffer_m)
        self.credentials = dict(credentials or {})
        self.allow_live_fetch = bool(allow_live_fetch)
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / DEFAULT_CACHE_DIRNAME
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._bbox_wgs84 = self._compute_bbox_wgs84(self.max_buffer_m)

    @property
    def bbox_wgs84(self) -> tuple[float, float, float, float]:
        return self._bbox_wgs84

    def _compute_bbox_wgs84(self, buffer_m: float) -> tuple[float, float, float, float]:
        gpd = require_geopandas()
        extent_poly = self.hedges_gdf.unary_union.buffer(float(buffer_m))
        extent_gdf = gpd.GeoDataFrame({"_": [1]}, geometry=[extent_poly], crs=self.working_crs).to_crs("EPSG:4326")
        minx, miny, maxx, maxy = extent_gdf.total_bounds
        # Clamp to valid lat/lon to avoid provider errors.
        minx = max(-180.0, float(minx))
        maxx = min(180.0, float(maxx))
        miny = max(-90.0, float(miny))
        maxy = min(90.0, float(maxy))
        return (minx, miny, maxx, maxy)

    def ensure_dataset(self, dataset_name: str) -> tuple[str | None, list[str]]:
        """Return local path (or remote raster URL if supported) after automatic acquisition."""
        notes: list[str] = []
        ref = self.registry.resolve(dataset_name)
        if ref.path and Path(ref.path).exists():
            return ref.path, notes

        cfg = self.profile_datasets.get(dataset_name, {}) or {}
        provider = cfg.get("auto_provider") or {}
        provider_type = str(provider.get("type", "")).strip()
        if not provider_type:
            return None, [f"No auto_provider configured for dataset '{dataset_name}'."]

        try:
            if provider_type == "osm_overpass_lines":
                path, meta = self._fetch_osm_lines(dataset_name, provider)
                self.registry.set_runtime_resolution(
                    dataset_name,
                    path=str(path),
                    mode="on_demand",
                    license="ODbL 1.0",
                    attribution=OSM_ODBL_ATTRIBUTION,
                    version=meta.get("version"),
                    metadata=meta,
                )
                return str(path), notes

            if provider_type == "osm_overpass_features":
                path, meta = self._fetch_osm_features(dataset_name, provider)
                self.registry.set_runtime_resolution(
                    dataset_name,
                    path=str(path),
                    mode="on_demand",
                    license="ODbL 1.0",
                    attribution=OSM_ODBL_ATTRIBUTION,
                    version=meta.get("version"),
                    metadata=meta,
                )
                return str(path), notes

            if provider_type == "arcgis_feature_layer":
                path, meta = self._fetch_arcgis_feature_layer(dataset_name, provider)
                self.registry.set_runtime_resolution(
                    dataset_name,
                    path=str(path),
                    mode="on_demand",
                    version=meta.get("version"),
                    metadata=meta,
                )
                return str(path), notes

            if provider_type == "pc_stac_raster":
                path, meta = self._fetch_pc_stac_raster(dataset_name, provider)
                self.registry.set_runtime_resolution(
                    dataset_name,
                    path=str(path),
                    mode="on_demand",
                    version=meta.get("version"),
                    metadata=meta,
                )
                return str(path), notes

            if provider_type == "unsupported":
                if dataset_name == "viirs_nightlights" and (
                    self.credentials.get("earthdata_token")
                    or (self.credentials.get("eog_username") and self.credentials.get("eog_password"))
                ):
                    notes.append(
                        "VIIRS credentials were provided, but authenticated VIIRS download is not implemented in this build yet; using fallback behavior if configured."
                    )
                notes.append(
                    f"Dataset '{dataset_name}' has no anonymous automatic source in this build ({provider.get('reason', 'unsupported')})."
                )
                return None, notes

            notes.append(f"Unknown auto_provider '{provider_type}' for dataset '{dataset_name}'.")
            return None, notes
        except Exception as exc:
            notes.append(f"Automatic acquisition failed for '{dataset_name}': {exc}")
            return None, notes

    def _provider_bbox_wgs84(self, provider: dict[str, Any]) -> tuple[float, float, float, float]:
        aoi_buffer_m = float(provider.get("aoi_buffer_m", self.max_buffer_m))
        if aoi_buffer_m == self.max_buffer_m:
            return self.bbox_wgs84
        return self._compute_bbox_wgs84(aoi_buffer_m)

    def _cache_subdir(
        self,
        dataset_name: str,
        provider_type: str,
        extra_key: str,
        bbox_wgs84: tuple[float, float, float, float],
    ) -> Path:
        minx, miny, maxx, maxy = bbox_wgs84
        bbox_key = f"{minx:.5f}_{miny:.5f}_{maxx:.5f}_{maxy:.5f}"
        key = sha1_text(f"{dataset_name}|{provider_type}|{bbox_key}|{extra_key}", length=20)
        path = self.cache_dir / dataset_name / key
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _fetch_osm_lines(self, dataset_name: str, provider: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
        gpd = require_geopandas()
        try:
            from shapely.geometry import LineString
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("shapely is required for OSM line parsing") from exc

        overpass_urls = provider.get("urls") or provider.get("url") or DEFAULT_OVERPASS_URLS
        query_kind = provider.get("query_kind", "roads")
        timeout_s = int(provider.get("timeout_s", 120))
        bbox_wgs84 = self._provider_bbox_wgs84(provider)
        ql = self._build_overpass_ql(query_kind, bbox_wgs84=bbox_wgs84, timeout_s=timeout_s)
        cache_dir = self._cache_subdir(dataset_name, "osm_overpass_lines", query_kind, bbox_wgs84)
        out_path = cache_dir / f"{dataset_name}.gpkg"
        if out_path.exists():
            return out_path, {
                "provider": "osm_overpass_lines",
                "provider_url": overpass_urls[0] if isinstance(overpass_urls, list) else overpass_urls,
                "query_kind": query_kind,
                "cache_hit": True,
                "bbox_wgs84": list(bbox_wgs84),
                "version": "live-overpass",
            }
        if not self.allow_live_fetch:
            raise RuntimeError(
                f"Frozen datasets mode is enabled and no cached snapshot exists for '{dataset_name}'."
            )

        payload, used_overpass_url = self._overpass_query(overpass_urls, ql, timeout_s=timeout_s)

        records: list[dict[str, Any]] = []
        for elem in payload.get("elements", []):
            if elem.get("type") != "way":
                continue
            coords = elem.get("geometry") or []
            if len(coords) < 2:
                continue
            line = LineString([(pt["lon"], pt["lat"]) for pt in coords])
            if line.is_empty or len(line.coords) < 2:
                continue
            tags = elem.get("tags", {}) or {}
            records.append(
                {
                    "osm_id": int(elem.get("id")),
                    "highway": tags.get("highway"),
                    "waterway": tags.get("waterway"),
                    "name": tags.get("name"),
                    "geometry": line,
                }
            )

        if records:
            out_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
        else:
            out_gdf = gpd.GeoDataFrame(
                {"osm_id": [], "highway": [], "waterway": [], "name": []},
                geometry=gpd.GeoSeries([], crs="EPSG:4326"),
                crs="EPSG:4326",
            )
        out_gdf = out_gdf.to_crs(self.working_crs)
        out_gdf.to_file(out_path, driver="GPKG")
        return out_path, {
            "provider": "osm_overpass_lines",
            "provider_url": used_overpass_url,
            "query_kind": query_kind,
            "cache_hit": False,
            "feature_count": int(len(out_gdf)),
            "bbox_wgs84": list(bbox_wgs84),
            "version": "live-overpass",
        }

    def _fetch_osm_features(self, dataset_name: str, provider: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
        gpd = require_geopandas()
        try:
            from shapely.geometry import LineString, Point, Polygon
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("shapely is required for OSM feature parsing") from exc

        overpass_urls = provider.get("urls") or provider.get("url") or DEFAULT_OVERPASS_URLS
        query_kind = provider.get("query_kind", "buildings")
        timeout_s = int(provider.get("timeout_s", 120))
        bbox_wgs84 = self._provider_bbox_wgs84(provider)
        ql = self._build_overpass_ql(query_kind, bbox_wgs84=bbox_wgs84, timeout_s=timeout_s)
        cache_dir = self._cache_subdir(dataset_name, "osm_overpass_features", query_kind, bbox_wgs84)
        out_path = cache_dir / f"{dataset_name}.gpkg"
        if out_path.exists():
            return out_path, {
                "provider": "osm_overpass_features",
                "provider_url": overpass_urls[0] if isinstance(overpass_urls, list) else overpass_urls,
                "query_kind": query_kind,
                "cache_hit": True,
                "bbox_wgs84": list(bbox_wgs84),
                "version": "live-overpass",
            }
        if not self.allow_live_fetch:
            raise RuntimeError(
                f"Frozen datasets mode is enabled and no cached snapshot exists for '{dataset_name}'."
            )

        payload, used_overpass_url = self._overpass_query(overpass_urls, ql, timeout_s=timeout_s)

        records: list[dict[str, Any]] = []
        for elem in payload.get("elements", []):
            tags = elem.get("tags", {}) or {}
            geom = None
            if elem.get("type") == "node":
                if "lat" in elem and "lon" in elem:
                    geom = Point(float(elem["lon"]), float(elem["lat"]))
            elif elem.get("type") in {"way", "relation"}:
                coords = elem.get("geometry") or []
                if len(coords) >= 1:
                    pts = [(float(pt["lon"]), float(pt["lat"])) for pt in coords]
                    if len(pts) >= 4 and pts[0] == pts[-1]:
                        try:
                            poly = Polygon(pts)
                            geom = poly.representative_point() if poly.is_valid else Point(pts[0])
                        except Exception:
                            geom = Point(pts[0])
                    elif len(pts) >= 2:
                        try:
                            line = LineString(pts)
                            geom = line.interpolate(0.5, normalized=True)
                        except Exception:
                            geom = Point(pts[0])
                    else:
                        geom = Point(pts[0])
            if geom is None or geom.is_empty:
                continue
            feature_class, subtype = _classify_osm_feature(query_kind, tags)
            if feature_class is None:
                continue
            records.append(
                {
                    "osm_id": int(elem.get("id")),
                    "osm_type": str(elem.get("type")),
                    "feature_class": feature_class,
                    "subtype": subtype,
                    "name": tags.get("name"),
                    "geometry": geom,
                }
            )

        if records:
            out_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
        else:
            out_gdf = gpd.GeoDataFrame(
                {"osm_id": [], "osm_type": [], "feature_class": [], "subtype": [], "name": []},
                geometry=gpd.GeoSeries([], crs="EPSG:4326"),
                crs="EPSG:4326",
            )
        out_gdf = out_gdf.to_crs(self.working_crs)
        out_gdf.to_file(out_path, driver="GPKG")
        return out_path, {
            "provider": "osm_overpass_features",
            "provider_url": used_overpass_url,
            "query_kind": query_kind,
            "cache_hit": False,
            "feature_count": int(len(out_gdf)),
            "bbox_wgs84": list(bbox_wgs84),
            "version": "live-overpass",
        }

    def _build_overpass_ql(
        self,
        query_kind: str,
        *,
        bbox_wgs84: tuple[float, float, float, float],
        timeout_s: int = 120,
    ) -> str:
        west, south, east, north = bbox_wgs84
        bbox = f"({south},{west},{north},{east})"
        if query_kind == "roads":
            roads_filter = provider_pattern_from_list(
                [
                    "motorway",
                    "trunk",
                    "primary",
                    "secondary",
                    "tertiary",
                    "unclassified",
                    "residential",
                    "service",
                    "living_street",
                    "road",
                    "motorway_link",
                    "trunk_link",
                    "primary_link",
                    "secondary_link",
                    "tertiary_link",
                ]
            )
            selector = f'way["highway"~"{roads_filter}"]{bbox};'
        elif query_kind == "waterways":
            water_filter = provider_pattern_from_list(["river", "stream", "drain", "ditch", "canal"])
            selector = f'way["waterway"~"{water_filter}"]{bbox};'
        elif query_kind == "buildings":
            selector = f'way["building"]{bbox};'
        elif query_kind == "structures_roost":
            selector = (
                f'way["bridge"]{bbox};'
                f'way["tunnel"]{bbox};'
                f'node["natural"="cave_entrance"]{bbox};'
                f'way["natural"="cave_entrance"]{bbox};'
                f'node["man_made"="adit"]{bbox};'
                f'way["man_made"="adit"]{bbox};'
                f'node["mine"]{bbox};'
                f'way["mine"]{bbox};'
            )
        else:
            raise ValueError(f"Unsupported OSM query_kind: {query_kind}")
        return f"[out:json][timeout:{timeout_s}];({selector});out body geom;"

    def _overpass_query(self, overpass_urls, ql: str, *, timeout_s: int) -> tuple[dict[str, Any], str]:
        urls = overpass_urls if isinstance(overpass_urls, list) else [overpass_urls]
        last_exc: Exception | None = None
        for url in urls:
            try:
                data = urllib.parse.urlencode({"data": ql}).encode("utf-8")
                req = urllib.request.Request(url, data=data, headers={"User-Agent": "hedge-features/0.1"})
                with urllib.request.urlopen(req, timeout=timeout_s + 30) as resp:
                    return json.load(resp), str(url)
            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code in {429, 502, 503, 504}:
                    continue
                raise
            except Exception as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Overpass query failed without a specific exception.")

    def _fetch_arcgis_feature_layer(self, dataset_name: str, provider: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
        gpd = require_geopandas()
        service_url = str(provider.get("service_url", "")).strip()
        if not service_url:
            raise ValueError(f"ArcGIS provider for {dataset_name} requires service_url.")
        page_size = int(provider.get("page_size", 2000))
        timeout_s = int(provider.get("timeout_s", 120))
        bbox_wgs84 = self._provider_bbox_wgs84(provider)
        cache_dir = self._cache_subdir(dataset_name, "arcgis_feature_layer", service_url, bbox_wgs84)
        out_path = cache_dir / f"{dataset_name}.gpkg"
        if out_path.exists():
            return out_path, {
                "provider": "arcgis_feature_layer",
                "service_url": service_url,
                "cache_hit": True,
                "bbox_wgs84": list(bbox_wgs84),
                "version": "live-service",
            }
        if not self.allow_live_fetch:
            raise RuntimeError(
                f"Frozen datasets mode is enabled and no cached snapshot exists for '{dataset_name}'."
            )

        features: list[dict[str, Any]] = []
        offset = 0
        max_pages = int(provider.get("max_pages", 50))
        for _ in range(max_pages):
            params = {
                "f": "geojson",
                "where": "1=1",
                "geometry": ",".join(str(v) for v in bbox_wgs84),
                "geometryType": "esriGeometryEnvelope",
                "inSR": "4326",
                "outSR": "4326",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "*",
                "resultOffset": str(offset),
                "resultRecordCount": str(page_size),
                "returnGeometry": "true",
            }
            url = f"{service_url}/query?{urllib.parse.urlencode(params)}"
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                page = json.load(resp)
            page_features = page.get("features", []) or []
            features.extend(page_features)
            if len(page_features) < page_size:
                break
            offset += len(page_features)

        if features:
            out_gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        else:
            out_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")
        out_gdf = out_gdf.to_crs(self.working_crs)
        out_gdf.to_file(out_path, driver="GPKG")
        return out_path, {
            "provider": "arcgis_feature_layer",
            "service_url": service_url,
            "cache_hit": False,
            "feature_count": int(len(out_gdf)),
            "bbox_wgs84": list(bbox_wgs84),
            "version": "live-service",
        }

    def _fetch_pc_stac_raster(self, dataset_name: str, provider: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
        rasterio = require_rasterio()
        collection = str(provider.get("collection", "")).strip()
        asset_key = str(provider.get("asset", "")).strip()
        if not collection or not asset_key:
            raise ValueError(f"pc_stac_raster provider for {dataset_name} requires collection and asset.")

        bbox_wgs84 = self._provider_bbox_wgs84(provider)
        cache_dir = self._cache_subdir(dataset_name, "pc_stac_raster", f"{collection}|{asset_key}", bbox_wgs84)
        out_path = cache_dir / f"{dataset_name}.tif"
        manifest_path = cache_dir / "manifest.json"
        if out_path.exists():
            meta = {}
            if manifest_path.exists():
                try:
                    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            meta.update({"cache_hit": True})
            return out_path, meta or {
                "provider": "pc_stac_raster",
                "collection": collection,
                "asset": asset_key,
                "bbox_wgs84": list(bbox_wgs84),
                "version": "live-stac",
                "cache_hit": True,
            }
        if not self.allow_live_fetch:
            raise RuntimeError(
                f"Frozen datasets mode is enabled and no cached snapshot exists for '{dataset_name}'."
            )

        items = self._pc_stac_search(collection=collection, bbox=bbox_wgs84, limit=int(provider.get("limit", 100)))
        if not items:
            raise RuntimeError(f"No STAC items found for collection '{collection}' in AOI.")
        hrefs: list[str] = []
        item_ids: list[str] = []
        for item in items:
            assets = item.get("assets", {}) or {}
            if asset_key not in assets:
                continue
            href = assets[asset_key].get("href")
            if href:
                hrefs.append(self._pc_sign_href(str(href)))
                item_ids.append(str(item.get("id", "")))
        if not hrefs:
            raise RuntimeError(f"No asset '{asset_key}' found in STAC items for collection '{collection}'.")

        self._merge_and_clip_remote_rasters(hrefs, out_path, bbox_wgs84=bbox_wgs84)
        meta = {
            "provider": "pc_stac_raster",
            "stac_url": PC_STAC_SEARCH_URL,
            "collection": collection,
            "asset": asset_key,
            "item_count": len(hrefs),
            "item_ids": item_ids,
            "bbox_wgs84": list(bbox_wgs84),
            "version": "live-stac",
            "cache_hit": False,
        }
        manifest_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return out_path, meta

    def _pc_stac_search(self, *, collection: str, bbox: tuple[float, float, float, float], limit: int) -> list[dict[str, Any]]:
        body = json.dumps({"collections": [collection], "bbox": list(bbox), "limit": int(limit)}).encode("utf-8")
        req = urllib.request.Request(
            PC_STAC_SEARCH_URL,
            data=body,
            headers={"Content-Type": "application/json", "User-Agent": "hedge-features/0.1"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.load(resp)
        return list(payload.get("features", []) or [])

    def _pc_sign_href(self, href: str) -> str:
        url = "https://planetarycomputer.microsoft.com/api/sas/v1/sign?href=" + urllib.parse.quote(href, safe="")
        with urllib.request.urlopen(url, timeout=120) as resp:
            payload = json.load(resp)
        signed = payload.get("href")
        if not signed:
            raise RuntimeError("Planetary Computer signing response missing href.")
        return str(signed)

    def _merge_and_clip_remote_rasters(
        self,
        hrefs: list[str],
        out_path: Path,
        *,
        bbox_wgs84: tuple[float, float, float, float],
    ) -> None:
        rasterio = require_rasterio()
        from rasterio.merge import merge
        from rasterio.warp import transform_bounds

        sources = [rasterio.open(href) for href in hrefs]
        try:
            first = sources[0]
            clip_bounds = (
                bbox_wgs84
                if str(first.crs) in {"EPSG:4326", "OGC:CRS84"}
                else transform_bounds("EPSG:4326", first.crs, *bbox_wgs84, densify_pts=21)
            )
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
            if profile.get("dtype") is None:
                profile["dtype"] = str(mosaic.dtype)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(mosaic)
        finally:
            for src in sources:
                try:
                    src.close()
                except Exception:
                    pass


def provider_pattern_from_list(values: list[str]) -> str:
    return "|".join(v.replace("|", "") for v in values)


def _classify_osm_feature(query_kind: str, tags: dict[str, Any]) -> tuple[str | None, str | None]:
    if query_kind == "buildings":
        b = tags.get("building")
        if not b:
            return None, None
        return "building", str(b)
    if query_kind == "structures_roost":
        bridge = tags.get("bridge")
        if bridge and str(bridge).lower() not in {"no", "false", "0"}:
            return "bridge", str(bridge)
        tunnel = tags.get("tunnel")
        if tunnel and str(tunnel).lower() not in {"no", "false", "0"}:
            return "tunnel", str(tunnel)
        if str(tags.get("natural", "")).lower() == "cave_entrance":
            return "cave", "cave_entrance"
        if str(tags.get("man_made", "")).lower() == "adit":
            return "cave", "adit"
        if tags.get("mine") is not None:
            return "cave", f"mine:{tags.get('mine')}"
    return None, None
