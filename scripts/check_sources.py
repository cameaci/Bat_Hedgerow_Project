#!/usr/bin/env python3
"""Probe every configured England data source one-by-one for a test area of interest.

Reports OK / EMPTY / MISSING / ERROR per source so dead sources can be pruned. Hits the
exact resolution path the app uses (local file discovery, then live API), so a green run
here means the app will get data.

Examples:
    python scripts/check_sources.py
    python scripts/check_sources.py --aoi sample_data/sample_hedgerows_england.gpkg
    python scripts/check_sources.py --bbox 449000 205000 452000 208000 --json
    python scripts/check_sources.py --only worldcover,ne_phi,os_open_rivers
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running from the repo root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hsi import config  # noqa: E402
from hsi.datasets import DataResolver  # noqa: E402

# A small known-good lowland-England test line (EPSG:27700, near Oxford) used by default.
DEFAULT_LINE = [(450000.0, 206000.0), (450600.0, 206050.0)]


def _aoi_gdf_from_bbox(bbox):
    import geopandas as gpd
    from shapely.geometry import LineString

    minx, miny, maxx, maxy = bbox
    line = LineString([(minx + (maxx - minx) * 0.4, miny + (maxy - miny) * 0.5),
                       (minx + (maxx - minx) * 0.6, miny + (maxy - miny) * 0.5)])
    return gpd.GeoDataFrame({"hf_uid": ["probe"]}, geometry=[line], crs=config.WORKING_CRS)


def _aoi_gdf_default():
    import geopandas as gpd
    from shapely.geometry import LineString

    return gpd.GeoDataFrame({"hf_uid": ["probe"]}, geometry=[LineString(DEFAULT_LINE)], crs=config.WORKING_CRS)


def _classify(name: str, count: int | None, has_result: bool, cfg: dict) -> str:
    if has_result and (count is None or count > 0):
        return "OK"
    if has_result and count == 0:
        return "EMPTY"
    if cfg.get("local_only"):
        return "MISSING"
    return "NO_DATA"


def probe(aoi_gdf, only: set[str] | None, allow_live: bool) -> list[dict]:
    resolver = DataResolver(aoi_gdf, allow_live_fetch=allow_live)
    rows: list[dict] = []
    for name, cfg in config.ENGLAND_DATASETS.items():
        if only and name not in only:
            continue
        is_raster = str(cfg.get("source_type", "")).startswith("raster")
        start = time.time()
        count: int | None = None
        has_result = False
        error = ""
        try:
            if is_raster:
                path = resolver.get_raster_path(name)
                has_result = path is not None
            else:
                gdf = resolver.get_vector(name)
                has_result = gdf is not None
                count = len(gdf) if gdf is not None else 0
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        elapsed = round(time.time() - start, 2)
        status = "ERROR" if error else _classify(name, count, has_result, cfg)
        rows.append({
            "source": name,
            "access": "raster" if is_raster else "vector",
            "status": status,
            "count": count if count is not None else "-",
            "elapsed_s": elapsed,
            "note": error or resolver.status.get(name, ""),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY"),
                        help="AOI bbox in EPSG:27700.")
    parser.add_argument("--aoi", type=str, help="Path to a hedgerow layer to use as the AOI.")
    parser.add_argument("--only", type=str, help="Comma-separated subset of dataset names to test.")
    parser.add_argument("--no-live", action="store_true", help="Only test local pre-downloaded data.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    args = parser.parse_args()

    if args.aoi:
        from hsi.ingest import load_hedgerows
        aoi_gdf, _ = load_hedgerows(args.aoi)
    elif args.bbox:
        aoi_gdf = _aoi_gdf_from_bbox(args.bbox)
    else:
        aoi_gdf = _aoi_gdf_default()

    only = {s.strip() for s in args.only.split(",")} if args.only else None
    rows = probe(aoi_gdf, only, allow_live=not args.no_live)

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    name_w = max(len(r["source"]) for r in rows)
    print(f"{'SOURCE':<{name_w}}  {'ACCESS':<6}  {'STATUS':<8}  {'COUNT':>7}  {'TIME':>6}  NOTE")
    print("-" * (name_w + 50))
    for r in rows:
        print(f"{r['source']:<{name_w}}  {r['access']:<6}  {r['status']:<8}  {str(r['count']):>7}  {r['elapsed_s']:>6}  {r['note'][:60]}")
    ok = sum(1 for r in rows if r["status"] == "OK")
    print(f"\n{ok}/{len(rows)} sources returned data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
