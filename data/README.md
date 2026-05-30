# Local data folder (England)

Drop pre-downloaded national datasets here **once**; the app auto-discovers them and clips
each to your area of interest. Everything below is **optional** — anything missing falls
back to a live API or a documented proxy — but the structural WSP indices (height, width,
gappiness, trees) need EA LiDAR, so that one matters most.

Verify what works for your area any time:

```bash
python scripts/check_sources.py --aoi sample_data/sample_hedgerows_england.gpkg
```

## Folder layout

```
data/
├── lidar/dtm/        EA LiDAR Composite DTM 1 m  (GeoTIFF/ASC tiles)   ← needed for SI1/SI2/SI3/SI5
├── lidar/dsm/        EA LiDAR Composite DSM 1 m  (GeoTIFF/ASC tiles)   ← needed for SI1/SI2/SI3/SI5
├── crome/            RPA Crop Map of England (CROME) shapefile(s)       (refines SI4)
├── os_open_rivers/   OS Open Rivers (GeoPackage/SHP)                    (SI7 + water; else OSM)
├── os_open_roads/    OS Open Roads (GeoPackage/SHP)                     (road quietness; else OSM)
├── nightlights/      VIIRS VNL annual or Falchi 2016 raster            (darkness; else proxy)
├── ancient_trees/    Woodland Trust Ancient Tree Inventory (points)     (roost refinement)
└── cache/            auto-generated clipped copies (safe to delete)
```

`ne_phi/`, `ne_awi/`, `worldcover/` are fetched live by default; drop local copies in those
folders only if you want to avoid live calls.

## Where to download (all Open Government Licence / open data)

| Folder | Dataset | Source |
|---|---|---|
| `lidar/dtm`, `lidar/dsm` | **EA National LiDAR Programme — LIDAR Composite DTM & DSM 1 m** | https://environment.data.gov.uk/survey — select your area, choose *LIDAR Composite DTM/DSM 2022 1m*, download the tiles. |
| `crome` | **RPA Crop Map of England (CROME)** | https://environment.data.gov.uk — search "Crop Map of England"; download the county shapefile(s) for your AOI. |
| `os_open_rivers` | **OS Open Rivers** | https://osdatahub.os.uk/downloads/open/OpenRivers |
| `os_open_roads` | **OS Open Roads** | https://osdatahub.os.uk/downloads/open/OpenRoads |
| `nightlights` | **VIIRS VNL V2 annual** (recommended) | https://eogdata.mines.edu/products/vnl/ — annual composite, Europe tile; *or* Falchi 2016 World Atlas (GFZ, DOI 10.5880/GFZ.1.4.2016.001). |
| `ancient_trees` | **Woodland Trust Ancient Tree Inventory** | https://ati.woodlandtrust.org.uk — export for your AOI. |

### Sizes / tips
- LiDAR tiles are ~25–50 MB each at 1 m; download only the tiles covering your survey area.
- OS Open Rivers/Roads are GB-wide (a few hundred MB) — the app clips them to your AOI on load.
- VIIRS/Falchi rasters are large (~1–3 GB); only one is needed, and it is optional.
