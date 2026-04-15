#!/usr/bin/env python3
"""Download NEON AOP RGB mosaic tiles, build a study-area mosaic, and tile for MillionTrees.

Workflow (local or cluster):
  1. Read crown polygons (e.g. combined_crown_polygons.gpkg).
  2. Pick a NEON site (default HARV for Harvard Forest) and flight year (default: latest
     calendar year with any AOP month in the NEON product API).
  3. Download DP3.30010.001 (high-resolution orthorectified camera mosaic) tiles that cover
     the annotation bounds via ``neonutilities.by_tile_aop``.
  4. Merge GeoTIFFs and crop to the union of crown geometries (``rasterio.merge`` +
     ``rasterio.mask``).
  5. Explode MultiPolygons to Polygons, reproject to the raster CRS, then run
     ``deepforest.preprocess.split_raster`` like ``Quebec_Lefebvre.py``.

Environment:
  Optionally set NEON_TOKEN for higher API rate limits (see NEON data API docs).

Examples:
  uv run python data_prep/neon_combined_crowns_prep.py \\
    --gpkg /path/to/combined_crown_polygons.gpkg \\
    --output-dir ./work/neon_combined_crowns \\
    --site HARV

  uv run python data_prep/neon_combined_crowns_prep.py \\
    --gpkg ... --output-dir ... --year 2024 --skip-download \\
    (use after tiles are already under ``output-dir/downloads``)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from deepforest.preprocess import split_raster
from deepforest.utilities import read_file
from rasterio.merge import merge
from rasterio.mask import mask
from shapely.geometry import mapping

RGB_DPID = "DP3.30010.001"


def _api_years_for_site(dpid: str, site: str) -> list[int]:
    """Return sorted calendar years present in NEON product ``availableDataUrls`` for a site."""
    r = requests.get(f"https://data.neonscience.org/api/v0/products/{dpid}",
                     timeout=60)
    r.raise_for_status()
    payload = r.json()["data"]
    for sc in payload["siteCodes"]:
        if sc["siteCode"] == site.upper():
            years: set[int] = set()
            for url in sc["availableDataUrls"]:
                m = re.search(r"/(20\d{2})-", url)
                if m:
                    years.add(int(m.group(1)))
            return sorted(years)
    raise ValueError(f"Site {site} not listed for product {dpid}")


def latest_api_year(dpid: str, site: str) -> int:
    years = _api_years_for_site(dpid, site)
    if not years:
        raise ValueError(f"No AOP months found in API for {site} / {dpid}")
    return years[-1]


def _tile_sample_coords(minx: float, miny: float, maxx: float, maxy: float,
                        pad_m: float) -> tuple[list[float], list[float]]:
    """Build paired easting/northing samples so ``by_tile_aop`` hits every 1 km tile index."""
    minx -= pad_m
    miny -= pad_m
    maxx += pad_m
    maxy += pad_m
    te0 = int(np.floor(minx / 1000) * 1000)
    tn0 = int(np.floor(miny / 1000) * 1000)
    te1 = int(np.floor(maxx / 1000) * 1000)
    tn1 = int(np.floor(maxy / 1000) * 1000)
    easting: list[float] = []
    northing: list[float] = []
    te = te0
    while te <= te1:
        tn = tn0
        while tn <= tn1:
            easting.append(te + 100.0)
            northing.append(tn + 100.0)
            tn += 1000
        te += 1000
    if not easting:
        easting = [(minx + maxx) / 2]
        northing = [(miny + maxy) / 2]
    return easting, northing


def _collect_rgb_tifs(download_root: Path) -> list[str]:
    pattern = str(download_root / "**" / "*.tif")
    paths = [
        p for p in glob.glob(pattern, recursive=True)
        if "manifest" not in p.lower() and "readme" not in p.lower()
    ]
    if not paths:
        raise FileNotFoundError(
            f"No .tif files under {download_root}. "
            "Run without --skip-download or check the NEON download.")
    return paths


def _find_downloaded_rgb_tifs(download_root: Path) -> list[str]:
    """Prefer ``download_root/DP3.30010.001`` (neonutilities layout), else search whole tree."""
    nested = download_root / RGB_DPID
    if nested.is_dir():
        return _collect_rgb_tifs(nested)
    return _collect_rgb_tifs(download_root)


def _merge_and_crop(
    tif_paths: list[str],
    aoi_union,
    aoi_crs,
    dst_path: Path,
) -> Path:
    """Merge rasters, clip to AOI union, write 3-band uint8 GeoTIFF."""
    sources = [rasterio.open(p) for p in tif_paths]
    try:
        # Align vector to first raster CRS
        g_series = gpd.GeoSeries([aoi_union], crs=aoi_crs)
        g_series = g_series.to_crs(sources[0].crs)
        geom = g_series.iloc[0]
        bounds = geom.bounds
        # Merge only the window needed (saves memory)
        mosaic, out_trans = merge(
            sources,
            bounds=(bounds[0], bounds[1], bounds[2], bounds[3]),
        )
        meta = sources[0].meta.copy()
        meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "count": mosaic.shape[0],
            "dtype": mosaic.dtype,
        })
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **meta) as dst:
            dst.write(mosaic)
        # Mask to exact union outline
        with rasterio.open(dst_path) as src:
            out_image, out_transform = mask(src, [mapping(geom)],
                                            crop=True,
                                            nodata=0)
            out_meta = src.meta.copy()
            out_meta.update(
                height=out_image.shape[1],
                width=out_image.shape[2],
                transform=out_transform,
            )
        with rasterio.open(dst_path, "w", **out_meta) as dst:
            dst.write(out_image)
    finally:
        for s in sources:
            s.close()
    return dst_path


def _explode_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Explode MultiPolygons; keep only Polygon rows for DeepForest ``read_file``."""
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf = gdf[gdf.geometry.geom_type == "Polygon"].copy()
    return gdf


def run(
    gpkg: Path,
    output_dir: Path,
    site: str,
    year: int | None,
    patch_size: int,
    pad_m: float,
    skip_download: bool,
    include_provisional: bool,
    source_name: str,
    token: str | None,
) -> pd.DataFrame:
    import neonutilities as nu

    gdf = gpd.read_file(gpkg)
    if gdf.crs is None:
        raise ValueError("GeoPackage has no CRS; cannot align to NEON imagery.")
    gdf = _explode_polygons(gdf)
    if gdf.empty:
        raise ValueError(
            "No polygon geometries left after exploding MultiPolygons.")

    flight_year = year if year is not None else latest_api_year(RGB_DPID, site)
    print(
        f"Using NEON site={site.upper()} year={flight_year} product={RGB_DPID}")

    download_root = output_dir / "downloads"
    if not skip_download:
        download_root.mkdir(parents=True, exist_ok=True)
        # Tile selection uses the same UTM coordinates as NEON's index (zone for this site).
        gdf_rough = gdf.to_crs("EPSG:32619")
        minx, miny, maxx, maxy = gdf_rough.total_bounds
        easting, northing = _tile_sample_coords(minx, miny, maxx, maxy, pad_m)
        nu.by_tile_aop(
            dpid=RGB_DPID,
            site=site.upper(),
            year=str(flight_year),
            easting=easting,
            northing=northing,
            buffer=0,
            savepath=str(download_root),
            check_size=False,
            token=token,
            verbose=True,
            skip_if_exists=True,
            overwrite="no",
            include_provisional=include_provisional,
        )
    else:
        print(
            f"Skipping download; expecting tiles under {download_root / RGB_DPID}"
        )

    tif_paths = _find_downloaded_rgb_tifs(
        download_root if skip_download else download_root / RGB_DPID)
    print(f"Found {len(tif_paths)} GeoTIFF tile(s)")

    with rasterio.open(tif_paths[0]) as src0:
        raster_crs = src0.crs
    gdf_r = gdf.to_crs(raster_crs)
    aoi_union = gdf_r.geometry.union_all()

    mosaic_path = output_dir / f"{site.upper()}_{flight_year}_rgb_mosaic_crop.tif"
    _merge_and_crop(tif_paths, aoi_union, gdf_r.crs, mosaic_path)

    gdf_img = gdf_r.copy()
    gdf_img["image_path"] = mosaic_path.name
    gdf_img["label"] = "Tree"
    annotations = read_file(
        gdf_img,
        root_dir=str(mosaic_path.parent),
        image_path=mosaic_path.name,
        label="Tree",
    )

    tiles_dir = output_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    split_df = split_raster(
        annotations_file=annotations,
        path_to_raster=str(mosaic_path),
        patch_size=patch_size,
        patch_overlap=0.05,
        allow_empty=False,
        save_dir=str(tiles_dir),
    )
    split_df["source"] = source_name
    split_df["existing_split"] = "test"
    out_csv = output_dir / "annotations_tiled.csv"
    split_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(split_df)} rows)")

    meta = {
        "site":
            site.upper(),
        "year":
            flight_year,
        "dpid":
            RGB_DPID,
        "gpkg":
            str(gpkg.resolve()),
        "mosaic":
            str(mosaic_path.resolve()),
        "tiles_dir":
            str(tiles_dir.resolve()),
        "n_tiles":
            len(split_df["image_path"].unique())
            if "image_path" in split_df.columns else None,
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    return split_df


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gpkg",
        type=Path,
        required=True,
        help="Path to combined crown polygons GeoPackage",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Working output directory (downloads, mosaic, tiles)",
    )
    p.add_argument(
        "--site",
        default="HARV",
        help="Four-letter NEON site code (default HARV for Harvard Forest)",
    )
    p.add_argument(
        "--year",
        type=int,
        default=None,
        help=
        "Flight year (default: latest year from NEON API for this site/product)",
    )
    p.add_argument("--patch-size",
                   type=int,
                   default=1024,
                   help="Tile size in pixels")
    p.add_argument(
        "--pad-m",
        type=float,
        default=80.0,
        help="Extra meters around vector bounds for tile selection",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Use already-downloaded tiles under output-dir/downloads",
    )
    p.add_argument(
        "--include-provisional",
        action="store_true",
        help=
        "Include provisional NEON releases in by_tile_aop (needed for some recent years)",
    )
    p.add_argument(
        "--source-name",
        default="NEON combined crowns",
        help="Value for the ``source`` column in tiled annotations",
    )
    args = p.parse_args(argv)

    token = os.environ.get("NEON_TOKEN")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run(
        gpkg=args.gpkg,
        output_dir=args.output_dir,
        site=args.site,
        year=args.year,
        patch_size=args.patch_size,
        pad_m=args.pad_m,
        skip_download=args.skip_download,
        include_provisional=args.include_provisional,
        source_name=args.source_name,
        token=token,
    )


if __name__ == "__main__":
    main()
