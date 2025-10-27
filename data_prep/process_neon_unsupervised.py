#!/usr/bin/env python3
"""
Combined script to format NEON unsupervised annotations and download tiles.
This script:
1. Reads unsupervised NEON detection results from CSV files
2. Formats them for TreeBoxes, TreePoints, and TreePolygons datasets
3. Downloads corresponding NEON tiles based on annotations
4. Tiles the downloaded images and creates patch-level annotations
"""

import os
import re
import shutil
import argparse
import glob
from typing import Optional, Tuple, List

import pandas as pd
from shapely.geometry import Point, Polygon
import geopandas as gpd

from deepforest.preprocess import split_raster
from deepforest.utilities import read_file


def format_neon_annotations(csv_glob_pattern: str, output_dir: str) -> Tuple[str, str, str]:
    """
    Format NEON unsupervised annotations into TreeBoxes, TreePoints, and TreePolygons formats.
    
    Args:
        csv_glob_pattern: Glob pattern to find CSV files with NEON detections
        output_dir: Directory to save formatted annotations
        
    Returns:
        Tuple of paths to the three output CSV files (boxes, points, polygons)
    """
    print("=== FORMATTING NEON UNSUPERVISED ANNOTATIONS ===")
    
    # Read the unsupervised NEON annotations
    csvs = glob.glob(csv_glob_pattern)
    if not csvs:
        raise ValueError(f"No CSV files found matching pattern: {csv_glob_pattern}")
        
    print(f"Found {len(csvs)} CSV files to process")
    
    annotations = []
    for csv in csvs[:2]:
        df = pd.read_csv(csv)
        df["source"] = "Weinstein et al. 2018 unsupervised"
        annotations.append(df)
    annotations = pd.concat(annotations, ignore_index=True)

    # Extract siteID and tile_name from the filename
    annotations["siteID"] = annotations["image_path"].apply(lambda x: x.split("_")[1])
    annotations["tile_name"] = annotations["image_path"].apply(lambda x: x.split("/")[2]+".tif")
    
    print(f"Number of sites: {annotations['siteID'].nunique()}")
    print(f"Number of tiles: {annotations['tile_name'].nunique()}")

    # Create metadata column
    annotations["metadata"] = annotations.apply(lambda row: f"{row['siteID']}_{row['tile_name']}", axis=1)

    # How many annotations, sites, and tiles?
    print(f"Number of annotations: {len(annotations)}")
    print(f"Number of sites: {annotations['siteID'].nunique()}")
    print(f"Number of tiles: {annotations['tile_name'].nunique()}")

    # Create points version
    points_annotations = annotations.copy()
    points_annotations["geometry"] = points_annotations.apply(
        lambda row: Point((row["xmin"] + row["xmax"]) / 2, (row["ymin"] + row["ymax"]) / 2), axis=1)

    # Create polygons version
    polygons_annotations = annotations.copy()
    polygons_annotations["geometry"] = polygons_annotations.apply(
        lambda row: Polygon([(row["xmin"], row["ymin"]), (row["xmax"], row["ymin"]), 
                            (row["xmax"], row["ymax"]), (row["xmin"], row["ymax"])]), axis=1)

    # Create box version (same as original but with metadata and source)
    boxes_annotations = annotations.copy()

    os.makedirs(output_dir, exist_ok=True)

    boxes_path = os.path.join(output_dir, "TreeBoxes_neon_unsupervised.csv")
    points_path = os.path.join(output_dir, "TreePoints_neon_unsupervised.csv")
    polygons_path = os.path.join(output_dir, "TreePolygons_neon_unsupervised.csv")
    
    points_annotations = read_file(points_annotations)
    points_annotations["source"] = "Weinstein et al. 2018 unsupervised"
    polygons_annotations = read_file(polygons_annotations)
    polygons_annotations["source"] = "Weinstein et al. 2018 unsupervised"
    boxes_annotations = read_file(boxes_annotations)
    boxes_annotations["source"] = "Weinstein et al. 2018 unsupervised"

    # # read_file forced siteID to siteid change it back
    points_annotations.rename(columns={"siteid": "siteID"}, inplace=True)
    polygons_annotations.rename(columns={"siteid": "siteID"}, inplace=True)
    boxes_annotations.rename(columns={"siteid": "siteID"}, inplace=True)

    # Full image_path
    points_annotations["image_path"] = points_annotations["image_path"].apply(
        lambda x: os.path.join("/orange/ewhite/DeepForest/unsupervised/images/", os.path.basename(x)))
    polygons_annotations["image_path"] = polygons_annotations["image_path"].apply(
        lambda x: os.path.join("/orange/ewhite/DeepForest/unsupervised/images/", os.path.basename(x)))
    boxes_annotations["image_path"] = boxes_annotations["image_path"].apply(
        lambda x: os.path.join("/orange/ewhite/DeepForest/unsupervised/images/", os.path.basename(x)))

    points_annotations.to_csv(points_path, index=False)
    polygons_annotations.to_csv(polygons_path, index=False)
    boxes_annotations.to_csv(boxes_path, index=False)
    
    print(f"Saved formatted annotations:")
    print(f"  TreeBoxes: {boxes_path}")
    print(f"  TreePoints: {points_path}")
    print(f"  TreePolygons: {polygons_path}")
    
    return boxes_path, points_path, polygons_path


def read_neon_token(token_path: str = "neon_token.txt") -> str:
    """Read NEON API token from a text file."""
    with open(token_path, "r") as file:
        token = file.read().strip()
    return token


def parse_tile_easting_northing(tile_name: str) -> Optional[Tuple[int, int]]:
    """Parse NEON tile easting/northing from a tile name.

    Supports patterns like '123000_456000' or 'E123000_N456000'. Returns integers.
    """
    # Try E######_N###### pattern
    m = re.search(r"E(\d{3,7}).*?N(\d{3,7})", tile_name)
    if m:
        return int(m.group(1)), int(m.group(2))
    # Try ######_###### pattern
    m = re.search(r"(\d{3,7})[_-](\d{3,7})", tile_name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def filter_unique_tiles(df: pd.DataFrame,
                        max_tiles_per_site: Optional[int], max_tiles: Optional[int]) -> pd.DataFrame:
    """Optionally limit number of unique tiles per site."""
   
    site_col = 'siteID'
    tile_col = 'tile_name'
    if max_tiles_per_site is None:
        return df
    # Keep at most N unique tiles per site
    tile_first = (df.drop_duplicates(
        [site_col, tile_col]).groupby(site_col).head(max_tiles_per_site))

    if max_tiles is not None:
        tile_first = tile_first.head(n=min(max_tiles, len(tile_first)))

    keep_tiles = set(zip(tile_first[site_col], tile_first[tile_col]))
    mask = list(zip(df[site_col], df[tile_col]))
    return df[[pair in keep_tiles for pair in mask]]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_tile_rgb(site: str,
                      easting: int,
                      northing: int,
                      year: int,
                      savepath: str,
                      token: str,
                      data_product: str = "DP3.30010.001") -> None:
    try:
        import neonutilities as nu  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Optional dependency missing: 'neonutilities'. Install with `pip install milliontrees[unsupervised]` or `pip install neonutilities`."
        ) from exc
    nu.by_tile_aop(
        dpid=data_product,
        site=site,
        easting=int(easting),
        northing=int(northing),
        year=year,
        token=token,
        include_provisional=True,
        check_size=False,
        savepath=savepath,
        verbose=True,
    )


def copy_downloads_to_images(download_root: str, images_dir: str) -> None:
    """Copy downloaded image tiles (e.g., .tif) into the dataset images directory."""
    ensure_dir(images_dir)
    tif_files = glob.glob(os.path.join(download_root, "**", "*.tif"), recursive=True)
    for src in tif_files:
        dst = os.path.join(images_dir, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

def download_and_tile_neon_data(annotations_path: str, data_dir: str, max_tiles_per_site: Optional[int],
                               patch_size: int, allow_empty: bool, token_path: str, 
                               data_product: str, download_dir: str, max_tiles: Optional[int] = None):
    """
    Download NEON tiles based on annotations and create tiled annotations.
    
    Args:
        annotations_path: Path to the formatted annotations CSV
        data_dir: Path to dataset directory
        max_tiles_per_site: Optional limit on tiles per site
        patch_size: Patch size for tiling in pixels
        allow_empty: Include empty crops during tiling
        token_path: Path to NEON API token file
        data_product: NEON data product ID
        download_dir: Temporary directory for downloads
        max_tiles: Optional absolute limit on total tiles
    """
    print(f"\n=== DOWNLOADING AND TILING NEON DATA ===")
    print(f"Processing annotations: {annotations_path}")
    
    images_dir = os.path.join(data_dir, 'images')
    
    # Load annotations    
    ann = read_file(annotations_path)
    ann = gpd.GeoDataFrame(ann, geometry="geometry")

    # Filter tiles per site if requested
    ann_filtered = filter_unique_tiles(ann, max_tiles_per_site, max_tiles)

    print(f"Filtered to {len(ann_filtered)} annotations from {ann_filtered['tile_name'].nunique()} unique tiles")

    # Download tiles
    token = read_neon_token(token_path)
    # Expect to be able to parse easting/northing from tile_name
    to_download = ann_filtered[["siteID", "tile_name"]].drop_duplicates()
    
    print(f"Downloading {len(to_download)} unique tiles...")
    
    for i, (_, row) in enumerate(to_download.iterrows()):
        site = row["siteID"]
        tile_name = str(row["tile_name"])
        parsed = parse_tile_easting_northing(tile_name)
        if parsed is None:
            print(f"Could not parse easting/northing from tile '{tile_name}', skipping download.")
            continue
        easting, northing = parsed

        # Year is the first 4 digits of the tile name
        year = int(tile_name[:4])

        print(f"[{i+1}/{len(to_download)}] Downloading site={site}, tile={tile_name}, easting={easting}, northing={northing}, year={year}")
        download_tile_rgb(site=site,
                            easting=easting,
                            northing=northing,
                            year=year,
                            savepath=download_dir,
                            token=token,
                            data_product=data_product)
        
        # Copy downloaded tiles into dataset images directory
        copy_downloads_to_images(download_dir, images_dir)

    # --- Tiling annotations per tile and saving as parquet collection ---
    tiled_output_dir = os.path.join(data_dir, 'unsupervised', 'unsupervised_annotations_tiled')
    os.makedirs(tiled_output_dir, exist_ok=True)

    def split_tile(ann_tile: pd.DataFrame, tile_image_path: str) -> Optional[pd.DataFrame]:
        """Split a single tile into patches and return tiled annotations."""
        # Run DeepForest tiling
        # Rename the tile_name to the image_path
        ann_tile = ann_tile.copy()
        ann_tile['image_path'] = os.path.basename(tile_image_path)
        tiled_df =    split_raster(
            annotations_file=ann_tile,
            path_to_raster=tile_image_path,
            save_dir=tiled_output_dir,
            patch_size=patch_size,
            allow_empty=allow_empty,
            root_dir=images_dir,
        )

        # Normalize schema for MillionTrees TreeBoxes
        tiled_df['filename'] = tiled_df['image_path'].apply(lambda x: os.path.join(images_dir, x))
        tiled_df['source'] = 'Weinstein et al. 2018 unsupervised'
        tiled_df['split'] = 'train'

        keep_cols = [
            c for c in
            ['xmin', 'ymin', 'xmax', 'ymax', 'filename', 'source', 'split']
            if c in tiled_df.columns
        ]
        tiled_df = tiled_df[keep_cols]

        return tiled_df

    # Process tiles and create tiled annotations
    print("Creating tiled annotations...")
    crop_annotations = []
    for tile_name in to_download.tile_name.unique():
        full_path = os.path.join(images_dir, tile_name)
        ann_tile = ann[ann.tile_name == tile_name]
        tiled_df = split_tile(ann_tile, full_path)
        if tiled_df is not None and len(tiled_df) > 0:
            crop_annotations.append(tiled_df)

    if crop_annotations:
        crop_annotations = pd.concat(crop_annotations, ignore_index=True)
        output_path = os.path.join(data_dir, 'unsupervised', 'unsupervised_neon_tiled.csv')
        crop_annotations.to_csv(output_path, index=False)
        print(f"Saved {len(crop_annotations)} tiled annotations to {output_path}")
    else:
        print("Warning: No tiled annotations were created")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process NEON unsupervised annotations: format, download tiles, and create tiled annotations."
    )
    
    # Formatting arguments
    parser.add_argument(
        '--csv_glob_pattern',
        default='/blue/ewhite/veitchmichaelisj/deeplidar/output/*.csv',
        help='Glob pattern to find CSV files with NEON detections'
    )
    parser.add_argument(
        '--format_output_dir',
        default='/orange/ewhite/DeepForest/unsupervised',
        help='Directory to save formatted annotations'
    )
    
    # Download and tiling arguments  
    parser.add_argument(
        '--data_dir',
        required=True,
        help='Path to dataset directory, e.g., /path/to/TreeBoxes_v0.2'
    )
    parser.add_argument(
        '--max_tiles_per_site',
        type=int,
        default=2,
        help='Optional limit on number of unique tiles per site (default: 2)'
    )
    parser.add_argument(
        '--max_tiles',
        type=int,
        default=10,
        help='Optional absolute limit on total tiles to process (default: 10)'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=400,
        help='Patch size for tiling (pixels). Default 400'
    )
    parser.add_argument(
        '--allow_empty',
        action='store_true',
        help='Include empty crops during tiling'
    )
    parser.add_argument(
        '--token_path',
        default='neon_token.txt',
        help='Path to NEON API token file'
    )
    parser.add_argument(
        '--data_product',
        default='DP3.30010.001',
        help='NEON data product (default RGB)'
    )
    parser.add_argument(
        '--download_dir',
        default='neon_downloads',
        help='Temporary directory to store NEON downloads before copying'
    )
    parser.add_argument(
        '--skip_format',
        action='store_true',
        help='Skip formatting step and use existing annotations'
    )
    parser.add_argument(
        '--skip_download',
        action='store_true', 
        help='Skip download step and only do tiling'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Step 1: Format annotations (unless skipped)
    if not args.skip_format:
        boxes_path, points_path, polygons_path = format_neon_annotations(
            args.csv_glob_pattern, 
            args.format_output_dir
        )
    else:
        # Use existing formatted annotations
        boxes_path = os.path.join(args.format_output_dir, "TreeBoxes_neon_unsupervised.csv")
        points_path = os.path.join(args.format_output_dir, "TreePoints_neon_unsupervised.csv")
        polygons_path = os.path.join(args.format_output_dir, "TreePolygons_neon_unsupervised.csv")
        print(f"Using existing formatted annotations:")
        print(f"  TreeBoxes: {boxes_path}")
        print(f"  TreePoints: {points_path}")
        print(f"  TreePolygons: {polygons_path}")
    
    # Step 2: Download tiles and create tiled annotations (unless skipped)
    if not args.skip_download:
        # Process TreeBoxes (main dataset for download/tiling)
        download_and_tile_neon_data(
            annotations_path=boxes_path,
            data_dir=args.data_dir,
            max_tiles_per_site=args.max_tiles_per_site,
            patch_size=args.patch_size,
            allow_empty=args.allow_empty,
            token_path=args.token_path,
            data_product=args.data_product,
            download_dir=args.download_dir,
            max_tiles=args.max_tiles
        )
    else:
        print("Skipping download and tiling step")
    
    print("\n=== PROCESSING COMPLETE ===")


if __name__ == '__main__':
    main()