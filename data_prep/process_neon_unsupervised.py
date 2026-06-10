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
from deepforest.visualize import plot_results

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

    # The output dir contains both per-site CSVs (e.g. ABBY_2019.csv, with
    # confidence_mean/confidence_std) and a roll-up `all_annotations.csv` that
    # holds the SAME boxes without the confidence columns. Globbing both ingests
    # every box twice (once valued, once NaN-confidence), which silently doubled
    # every annotation in past releases. Read only the per-site files.
    csvs = [c for c in csvs if os.path.basename(c) != "all_annotations.csv"]
    if not csvs:
        raise ValueError(
            "Only all_annotations.csv was found; expected per-site CSVs with "
            "confidence columns to avoid duplicate boxes."
        )

    print(f"Found {len(csvs)} per-site CSV files to process")

    annotations = []
    for csv in csvs:
        df = pd.read_csv(csv)
        df["source"] = "Weinstein et al. 2018 unsupervised"
        annotations.append(df)
    annotations = pd.concat(annotations, ignore_index=True)

    # Safety net in case any per-site file overlaps another: collapse exact box
    # duplicates, preferring the copy that carries a confidence score.
    box_id = ["image_path", "xmin", "ymin", "xmax", "ymax"]
    if "confidence_mean" in annotations.columns:
        annotations = annotations.sort_values("confidence_mean", na_position="last")
    n_before = len(annotations)
    annotations = annotations.drop_duplicates(subset=box_id, keep="first").reset_index(drop=True)
    if len(annotations) < n_before:
        print(f"Dropped {n_before - len(annotations)} duplicate boxes "
              f"({n_before} -> {len(annotations)})")

    # Extract siteID and tile_name from the filename
    annotations["siteID"] = annotations["image_path"].apply(lambda x: x.split("_")[1])
    annotations["tile_name"] = annotations["image_path"].apply(lambda x: x.split("/")[2]+".tif")
    
    print(f"Number of sites: {annotations['siteID'].nunique()}")
    print(f"Number of tiles: {annotations['tile_name'].nunique()}")

    # Sample up to 30 tiles per site, keeping all trees from sampled tiles
    unique_tiles = annotations[["siteID", "tile_name"]].drop_duplicates()
    sampled_tiles = pd.concat([
        grp.sample(min(len(grp), 30), random_state=42)
        for _, grp in unique_tiles.groupby("siteID")
    ])
    annotations = annotations.merge(sampled_tiles[["siteID", "tile_name"]], on=["siteID", "tile_name"])
    print(f"After sampling (30 tiles per site): {len(annotations)} annotations")

    # Create metadata column
    annotations["metadata"] = annotations.apply(lambda row: f"{row['siteID']}_{row['tile_name']}", axis=1)

    # How many annotations, sites, and tiles?
    print(f"Number of annotations: {len(annotations)}")
    print(f"Number of sites: {annotations['siteID'].nunique()}")
    print(f"Number of tiles: {annotations['tile_name'].nunique()}")

    boxes_annotations = annotations.copy()

    os.makedirs(output_dir, exist_ok=True)
    boxes_path = os.path.join(output_dir, "TreeBoxes_neon_unsupervised.csv")
    
    boxes_annotations = read_file(boxes_annotations, root_dir=output_dir)
    boxes_annotations["source"] = "Weinstein et al. 2018 unsupervised"

    input_dir = os.path.dirname(csv_glob_pattern)
    copy_neon_images(boxes_annotations, input_dir, output_dir)

    boxes_annotations["image_path"] = boxes_annotations["image_path"].apply(
        lambda x: os.path.join(output_dir, os.path.basename(x)))

    boxes_annotations.to_csv(boxes_path, index=False)
    
    print(f"Saved formatted annotations:")
    print(f"  TreeBoxes: {boxes_path}")
    
    return boxes_path

def copy_neon_images(annotations: pd.DataFrame, input_dir: str, output_dir: str):
    """
    Copy the neon images to the output directory
    """
    images_dir = "/blue/ewhite/veitchmichaelisj/deeplidar/output/"
    for image_path in annotations.image_path.unique():
        input_path = os.path.join(input_dir, image_path)
        if not os.path.exists(input_path):
            continue
        else:
            shutil.copy(input_path, output_dir)

if __name__ == "__main__":
    csv_glob_pattern = "/blue/ewhite/veitchmichaelisj/deeplidar/output/*.csv"
    output_dir = "/orange/ewhite/DeepForest/neon_unsupervised/"
    os.makedirs(output_dir, exist_ok=True)
    annotations = format_neon_annotations(csv_glob_pattern, output_dir)
