#!/usr/bin/env python3
"""
Combined script to format NEON weak supervised annotations and download tiles.
This script:
1. Reads weak supervised NEON detection results from CSV files
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
    Format NEON weak supervised annotations into TreeBoxes, TreePoints, and TreePolygons formats.
    
    Args:
        csv_glob_pattern: Glob pattern to find CSV files with NEON detections
        output_dir: Directory to save formatted annotations
        
    Returns:
        Tuple of paths to the three output CSV files (boxes, points, polygons)
    """
    print("=== FORMATTING NEON WEAK SUPERVISED ANNOTATIONS ===")
    
    # Read the weak supervised NEON annotations
    csvs = glob.glob(csv_glob_pattern)
    if not csvs:
        raise ValueError(f"No CSV files found matching pattern: {csv_glob_pattern}")
        
    print(f"Found {len(csvs)} CSV files to process")
    
    annotations = []
    for csv in csvs[:2]:
        df = pd.read_csv(csv)
        df["source"] = "Weinstein et al. 2018 weak supervised"
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

    boxes_annotations = annotations.copy()

    os.makedirs(output_dir, exist_ok=True)

    boxes_path = os.path.join(output_dir, "TreeBoxes_neon_weak_supervised.csv")
    
    boxes_annotations = read_file(boxes_annotations)
    boxes_annotations["source"] = "Weinstein et al. 2018 weak supervised"

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
    output_dir = "/orange/ewhite/DeepForest/neon_weak_supervised/"
    os.makedirs(output_dir, exist_ok=True)
    annotations = format_neon_annotations(csv_glob_pattern, output_dir)
