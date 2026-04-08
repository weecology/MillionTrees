#!/usr/bin/env python3
"""Regenerate docs/public/ sample images from already-packaged mini datasets.

Run from the repo root:
    python data_prep/regenerate_doc_images.py

Reads the existing MiniTree*_<version>/random.csv files and re-renders one
image per source, fixing the bug where annotations from multiple images were
overlaid on a single background.
"""
import os
import cv2
import pandas as pd
from deepforest.visualize import plot_results
from deepforest.utilities import read_file

BASE_DIR = "/orange/ewhite/web/public/MillionTrees/"
VERSION = "v0.12"
OUTPUT_DIR = "docs/public/"

DATASET_TYPES = ["TreeBoxes", "TreePoints", "TreePolygons"]


def regenerate_visualizations(base_dir, dataset_type, version, output_dir):
    csv_path = f"{base_dir}Mini{dataset_type}_{version}/random.csv"
    images_dir = f"{base_dir}Mini{dataset_type}_{version}/images/"

    if not os.path.exists(csv_path):
        print(f"Skipping {dataset_type}: {csv_path} not found")
        return

    annotations = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    for source, group in annotations.groupby("source"):
        print(f"  {source}")
        # Use only the first image to avoid mixing coordinates from different images
        first_image = group["filename"].iloc[0]
        group = group[group["filename"] == first_image].copy()
        group["image_path"] = group["filename"]
        try:
            group = read_file(group, root_dir=images_dir)
            group.root_dir = images_dir

            image_path = os.path.join(images_dir, group.image_path.iloc[0])
            img = cv2.imread(image_path)
            if img is None:
                print(f"    WARNING: could not read {image_path}, skipping")
                continue
            height, width, _ = img.shape

            safe_name = source.replace(" ", "_")
            plot_results(group, savedir=output_dir, basename=safe_name, height=height, width=width)
        except Exception as e:
            print(f"    ERROR: {e}")


if __name__ == "__main__":
    for dataset_type in DATASET_TYPES:
        print(f"\n=== {dataset_type} ===")
        regenerate_visualizations(BASE_DIR, dataset_type, VERSION, OUTPUT_DIR)
    print("\nDone.")
