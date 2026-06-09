"""Prepare the SelvaMask tropical tree-crown segmentation dataset for MillionTrees.

SelvaMask (https://huggingface.co/datasets/selvamask/SelvaMask) ships COCO-style
polygon annotations stored in HuggingFace parquet shards across ``train``,
``validation`` and ``test`` splits. Each row is an RGB tile with crown polygons
in pixel coordinates relative to the tile origin (0, 0 = top-left), which is
already the MillionTrees image coordinate convention.

We honor the published split: the HF ``test`` split becomes ``existing_split ==
"test"`` and ``train`` + ``validation`` are folded into ``train`` (MillionTrees
only carries a train/test split, and validation is an intermediate set).
"""
import io
import os

import geopandas as gpd
import pandas as pd
import requests
from PIL import Image
from shapely.geometry import Polygon
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

SOURCE = "SelvaMask"
DATASET_NAME = "selvamask/SelvaMask"
OUTPUT_DIR = "/orange/ewhite/DeepForest/SelvaMask"


def polygon_from_segmentation(segmentation):
    """Build a shapely Polygon from a COCO segmentation entry.

    ``segmentation`` is a list of rings; each ring is a flat ``[x1, y1, x2, y2,
    ...]`` list. We use the first (exterior) ring and drop annotations with
    fewer than three vertices.
    """
    if not segmentation:
        return None
    ring = segmentation[0]
    coords = list(zip(ring[0::2], ring[1::2]))
    if len(coords) < 3:
        return None
    return Polygon(coords)


def download_selvamask(force_download=False):
    """Download SelvaMask parquet shards, write PNG tiles and a polygon CSV."""
    images_dir = os.path.join(OUTPUT_DIR, "images")
    cache_dir = os.path.join(OUTPUT_DIR, "cache")
    annotations_csv = os.path.join(OUTPUT_DIR, "annotations.csv")

    if not force_download and os.path.exists(annotations_csv):
        print(f"Dataset already exists at {annotations_csv}; pass force_download=True to rebuild.")
        return annotations_csv

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    api_url = f"https://datasets-server.huggingface.co/parquet?dataset={DATASET_NAME}"
    parquet_files = requests.get(api_url).json()["parquet_files"]

    records = []
    for file_info in parquet_files:
        split = file_info["split"]
        existing_split = "test" if split == "test" else "train"
        cached_parquet = os.path.join(cache_dir, f"{split}_{file_info['filename']}")

        if force_download or not os.path.exists(cached_parquet):
            print(f"Downloading {split}/{file_info['filename']} ...")
            response = requests.get(file_info["url"], stream=True)
            response.raise_for_status()
            with open(cached_parquet, "wb") as f:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    f.write(chunk)

        df = pd.read_parquet(cached_parquet)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
            image_filename = os.path.splitext(row["tile_name"])[0] + ".png"
            image_path = os.path.join(images_dir, image_filename)
            if force_download or not os.path.exists(image_path):
                with Image.open(io.BytesIO(row["image"]["bytes"])) as img:
                    img.convert("RGB").save(image_path, "PNG")

            for segmentation in row["annotations"]["segmentation"]:
                polygon = polygon_from_segmentation(segmentation)
                if polygon is None:
                    continue
                records.append({
                    "image_path": image_path,
                    "geometry": polygon.wkt,
                    "label": "tree",
                    "source": SOURCE,
                    "existing_split": existing_split,
                })

    annotations = gpd.GeoDataFrame(records)
    annotations.to_csv(annotations_csv, index=False)
    print(f"Saved {len(annotations)} polygons across "
          f"{annotations['image_path'].nunique()} tiles to {annotations_csv}")
    print(annotations["existing_split"].value_counts())
    return annotations_csv


if __name__ == "__main__":
    download_selvamask()
