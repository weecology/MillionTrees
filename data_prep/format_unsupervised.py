# Create unsupervised annotations from NEON dataset
import pandas as pd
import os

from deepforest import utilities
from shapely.geometry import Point, Polygon

# Read the unsupervised annotations
unsupervised_annotations_path = "/orange/ewhite/b.weinstein/NeonTreeEvaluation/pretraining.csv"
annotations = utilities.read_file(unsupervised_annotations_path)

# How many annotations, sites, and tiles?
print(f"Number of annotations: {len(annotations)}")

# Extract siteID and tile_name from the filename
annotations["siteID"] = annotations["image_path"].apply(lambda x: x.split("/")[-2])
annotations["tile_name"] = annotations["image_path"].apply(lambda x: x.split("/")[-1].split(".")[0])

print(f"Number of sites: {annotations['siteID'].nunique()}")
print(f"Number of tiles: {annotations['tile_name'].nunique()}")

# Create metadata column
annotations["metadata"] = annotations.apply(lambda row: f"{row['siteID']}_{row['tile_name']}", axis=1)

# Create points version
points_annotations = annotations.copy()
points_annotations["geometry"] = points_annotations.apply(lambda row: Point((row["xmin"] + row["xmax"]) / 2, (row["ymin"] + row["ymax"]) / 2), axis=1)
points_annotations["source"] = "Weinstein et al. 2018 unsupervised"

# Create polygons version
polygons_annotations = annotations.copy()
polygons_annotations["geometry"] = polygons_annotations.apply(lambda row: Polygon([(row["xmin"], row["ymin"]), (row["xmax"], row["ymin"]), (row["xmax"], row["ymax"]), (row["xmin"], row["ymax"])]), axis=1)
polygons_annotations["source"] = "Weinstein et al. 2018 unsupervised"

# Create box version (same as original but with metadata and source)
boxes_annotations = annotations.copy()
boxes_annotations["source"] = "Weinstein et al. 2018 unsupervised"
