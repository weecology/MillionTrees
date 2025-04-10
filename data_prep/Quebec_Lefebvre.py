import geopandas as gpd
from deepforest.utilities import read_file
from deepforest.preprocess import split_raster
import glob
import os
import pandas as pd
from shapely.geometry import Polygon
from PIL import Image
import rasterio

# Define directories
vector_data_dir = "/orange/ewhite/DeepForest/Quebec_Lefebvre/Dataset/Vector_Data"
raster_data_dir = "/orange/ewhite/DeepForest/Quebec_Lefebvre/Dataset/Photogrammetry_Products"
output_dir = "/orange/ewhite/DeepForest/Quebec_Lefebvre/Dataset/Crops"
os.makedirs(output_dir, exist_ok=True)

# Load all GeoPackages
geopackages = glob.glob(os.path.join(vector_data_dir, "*.gpkg"))
annotations = []

for geopackage in geopackages:
    basename = os.path.splitext(os.path.basename(geopackage))[0]
    corresponding_tif = os.path.join(raster_data_dir,basename,f"{basename}_rgb.cog.tif")
    
    if not os.path.exists(corresponding_tif):
        print(f"Skipping {geopackage}, no matching .tif file found.")
        continue

    # Load GeoPackage
    gdf = gpd.read_file(geopackage)

    # Update image paths
    gdf["image_path"] = os.path.basename(corresponding_tif)
    gdf["label"] = "tree"
    
    annotation = read_file(gdf, root_dir=os.path.dirname(corresponding_tif))

    # Split rasters into smaller patches
    crop_annotations = split_raster(
        annotations,
        path_to_raster=corresponding_tif,
        patch_size=1000,
        allow_empty=False,
        base_dir=output_dir
    )

    annotations.append(crop_annotations)

# Combine all annotations
annotations = pd.concat(annotations)

# Update full image paths
crop_annotations["image_path"] = crop_annotations["image_path"].apply(lambda x: os.path.join(output_dir, x))
crop_annotations["source"] = "Lefebvre et al. 2024"

# Save combined annotations
output_csv = os.path.join(output_dir, "annotations.csv")
crop_annotations.to_csv(output_csv, index=False)
print(f"Annotations saved to {output_csv}")