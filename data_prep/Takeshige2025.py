import os
import glob
import pandas as pd
import geopandas as gpd
from deepforest.preprocess import split_raster, read_file
import rasterio as rio
from deepforest.visualize import plot_results
import matplotlib.pyplot as plt
import numpy as np

def process_takeshige2025():
    # Directories
    base_dir = "/orange/ewhite/DeepForest/takeshige2025"
    ortho_dir = os.path.join(base_dir, "Ortho")
    crown_dir = os.path.join(base_dir, "Crown")
    output_dir = os.path.join(base_dir, "crops")
    os.makedirs(output_dir, exist_ok=True)

    # Find all orthophotos
    ortho_files = glob.glob(os.path.join(ortho_dir, "*.tif"))
    
    all_annotations = []

    for ortho_path in ortho_files:
        # Extract the base identifier (e.g., "Ashoro_AS-DB1" from "Ashoro_AS-DB1_Ortho.tif")
        basename = os.path.splitext(os.path.basename(ortho_path))[0]
        base_id = basename.replace("_Ortho", "")
        
        # Find matching crown shapefile
        crown_path = os.path.join(crown_dir, f"{base_id}_crown.shp")
        
        if not os.path.exists(crown_path):
            print(f"Skipping {ortho_path}, no matching crown shapefile found.")
            continue

        print(f"Processing {ortho_path} and {crown_path}")

        # Read crown polygons
        gdf = gpd.read_file(crown_path)
        
        # Ensure we only have polygons
        gdf = gdf[gdf.geometry.type == "Polygon"].copy()
        
        # Add required columns for deepforest format
        gdf["image_path"] = os.path.basename(ortho_path)
        gdf["label"] = "tree"

        # Read as MillionTrees annotation
        annotation = read_file(gdf, root_dir=ortho_dir)

        # Read image with rasterio and handle NaN values
        with rio.open(ortho_path) as src:
            # Read all bands
            image = src.read()
            # Transpose to channels last format (H, W, C)
            image = np.transpose(image, (1, 2, 0))
            # Replace NaN values with 0
            image = np.nan_to_num(image, nan=0)
            # Ensure values are in uint8 range
            image = np.clip(image, 0, 255).astype(np.uint8)
            
        # Split raster into patches using the cleaned image
        crop_annotations = split_raster(
            annotation,
            numpy_image=image,
            patch_size=1000,  # Using 1000px windows as requested
            allow_empty=False,
            image_name=os.path.basename(ortho_path),
            save_dir=output_dir
        )
        all_annotations.append(crop_annotations)

    # Combine and finalize
    annotations = pd.concat(all_annotations)
    annotations["image_path"] = annotations["image_path"].apply(lambda x: os.path.join(output_dir, x))
    annotations["source"] = "Takeshige et al. 2025"

    # Save
    output_csv = os.path.join(output_dir, "annotations.csv")
    annotations.to_csv(output_csv, index=False)
    print(f"Annotations saved to {output_csv}")

if __name__ == "__main__":
    process_takeshige2025()
    
    # Optional: Visualize results for the first image
    df = read_file("/orange/ewhite/DeepForest/takeshige2025/crops/annotations.csv")
    df = df[df["image_path"] == df["image_path"].iloc[0]]
    df.root_dir = "/orange/ewhite/DeepForest/takeshige2025/crops"
    plot_results(df)
    plt.savefig("/orange/ewhite/DeepForest/takeshige2025/takeshige2025.png")