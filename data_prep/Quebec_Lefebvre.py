import geopandas as gpd
from deepforest.utilities import read_file
from deepforest.preprocess import split_raster
from deepforest.visualize import plot_annotations
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

# Define directories
vector_data_dir = "/orange/ewhite/DeepForest/Quebec_Lefebvre/Dataset/Vector_Data"
raster_data_dir = "/orange/ewhite/DeepForest/Quebec_Lefebvre/Dataset/Photogrammetry_Products"
output_dir = "/orange/ewhite/DeepForest/Quebec_Lefebvre/Dataset/Crops"
os.makedirs(output_dir, exist_ok=True)

# Load all GeoPackages
geopackages = glob.glob(os.path.join(vector_data_dir, "*.gpkg"))

# remove one 
annotations = []

for geopackage in geopackages:
    print(geopackage)
    basename = os.path.splitext(os.path.basename(geopackage))[0]
    corresponding_tif = os.path.join(raster_data_dir,basename,f"{basename}_rgb.cog.tif")
    
    if not os.path.exists(corresponding_tif):
        print(f"Skipping {geopackage}, no matching .tif file found.")
        continue

    # Load GeoPackage
    gdf = gpd.read_file(geopackage, layer=1)

    # Update image paths
    gdf["image_path"] = os.path.basename(corresponding_tif)
    gdf["label"] = "tree"
    
    # Save the .shp
    shpfile = os.path.join(output_dir, os.path.basename(geopackage).replace(".gpkg", ".shp"))
    gdf.to_file(shpfile)
    # if os.path.exists(output_dir + "/"+ os.path.basename(corresponding_tif).replace(".tif", ".csv")):
    #     print(f"Skipping {corresponding_tif}, already processed.")
    #     crop_annotations = pd.read_csv(output_dir + "/"+ os.path.basename(corresponding_tif).replace(".tif", ".csv"))
    # else:
        # Split rasters into smaller patches
    annotation = read_file(shpfile, root_dir=os.path.dirname(corresponding_tif))
    crop_annotations = split_raster(
        annotation,
        path_to_raster=corresponding_tif,
        patch_size=3000,
        allow_empty=False,
        base_dir=output_dir
    )

    annotations.append(crop_annotations)

# Combine all annotations
annotations = pd.concat(annotations)

# Update full image paths
annotations["image_path"] = annotations["image_path"].apply(lambda x: os.path.join(output_dir, x))
annotations["source"] = "Lefebvre et al. 2024"

# Save combined annotations
output_csv = os.path.join(output_dir, "annotations.csv")
annotations.to_csv(output_csv, index=False)
print(f"Annotations saved to {output_csv}")

# Plot some sample images
for index, image_path in enumerate(annotations["image_path"].unique()[:10]):
    
    image_annotations = annotations[annotations["image_path"] == image_path].copy(deep=True)
    #image_annotations["image_path"] = image_annotations["image_path"].apply(lambda x: os.path.basename(x))
    image_annotations = read_file(image_annotations)
    image_annotations.root_dir = os.path.dirname(image_path)
    plot_annotations(annotations=image_annotations,radius=20)
    plt.savefig(f"current_{index}.png")