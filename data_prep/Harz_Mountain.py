# Harz Mountain polygon dataset
import geopandas as gpd
from deepforest.utilities import read_file
import glob
import os
import pandas as pd
from shapely.geometry import Polygon
from PIL import Image
import rasterio
from deepforest.visualize import plot_results
from matplotlib import pyplot as plt

# Load in all train and test shapefiles
shapefiles = glob.glob("/orange/ewhite/DeepForest/Harz_Mountains/ML_TreeDetection_Harz/test/annotations/*.shp") + glob.glob("/orange/ewhite/DeepForest/Harz_Mountains/ML_TreeDetection_Harz/train/annotations/*.shp")

annotations = []
for shapefile in shapefiles:
    print("Processing: {}".format(shapefile))
    try:
        gdf = gpd.read_file(shapefile)
        # Remove MultiPolygons
        gdf = gdf[gdf.geometry.type == "Polygon"]   
        
        # Convert Z polygon to 2D
        gdf = gdf.set_geometry(gdf.geometry.apply(lambda geom: Polygon([(x, y) for x, y, z in geom.exterior.coords]) if geom.has_z else geom))
    except Exception as e:
            print(f"Could not read {shapefile}: {e}")
            continue

    # Update image paths
    gdf["image_path"] = os.path.basename(shapefile).replace(".shp", ".tif")
    gdf["image_path"] = gdf["image_path"].apply(lambda x: "aerial_" + x)

    # Convert .tif to .png
    gdf["label"] = "tree"
    annotation = read_file(gdf, root_dir="/orange/ewhite/DeepForest/Harz_Mountains/ML_TreeDetection_Harz/all_images")
    unique_images = gdf["image_path"].unique()
    for row in unique_images:
        tif_path = os.path.join("/orange/ewhite/DeepForest/Harz_Mountains/ML_TreeDetection_Harz/all_images", row)
        png_path = tif_path.replace(".tif", ".png")
        try:
            with rasterio.open(tif_path) as src:
                img = src.read()
                # remove NIR band
                # Flip channels from BGR to RGB
                img = img[:3, :, :]  # Keep only the first three channels (RGB)
                img = img.transpose(1, 2, 0)  # Change to HWC format
                img = img.astype("uint8")
                img = Image.fromarray(img)
                img.save(png_path, "PNG")
        except Exception as e:
            print(f"Could not convert {tif_path} to PNG: {e}")
            continue
    
    annotation["image_path"] = gdf["image_path"].apply(lambda x: x.replace(".tif", ".png"))
    annotations.append(annotation)

# Combine all annotations
annotations = pd.concat(annotations)

# Update full image paths
annotations["image_path"] = annotations["image_path"].apply(
    lambda x: os.path.join("/orange/ewhite/DeepForest/Harz_Mountains/ML_TreeDetection_Harz/all_images", x)
)
annotations["source"] = "Lucas et al. 2024"

# Save combined annotations
annotations.to_csv("/orange/ewhite/DeepForest/Harz_Mountains/ML_TreeDetection_Harz/annotations.csv", index=False)
print("Annotations saved to /orange/ewhite/DeepForest/Harz_Mountains/ML_TreeDetection_Harz/annotations.csv")

sample_image = annotations.head()
sample_image["label"] = "Tree"
sample_image.root_dir = "/orange/ewhite/DeepForest/Harz_Mountains/ML_TreeDetection_Harz/all_images"
# Read and get the height and width of the image
image_path = sample_image["image_path"].iloc[0]
image = Image.open(image_path)
width, height = image.size
print(f"Image size: {width} x {height}")
ax = plot_results(sample_image, savedir="/orange/ewhite/DeepForest/Harz_Mountains/ML_TreeDetection_Harz/", basename="sample_image",width=width, height=height)