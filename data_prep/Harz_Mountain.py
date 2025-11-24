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

BASE_DIR = "/orange/ewhite/DeepForest/Harz_Mountains/ML_TreeDetection_Harz"
ALL_IMAGES_DIR = os.path.join(BASE_DIR, "all_images")


def process_split(shapefile_paths: list[str], split_name: str) -> list[pd.DataFrame]:
    results = []
    for shapefile in shapefile_paths:
        print("Processing: {}".format(shapefile))
        try:
            gdf = gpd.read_file(shapefile)
            # Remove MultiPolygons
            gdf = gdf[gdf.geometry.type == "Polygon"]
            # Convert Z polygon to 2D
            gdf = gdf.set_geometry(
                gdf.geometry.apply(
                    lambda geom: Polygon([(x, y) for x, y, z in geom.exterior.coords]) if geom.has_z else geom
                )
            )
        except Exception as e:
            print(f"Could not read {shapefile}: {e}")
            continue

        # Update image paths
        gdf["image_path"] = os.path.basename(shapefile).replace(".shp", ".tif")
        gdf["image_path"] = gdf["image_path"].apply(lambda x: "aerial_" + x)

        # Convert .tif to .png
        gdf["label"] = "tree"
        annotation = read_file(gdf, root_dir=ALL_IMAGES_DIR)
        unique_images = gdf["image_path"].unique()
        for row in unique_images:
            tif_path = os.path.join(ALL_IMAGES_DIR, row)
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
        annotation["existing_split"] = split_name
        results.append(annotation)
    return results


# Load train and test shapefiles separately
train_shapefiles = glob.glob(os.path.join(BASE_DIR, "train", "annotations", "*.shp"))
test_shapefiles = glob.glob(os.path.join(BASE_DIR, "test", "annotations", "*.shp"))

annotations_parts = []
annotations_parts += process_split(train_shapefiles, "train")
annotations_parts += process_split(test_shapefiles, "test")

# Combine all annotations
annotations = pd.concat(annotations_parts)

# Update full image paths
annotations["image_path"] = annotations["image_path"].apply(
    lambda x: os.path.join(ALL_IMAGES_DIR, x)
)
annotations["source"] = "Lucas et al. 2024"

# Save combined annotations
annotations.to_csv(os.path.join(BASE_DIR, "annotations.csv"), index=False)
print(f"Annotations saved to {os.path.join(BASE_DIR, 'annotations.csv')}")

sample_image = annotations.head()
sample_image["label"] = "Tree"
sample_image.root_dir = ALL_IMAGES_DIR
# Read and get the height and width of the image
image_path = sample_image["image_path"].iloc[0]
image = Image.open(image_path)
width, height = image.size
print(f"Image size: {width} x {height}")
ax = plot_results(sample_image, savedir=f"{BASE_DIR}/", basename="sample_image", width=width, height=height)