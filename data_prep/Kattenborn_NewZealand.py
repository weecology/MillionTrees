import geopandas as gpd
from deepforest.utilities import read_file
from deepforest.preprocess import split_raster
import glob
import os
import pandas as pd
from shapely.geometry import Polygon
from PIL import Image
import rasterio

# Define the directory containing the shapefiles
shapefiles_dir = "/orange/ewhite/DeepForest/Kattenborn/uav_newzealand_waititu"

# Define the .tif files for ecm and wcm
ecm_tif = "ecm_uav_rgb_dsm.tif"
wcm_tif = "wcm_uav_rgb_dsm.tif"

# Process ecm shapefiles
ecm_shapefiles = glob.glob(os.path.join(shapefiles_dir, "ecm*.shp"))
ecm_annotations = []
for shapefile in ecm_shapefiles:
    print(f"Processing ECM shapefile: {shapefile}")
    try:
        gdf = gpd.read_file(shapefile)
        # Remove MultiPolygons
        gdf = gdf[gdf.geometry.type == "Polygon"]

    except Exception as e:
        print(f"Could not read {shapefile}: {e}")
        continue

    # Update image paths
    gdf["image_path"] = ecm_tif

    # Convert .tif to .png
    gdf["label"] = "tree"
    annotation = read_file(gdf, root_dir=shapefiles_dir)
    png_path = os.path.join(shapefiles_dir, ecm_tif.replace(".tif", ".png"))
    try:
        with rasterio.open(os.path.join(shapefiles_dir, ecm_tif)) as src:
            img = src.read()
            # Remove NIR band
            img = img[:3, :, :]  # Keep only the first three channels (RGB)
            img = img.transpose(1, 2, 0)  # Change to HWC format
            img = img.astype("uint8")
            img = Image.fromarray(img)
            img.save(png_path, "PNG")
    except Exception as e:
        print(f"Could not convert {ecm_tif} to PNG: {e}")
        continue

    annotation["image_path"] = png_path
    ecm_annotations.append(annotation)

# Process wcm shapefiles
wcm_shapefiles = glob.glob(os.path.join(shapefiles_dir, "wcm*.shp"))
wcm_annotations = []
for shapefile in wcm_shapefiles:
    print(f"Processing WCM shapefile: {shapefile}")
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
    gdf["image_path"] = wcm_tif

    # Convert .tif to .png
    gdf["label"] = "tree"
    annotation = read_file(gdf, root_dir=shapefiles_dir)
    png_path = os.path.join(shapefiles_dir, wcm_tif.replace(".tif", ".png"))
    try:
        with rasterio.open(os.path.join(shapefiles_dir, wcm_tif)) as src:
            img = src.read()
            # Remove NIR band
            img = img[:3, :, :]  # Keep only the first three channels (RGB)
            img = img.transpose(1, 2, 0)  # Change to HWC format
            img = img.astype("uint8")
            img = Image.fromarray(img)
            img.save(png_path, "PNG")
    except Exception as e:
        print(f"Could not convert {wcm_tif} to PNG: {e}")
        continue

    annotation["image_path"] = png_path
    wcm_annotations.append(annotation)

# split_raster 
os.makedirs("/orange/ewhite/DeepForest/Kattenborn/uav_newzealand_waititu/crops/", exist_ok=True)

ecm_annotations = pd.concat(ecm_annotations).reset_index(drop=True)
ecm_annotations["image_path"] = ecm_annotations["image_path"].apply(lambda x: os.path.basename(x))
crop_ecm_annotations = split_raster(
    ecm_annotations,
    path_to_raster='/orange/ewhite/DeepForest/Kattenborn/uav_newzealand_waititu/' + ecm_tif.replace(".tif", ".png"),
    patch_size=1500,
    allow_empty=False,  
    base_dir="/orange/ewhite/DeepForest/Kattenborn/uav_newzealand_waititu/crops")

wcm_annotations = pd.concat(wcm_annotations).reset_index(drop=True)
wcm_annotations["image_path"] = wcm_annotations["image_path"].apply(lambda x: os.path.basename(x))
crop_wcm_annotations = split_raster(
    wcm_annotations,
    path_to_raster='/orange/ewhite/DeepForest/Kattenborn/uav_newzealand_waititu/' + wcm_tif.replace(".tif", ".png"),
    patch_size=1500,
    allow_empty=False,  
    base_dir="/orange/ewhite/DeepForest/Kattenborn/uav_newzealand_waititu/crops")

# Combine ecm and wcm annotations
annotations = pd.concat([crop_wcm_annotations, crop_ecm_annotations])

# Update full image paths
annotations["image_path"] = annotations["image_path"].apply(lambda x: os.path.join("/orange/ewhite/DeepForest/Kattenborn/uav_newzealand_waititu/crops", x))
annotations["source"] = "Kattenborn et al. 2023"

# Save combined annotations
output_csv = os.path.join("/orange/ewhite/DeepForest/Kattenborn/uav_newzealand_waititu/crops", "annotations.csv")
annotations.to_csv(output_csv, index=False)
print(f"Annotations saved to {output_csv}")