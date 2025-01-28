# Harz Mountain polygon dataset
import geopandas as gpd
from deepforest.utilities import read_file
import glob
import os
import pandas as pd

# Load in all train and test shapeiles
shapefiles = glob.glob("/orange/ewhite/DeepForest/Harz_Mountains/ML_TreeDetection_Harz/test/annotations/*.shp") + glob.glob("/orange/ewhite/DeepForest/Harz_Mountains/ML_TreeDetection_Harz/train/annotations/*.shp")

annotations = []
for shapefile in shapefiles:
    print("Processing: {}".format(shapefile))
    try:
        gdf = gpd.read_file(shapefile)
    except:
        print("Could not read: {}".format(shapefile))
        continue
    gdf["image_path"] = os.path.basename(shapefile).replace(".shp", ".tif")
    # Append the word "aerial" to the image path at the beginning
    gdf["image_path"] = gdf["image_path"].apply(lambda x: "aerial_"+x)
    # remove MultiPolygons
    gdf = gdf[gdf.geometry.type == "Polygon"]

    gdf["label"] = "tree"
    annotation = read_file(gdf, root_dir="/orange/ewhite/DeepForest/Harz_Mountains/ML_TreeDetection_Harz/all_images")
    annotations.append(annotation)

# full image path
annotations = pd.concat(annotations)
annotations["image_path"] = annotations["image_path"].apply(lambda x: os.path.join("/orange/ewhite/DeepForest/Harz_Mountains/ML_TreeDetection_Harz/all_images",x))

annotations.to_csv("/orange/ewhite/DeepForest/Harz_Mountains/ML_TreeDetection_Harz/annotations.csv", index=False)
