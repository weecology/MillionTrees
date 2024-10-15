from deepforest import main
from deepforest.utilities import read_file
from deepforest.preprocess import split_raster
from deepforest.visualize import plot_results
import os
import geopandas as gpd
import pandas as pd
import rasterio as rio
from matplotlib import pyplot as plt
from shapely.geometry import box

gdf = gpd.read_file("/orange/ewhite/DeepForest/Araujo_2020/crown_delineation_shapefile.shp")
gdf =  gdf[gdf.geometry.type=="Polygon"]
gdf["image_path"] = "Orthomosaic_WGS84_UTM20S.tif"
gdf["label"] = "Tree"
gdf["source"] = "Araujo et al. 2020"
df = read_file(gdf, root_dir="/orange/ewhite/DeepForest/Araujo_2020/")
df.root_dir = "/orange/ewhite/DeepForest/Araujo_2020/"
df = df[["geometry", "image_path", "label", "source"]]
split_files = split_raster(df, path_to_raster="/orange/ewhite/DeepForest/Araujo_2020/Orthomosaic_WGS84_UTM20S.tif", root_dir="/orange/ewhite/DeepForest/Araujo_2020/",
                           base_dir="/orange/ewhite/DeepForest/Araujo_2020/crops/", patch_size=2000, patch_overlap=0)

for image in split_files.image_path.unique():
    image_df = split_files[split_files.image_path==image]
    image_df.root_dir = "/orange/ewhite/DeepForest/Araujo_2020/crops/"
    image_df["score"] = 1
    channels, height, width = rio.open(os.path.join("/orange/ewhite/DeepForest/Araujo_2020/crops/", image)).read().shape
    plot_results(image_df, height=height, width=width)

split_files["image_path"] = split_files["image_path"].apply(lambda x: os.path.join("/orange/ewhite/DeepForest/Araujo_2020/crops/", x))
split_files.to_csv("/orange/ewhite/DeepForest/Araujo_2020/annotations.csv")