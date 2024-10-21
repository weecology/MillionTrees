import geopandas as gpd
from deepforest.utilities import read_file
from deepforest.preprocess import split_raster
from deepforest.visualize import plot_results
import rasterio as rio
from rasterio import warp
import os
from matplotlib import pyplot as plt

gdf = gpd.read_file("/orange/ewhite/DeepForest/Tonga/Kolovai-Trees-20180108_projected.shp")

gdf["label"] = "Tree"
gdf["source"] = "Kolovai-Trees"
gdf["image_path"] = "Kolovai-Trees-20180108.tif"

df = read_file(gdf, root_dir="/orange/ewhite/DeepForest/Tonga")
split_files = split_raster(df,
            path_to_raster="/orange/ewhite/DeepForest/Tonga/Kolovai-Trees-20180108.tif",
            patch_overlap=0,
            patch_size=1000,
            allow_empty=False,
            save_dir="/orange/ewhite/DeepForest/Tonga/crops/")

for image in split_files.image_path.unique():
    image_df = split_files[split_files.image_path==image]
    image_df.root_dir = "/orange/ewhite/DeepForest/Tonga/crops"
    image_df["score"] = 1
    plot_results(image_df)

split_files["image_path"] = split_files["image_path"].apply(lambda x: os.path.join("/orange/ewhite/DeepForest/Tonga/crops/", x))
split_files.to_csv("/orange/ewhite/DeepForest/Tonga/annotations.csv")



