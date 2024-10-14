from deepforest import main
from deepforest.utilities import read_file
from deepforest.preprocess import split_raster
import os
import geopandas as gpd
import pandas as pd

gdf = gpd.read_file("/orange/ewhite/DeepForest/Araujo_2020/crown_delineation_shapefile.shp")
gdf =  gdf[gdf.geometry.type=="Polygon"]
gdf["image_path"] = "Orthomosaic_WGS84_UTM20S.tif"
gdf["label"] = "Tree"
gdf["source"] = "Araujo et al. 2020"
df = read_file(gdf, root_dir="/orange/ewhite/DeepForest/Araujo_2020/")
df = df[["geometry", "image_path", "label", "source"]]
df["polygon"] = df.geometry.apply(lambda x: x.wkt)
df.drop(columns=["geometry"], inplace=True)
df = pd.DataFrame(df)
split_files = split_raster(df, path_to_raster="/orange/ewhite/DeepForest/Araujo_2020/Orthomosaic_WGS84_UTM20S.tif", root_dir="/orange/ewhite/DeepForest/Araujo_2020/",
                           base_dir="/orange/ewhite/DeepForest/Araujo_2020/crops/", patch_size=1500, patch_overlap=0)

split_files["image_path"] = split_files["image_path"].apply(lambda x: os.path.join("/orange/ewhite/DeepForest/Araujo_2020/crops/", x))
split_files.to_csv("/orange/ewhite/DeepForest/Araujo_2020/annotations.csv")