import geopandas as gpd
from deepforest.preprocess import split_raster
from deepforest.utilities import read_file

# 2020 manually validated crown map
gdf = gpd.read_file("BCI_50ha_2020_08_01_crownmap_formatted.shp")

# Remove Multipolygons
gdf = gdf[gdf.geometry.type == "Polygon"]
gdf["image_path"] = "BCI_50ha_2020_08_01_global.tif"
gdf["label"] = "Tree"
annotations = read_file(gdf)

rgb = "/orange/ewhite/DeepForest/BCI/BCI_50ha_2020_08_01_crownmap_raw/BCI_50ha_2020_08_01_global.tif"
# Cut into crops
crop_annotations = split_raster(
    annotations,
    path_to_raster=rgb,
    patch_size=2000,
    allow_empty=False,
    base_dir="/orange/ewhite/DeepForest/BCI/BCI_50ha_2020_08_01_crownmap_raw/crops")

crop_annotations = crop_annotations[["geometry","image_path"]]
crop_annotations["source"] = "Vasquez et al. 2023"
# Full path
crop_annotations["image_path"] = crop_annotations["image_path"].apply(lambda x: "/orange/ewhite/DeepForest/BCI/BCI_50ha_2020_08_01_crownmap_raw/crops/"+x)
crop_annotations.to_csv("/orange/ewhite/DeepForest/BCI/BCI_50ha_2020_08_01_crownmap_raw/annotations.csv")