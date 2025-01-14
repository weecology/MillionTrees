import geopandas as gpd
from deepforest.preprocess import split_raster
from deepforest.utilities import read_file

# 2020 manually validated crown map
gdf = gpd.read_file("/orange/ewhite/DeepForest/BCI/BCI_50ha_2020_08_01_crownmap_raw/BCI_50ha_2020_08_01_crownmap_formatted.shp")

# Remove Multipolygons
gdf = gdf[gdf.geometry.type == "Polygon"]
gdf["image_path"] = "BCI_50ha_2020_08_01_global.tif"
gdf["label"] = "Tree"
annotations = read_file(gdf, root_dir="/orange/ewhite/DeepForest/BCI/BCI_50ha_2020_08_01_crownmap_raw/")

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


# Training only polygons, inferred from model + validation development, not for testing

gdf_train = gpd.read_file("/orange/ewhite/DeepForest/BCI/BCI_50ha_2022_09_29_crownmap_raw/BCI_50ha_2022_09_29_crownmap_formatted.shp")
gdf_train = gdf_train[gdf_train.geometry.type == "Polygon"]
gdf_train["image_path"] = "BCI_50ha_2022_09_29_global.tif"
gdf_train["label"] = "Tree"
annotations_train = read_file(gdf_train, root_dir="/orange/ewhite/DeepForest/BCI/BCI_50ha_2022_09_29_crownmap_raw/")
rgb = "/orange/ewhite/DeepForest/BCI/BCI_50ha_2022_09_29_crownmap_raw/BCI_50ha_2022_09_29_global.tif"
crop_annotations_train = split_raster(
    annotations_train,
    path_to_raster=rgb,
    patch_size=2000,
    allow_empty=False,
    base_dir="/orange/ewhite/DeepForest/BCI/BCI_50ha_2022_09_29_crownmap_raw/crops")

crop_annotations_train = crop_annotations_train[["geometry","image_path"]]
crop_annotations_train["source"] = "Vasquez et al. 2023 - training"
crop_annotations_train["image_path"] = crop_annotations_train["image_path"].apply(lambda x: "/orange/ewhite/DeepForest/BCI/BCI_50ha_2022_09_29_crownmap_raw/crops/"+x)
crop_annotations_train.to_csv("/orange/ewhite/DeepForest/BCI/BCI_50ha_2022_09_29_crownmap_raw/annotations.csv")

