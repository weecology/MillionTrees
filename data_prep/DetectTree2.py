from deepforest.preprocess import split_raster, read_file
from deepforest.visualize import plot_results
import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
import random
import os
import cv2

def clean_up_rgb():
    rgb = "/orange/ewhite/DeepForest/DetectTree2/Sep_MA14_21_orthomosaic_20141023_reprojected_full_res.tif"
    src = rasterio.open(rgb)
    r = src.read()
    print(r.shape)
    r = r[:3,:,:]
    r = r/65535.0 * 255
    # Set no data to 0
    r[np.isnan(r)] = 0
    r = r.astype(int)

    # Save raster
    meta = src.meta.copy()
    meta.update(count = 3)
    meta.update(dtype=rasterio.uint8)
    meta.update(nodata=0)

    with rasterio.open("/orange/ewhite/DeepForest/DetectTree2/Sep_MA14_21_orthomosaic_20141023_reprojected_full_res_corrected.tif", 'w', **meta) as dst:
        dst.write(r)

def generate():
    rgb = "/orange/ewhite/DeepForest/DetectTree2/RCD105_MA14_21_orthomosaic_20141023_reprojected_full_res_crop1_rgb_corrected.tif"
    shps = ["/orange/ewhite/DeepForest/DetectTree2/sep_east.shp", "/orange/ewhite/DeepForest/DetectTree2/sep_west.shp"]
    all_annotations = []
    for shp in shps:
        gdf = gpd.read_file(shp)
        gdf["image_path"] = rgb
        gdf["label"] = "Tree"
        annotations = read_file(gdf)
        annotations = annotations[annotations.is_valid]
        annotations["image_path"] = os.path.basename(rgb)
        annotations = read_file(annotations, root_dir="/orange/ewhite/DeepForest/DetectTree2/")
        all_annotations.append(annotations)
    all_annotations = pd.concat(all_annotations)
    
    crop_anotations = split_raster(
        all_annotations,
        path_to_raster=rgb,
        patch_size=1500,
        allow_empty=False,
        base_dir="/orange/ewhite/DeepForest/DetectTree2/pngs/")
    
    # Make full path
    crop_anotations["image_path"] = "/orange/ewhite/DeepForest/DetectTree2/pngs/" + annotations["image_path"]
    crop_anotations["source"] = "Ball et al. 2023"
    crop_anotations.to_csv("/orange/ewhite/DeepForest/DetectTree2/annotations.csv")

    return annotations

if __name__ == "__main__":
    clean_up_rgb()
    annotations_base_path = generate()    
    annotations_base_path["image_path"] = annotations_base_path["image_path"].apply(lambda x: os.path.basename(x))
    annotations_base_path.root_dir = "/orange/ewhite/DeepForest/DetectTree2/pngs/"

    # plot 5 samples in a panel  
    images_to_plot = random.sample(annotations_base_path.image_path.unique().tolist(), 5)
    for image in images_to_plot:
        df_to_plot = annotations_base_path[annotations_base_path.image_path == image]
        df_to_plot = read_file(df_to_plot)
        df_to_plot.root_dir = "/orange/ewhite/DeepForest/DetectTree2/pngs/"
        height, width, channels = cv2.imread(df_to_plot.root_dir + df_to_plot.image_path.iloc[0]).shape
        plot_results(df_to_plot, height=height,width=width)