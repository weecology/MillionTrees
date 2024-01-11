from deepforest.preprocess import split_raster, read_file
import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
import os

def clean_up_rgb():
    rgb = "/blue/ewhite/DeepForest/Hickman2021/RCD105_MA14_21_orthomosaic_20141023_reprojected_full_res_crop1.tif"
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

    with rasterio.open("/blue/ewhite/DeepForest/Hickman2021/RCD105_MA14_21_orthomosaic_20141023_reprojected_full_res_crop1_rgb_corrected.tif", 'w', **meta) as dst:
        dst.write(r)

    rgb = "/blue/ewhite/DeepForest/Hickman2021/RCD105_MA14_21_orthomosaic_20141023_reprojected_full_res_crop2.tif"
    src = rasterio.open(rgb)
    r = src.read()
    print(r.shape)
    r = r[:3,:,:]
    r = r/65535.0 * 255
    r[np.isnan(r)] = 0
    r = r.astype(int)

    # Save raster
    meta = src.meta.copy()
    meta.update(count = 3)
    meta.update(dtype=rasterio.uint8)
    meta.update(nodata=0)

    with rasterio.open("/blue/ewhite/DeepForest/Hickman2021/RCD105_MA14_21_orthomosaic_20141023_reprojected_full_res_crop2_rgb_corrected.tif", 'w', **meta) as dst:
        dst.write(r)

def Hickman2021():
    rgb = "/blue/ewhite/DeepForest/Hickman2021/RCD105_MA14_21_orthomosaic_20141023_reprojected_full_res_crop1_rgb_corrected.tif"
    shp = "/blue/ewhite/DeepForest/Hickman2021/manual_crowns_sepilok.shp"
    gdf = gpd.read_file(shp)
    gdf["image_path"] = rgb
    gdf["label"] = "Tree"
    annotations = read_file(gdf)
    annotations = annotations[annotations.is_valid]
    annotations["image_path"] = os.path.basename(rgb)
    train_annotations = split_raster(
        annotations,
        path_to_raster=rgb,
        patch_size=1000,
        allow_empty=False,
        base_dir="/blue/ewhite/DeepForest/Hickman2021/pngs/")
    
    rgb = "/blue/ewhite/DeepForest/Hickman2021/RCD105_MA14_21_orthomosaic_20141023_reprojected_full_res_crop2_rgb_corrected.tif"
    annotations["image_path"] = os.path.basename(rgb)
    test_annotations = split_raster(
        annotations,
        path_to_raster=rgb,
        patch_size=1000,
        allow_empty=False,  
        base_dir="/blue/ewhite/DeepForest/Hickman2021/pngs/")

    test_annotations["split"] = "test"
    train_annotations["split"] = "train"
    annotations = pd.concat([test_annotations, train_annotations])
    annotations.to_csv("/blue/ewhite/DeepForest/Hickman2021/annotations.csv")

if __name__ == "__main__":
    #clean_up_rgb()
    Hickman2021()