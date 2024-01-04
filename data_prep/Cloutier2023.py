import glob
import os
import pandas as pd
from deepforest.utilities import read_file
from deepforest.preprocess import split_raster
import geopandas as gpd
import shutil

def Cloutier2023():
    # Zone 3 is test, Zone 1 and 2 is train. Intentionally vary window size.

    drone_flights = glob.glob("/blue/ewhite/DeepForest/Cloutier2023/**/*.tif",recursive=True)
    zone1 = "/blue/ewhite/DeepForest/Cloutier2023/quebec_trees_dataset_2021-06-17/Z1_polygons.gpkg"
    zone2 = "/blue/ewhite/DeepForest/Cloutier2023/quebec_trees_dataset_2021-06-17/Z2_polygons.gpkg"
    zone3 = "/blue/ewhite/DeepForest/Cloutier2023/quebec_trees_dataset_2021-06-17/Z3_polygons.gpkg"
    
    train = []
    test = []
    for flight in drone_flights:
            train_zone1 = read_file(zone1, rgb=flight)
            split_annotations_1 = split_raster(train_zone1, path_to_raster=flight, patch_size=1000, allow_empty=True, base_dir="/blue/ewhite/DeepForest/Cloutier2023/images/")
            train_zone2 = read_file(zone2, rgb=flight)
            split_annotations_2 = split_raster(train_zone2, path_to_raster=flight, patch_size=2000, allow_empty=True, base_dir="/blue/ewhite/DeepForest/Cloutier2023/images/")
            test_zone3 = read_file(zone3, rgb=flight)
            split_annotations_3 = split_raster(test_zone3, path_to_raster=flight, patch_size=1500, allow_empty=True, base_dir="/blue/ewhite/DeepForest/Cloutier2023/images/")
            train.append(split_annotations_1)
            train.append(split_annotations_2)
            test.append(split_annotations_3)
    train = pd.concat(train)
    test = pd.concat(test)

    test.to_csv("/blue/ewhite/DeepForest/Cloutier2023/images/test.csv")
    train.to_csv("/blue/ewhite/DeepForest/Cloutier2023/images/train.csv")

    #combined
    train["split"] = "train"
    test["split"] = "test"
    combined = pd.concat([train,test])
    combined["source"] = "Cloutier et al. 2023"
    combined.to_csv("/blue/ewhite/DeepForest/Cloutier2023/images/annotations.csv")

    # Copy to MillionTrees dir
    for path in combined.image_path.unique():
        shutil.copy(path, "/blue/ewhite/DeepForest/MillionTrees/images/")

if __name__ == "__main__":
    Cloutier2023()

