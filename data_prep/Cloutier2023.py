import glob
import pandas as pd
from preprocess_polygons import split_raster_with_polygons
import geopandas as gpd
from utilities import read_file
from deepforest.visualize import plot_predictions
import cv2
import os

def Cloutier2023():
    # Zone 3 is test, Zone 1 and 2 is train. Intentionally vary window size.

    drone_flights = glob.glob("/orange/ewhite/DeepForest/Cloutier2023/**/*.tif",recursive=True)
    zone1 = "/orange/ewhite/DeepForest/Cloutier2023/quebec_trees_dataset_2021-06-17/Z1_polygons.gpkg"
    zone2 = "/orange/ewhite/DeepForest/Cloutier2023/quebec_trees_dataset_2021-06-17/Z2_polygons.gpkg"
    zone3 = "/orange/ewhite/DeepForest/Cloutier2023/quebec_trees_dataset_2021-06-17/Z3_polygons.gpkg"
    
    train = []
    test = []
    for flight in drone_flights:
            train_zone1 = read_file(zone1,rgb=flight)
            split_annotations_1 = split_raster_with_polygons(train_zone1, path_to_raster=flight, patch_overlap=0, patch_size=3000, allow_empty=False, base_dir="/orange/ewhite/DeepForest/Cloutier2023/images/")
            train_zone2 = read_file(zone2,rgb=flight)
            split_annotations_2 = split_raster_with_polygons(train_zone2, path_to_raster=flight, patch_overlap=0, patch_size=3000, allow_empty=False, base_dir="/orange/ewhite/DeepForest/Cloutier2023/images/")
            test_zone3 = read_file(zone3,rgb=flight)
            split_annotations_3 = split_raster_with_polygons(test_zone3, path_to_raster=flight, patch_overlap=0, patch_size=3000, allow_empty=False, base_dir="/orange/ewhite/DeepForest/Cloutier2023/images/")
            train.append(split_annotations_1)
            train.append(split_annotations_2)
            test.append(split_annotations_3)
    train = pd.concat(train)
    test = pd.concat(test)

    #combined
    train["split"] = "train"
    test["split"] = "test"
    combined = pd.concat([train,test])
    combined["source"] = "Cloutier et al. 2023"

    
    # Make full path
    combined["image_path"] = "/orange/ewhite/DeepForest/Cloutier2023/images/" + combined["image_path"]

    combined.to_csv("/orange/ewhite/DeepForest/Cloutier2023/images/annotations.csv")

    # View one image
    sample_image_annotations = combined[combined.image_path==combined.image_path.unique()[0]] 
    sample_image = cv2.imread(os.path.join("/orange/ewhite/DeepForest/Cloutier2023/images/",sample_image_annotations.image_path.unique()[0]))
    plot_predictions(sample_image,patch_size=3000)



if __name__ == "__main__":
    Cloutier2023()

