import glob
import os
import pandas as pd
import shutil
import geopandas as gpd
from deepforest.utilities import read_file, shapefile_to_annotations
from deepforest.preprocess import split_raster

def generate_NEON_benchmark():
    BENCHMARK_PATH = "/orange/idtrees-collab/NeonTreeEvaluation/"
    tifs = glob.glob(BENCHMARK_PATH + "evaluation/RGB/*.tif")
    xmls = [os.path.splitext(os.path.basename(x))[0] for x in tifs] 
    xmls = [os.path.join(BENCHMARK_PATH, "annotations", x) + ".xml" for x in xmls] 
    
    #Load and format xmls, not every RGB image has an annotation
    annotation_list = []   
    for xml_path in xmls:
        try:
            annotation = read_file(xml_path)
        except:
            continue
        annotation_list.append(annotation)
    benchmark_annotations = pd.concat(annotation_list, ignore_index=True)

    benchmark_annotations["source"] = "NEON_benchmark"
    for image_path in benchmark_annotations.image_path.unique():
        dst = os.path.join(BENCHMARK_PATH, "evaluation/RGB/", image_path)
        shutil.copy(dst, "/blue/ewhite/DeepForest/NEON_benchmark/images/")

    benchmark_annotations.to_csv("/blue/ewhite/DeepForest/NEON_benchmark/images/test.csv")

    # Copy images to test location
    benchmark_annotations["source"] = "NEON_benchmark"
    for image_path in benchmark_annotations.image_path.unique():
        dst = os.path.join(BENCHMARK_PATH, "evaluation/RGB/", image_path)
        shutil.copy(dst, "/blue/ewhite/DeepForest/NEON_benchmark/images/")

    ## Train annotations ##

    BASE_PATH = "/orange/ewhite/b.weinstein/NeonTreeEvaluation/hand_annotations/"
    #convert hand annotations from xml into retinanet format
    xmls = glob.glob(BENCHMARK_PATH + "annotations/" + "*.xml")
    annotation_list = []
    for xml in xmls:
        #check if it is in the directory
        image_name = "{}.tif".format(os.path.splitext(os.path.basename(xml))[0])
        if os.path.exists(os.path.join(BASE_PATH,image_name)):
            print(xml)
            annotation = read_file(xml)
            annotation_list.append(annotation)
        
    #Collect hand annotations
    annotations = pd.concat(annotation_list, ignore_index=True)      
    
    #collect shapefile annotations
    shps = glob.glob(BASE_PATH + "*.shp")
    shps_tifs = glob.glob(BASE_PATH + "*.tif")
    shp_results = []
    for shp in shps: 
        print(shp)
        rgb = "{}.tif".format(os.path.splitext(shp)[0])
        gdf = gpd.read_file(shp)
        gdf["label"] = "Tree"
        shp_df = shapefile_to_annotations(gdf, rgb)
        shp_df = pd.DataFrame(shp_df)        
        shp_results.append(shp_df)
    
    
    shp_results = pd.concat(shp_results,ignore_index=True)
    annotations = pd.concat([annotations,shp_results])
    
    annotations.to_csv(BASE_PATH + "hand_annotations.csv",index=False)
    
    #Collect tiles
    xmls = glob.glob(BASE_PATH + "*.xml")
    xmls = [os.path.splitext(os.path.basename(x))[0] for x in xmls] 
    raster_list = [BASE_PATH + x + ".tif" for x in xmls] 
    raster_list = raster_list + shps_tifs 
    
    cropped_annotations = [ ]
    
    for raster in raster_list:
        try:
            annotations_df= split_raster(path_to_raster=raster,
                                            annotations_file=BASE_PATH + "hand_annotations.csv",
                                            save_dir=BASE_PATH + "crops/",
                                            patch_size=400,
                                            patch_overlap=0.05)
        except ValueError:
            continue
        cropped_annotations.append(annotations_df)
    
    ##Gather annotation files into a single file
    train_annotations = pd.concat(cropped_annotations, ignore_index=True)   
    
    #Ensure column order
    train_annotations.to_csv("/blue/ewhite/DeepForest/NEON_benchmark/images/train.csv",index=False, header=True)
   
    train_annotations["source"] = "NEON_benchmark"
    for image_path in train_annotations.image_path.unique():
        dst = os.path.join(BASE_PATH, "crops", image_path)
        shutil.copy(dst, "/blue/ewhite/DeepForest/NEON_benchmark/images/")

    # Copy to main dir
    for image_path in train_annotations.image_path.unique():
        dst = os.path.join(BASE_PATH, "crops", image_path)
        shutil.copy(dst, "/blue/ewhite/DeepForest/MillionTrees/images/")
    
    train_annotations.to_csv("/blue/ewhite/DeepForest/MillionTrees/annotations/NEON_benchmark_train.csv")

    for image_path in benchmark_annotations.image_path.unique():
        dst = os.path.join("/blue/ewhite/DeepForest/NEON_benchmark/images/", image_path)
        shutil.copy(dst, "/blue/ewhite/DeepForest/MillionTrees/images/")
    
    benchmark_annotations.to_csv("/blue/ewhite/DeepForest/MillionTrees/annotations/NEON_benchmark_test.csv")

if __name__ == "__main__":
    generate_NEON_benchmark()