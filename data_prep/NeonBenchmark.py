import glob
import os
import pandas as pd
import shutil
import geopandas as gpd
from deepforest.utilities import read_file

## Train annotations ##
BASE_PATH = "/orange/ewhite/b.weinstein/NeonTreeEvaluation/hand_annotations/"
#convert hand annotations from xml into retinanet format
xmls = glob.glob(BASE_PATH + "*.xml")
annotation_list = []
for xml in xmls:
    #check if it is in the directory
    image_name = "{}.tif".format(os.path.splitext(os.path.basename(xml))[0])
    if os.path.exists(os.path.join(BASE_PATH, image_name)):
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
    gdf["image_path"] = os.path.join(BASE_PATH, rgb)
    shp_df = read_file(gdf, root_dir=BASE_PATH)
    shp_df = pd.DataFrame(shp_df)        
    shp_results.append(shp_df)

shp_results = pd.concat(shp_results, ignore_index=True)
annotations = pd.concat([annotations, shp_results])

#Ensure column order
annotations["source"] = "Weecology_University_Florida"
annotations["label"] = "Tree"
annotations["image_path"] = annotations.image_path.apply(lambda x: os.path.join("/orange/ewhite/DeepForest/NEON_benchmark/images/", x))

annotations.to_csv("/orange/ewhite/DeepForest/NEON_benchmark/University_of_Florida.csv")