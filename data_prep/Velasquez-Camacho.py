import glob
import pandas as pd
from deepforest.utilities import read_file
import os
import rasterio
from rasterio.enums import Resampling
import numpy as np
from PIL import Image

def Velasquez2023():
    xmls = glob.glob("/orange/ewhite/DeepForest/Velasquez_urban_trees/tree_canopies/nueva_carpeta/*.xml")

    tif_files = glob.glob("/orange/ewhite/DeepForest/Velasquez_urban_trees/tree_canopies/nueva_carpeta/*.tif")

    for tif_path in tif_files:
        with rasterio.open(tif_path) as src:
            array = src.read()
            png_path = tif_path.replace(".tif", ".jpg")
            with rasterio.open(png_path, 'w', height=array.shape[1], width=array.shape[2], count=3, dtype=array.dtype) as dst:
                dst.write(array)
    
    # read the pngs and save them as jpgs
    png_files = glob.glob("/orange/ewhite/DeepForest/Velasquez_urban_trees/tree_canopies/nueva_carpeta/*.png")
    for png_path in png_files:
        im = Image.open(png_path)
        im = im.convert("RGB")
        jpg_path = png_path.replace(".png", ".jpg")
        im.save(jpg_path)

    #Load and format xmls
    annotation_list = []   
    for xml_path in xmls:
        try:
            annotation = read_file(xml_path)
            annotation["image_path"] = os.path.basename(xml_path).replace(".xml", ".jpg")
        except:
            continue
        annotation_list.append(annotation)
    annotations = pd.concat(annotation_list, ignore_index=True)      
    annotations["label"] = "Tree"
    annotations["source"] = "Velasquez-Camacho et al. 2023"  
    
    # Add full path to images
    annotations["image_path"] = annotations.image_path.apply(lambda x: "/orange/ewhite/DeepForest/Velasquez_urban_trees/tree_canopies/nueva_carpeta/{}".format(x))
    annotations.to_csv("/orange/ewhite/DeepForest/Velasquez_urban_trees/tree_canopies/nueva_carpeta/annotations.csv")

if __name__ == "__main__":
    Velasquez2023()