import random
import glob
import os
import pandas as pd
import shutil
import rasterio as rio
import xml.etree.ElementTree as ET

from deepforest.utilities import crop_raster, read_file
from shapely.geometry import Point, box

def read_xml_Beloiu(path):
    tree = ET.parse(path)
    root = tree.getroot()

    # Initialize lists to store data
    filename_list = []
    name_list = []
    xmin_list = []
    ymin_list = []
    xmax_list = []
    ymax_list = []

    # Iterate through each 'object' element
    for obj in root.findall('.//object'):
        filename = root.find('.//filename').text
        name = obj.find('name').text
        xmin = float(obj.find('bndbox/xmin').text)
        ymin = float(obj.find('bndbox/ymin').text)
        xmax = float(obj.find('bndbox/xmax').text)
        ymax = float(obj.find('bndbox/ymax').text)
        
        # Append data to lists
        filename_list.append(filename)
        name_list.append(name)
        xmin_list.append(xmin)
        ymin_list.append(ymin)
        xmax_list.append(xmax)
        ymax_list.append(ymax)

    # Create a DataFrame
    data = {
        'image_path': filename_list,
        'name': name_list,
        'xmin': xmin_list,
        'ymin': ymin_list,
        'xmax': xmax_list,
        'ymax': ymax_list
    }

    df = pd.DataFrame(data)

    return df


def Beloiu_2023():
    xmls = glob.glob("/orange/ewhite/DeepForest/Beloiu_2023/labels/*")
    annotations = []
    for path in xmls:
        df = read_xml_Beloiu(path)
        gdf = read_file(df)
        annotations.append(gdf)

    annotations = pd.concat(annotations)
    annotations["label"] = "Tree"
    annotations["source"] = "Beloiu et al. 2023"
    print("There are {} annotations in {} images".format(annotations.shape[0], len(annotations.image_path.unique())))
    images = annotations.image_path.unique()

    # Convert projected tiff to png
    for image in images:
        with rio.open(os.path.join("/orange/ewhite/DeepForest/Beloiu_2023/images",image)) as src:
            bounds = src.bounds
            res = src.res[0]

        # read tif and save as png
        filename = crop_raster(
            bounds=bounds,
            rgb_path=os.path.join("/orange/ewhite/DeepForest/Beloiu_2023/images",image),
            savedir="/orange/ewhite/DeepForest/Beloiu_2023/pngs/",
            filename=os.path.splitext(os.path.basename(image))[0],
            driver="PNG"
            )

    #Set image path to png
    annotations["image_path"] = annotations.image_path.apply(lambda x: os.path.join("/orange/ewhite/DeepForest/Beloiu_2023/pngs/", os.path.splitext(os.path.basename(x))[0] + ".png"))
    random.shuffle(images)
    train_images = images[0:int(len(images)*0.9)]
    train = annotations[annotations.image_path.isin(train_images)]
    test = annotations[~(annotations.image_path.isin(train_images))]

    train["split"] = "train"
    test["split"] = "test"

    to_save = pd.concat([train,test])
    to_save.to_csv("/orange/ewhite/DeepForest/Beloiu_2023/annotations.csv")
    to_save.to_csv("/orange/ewhite/DeepForest/Beloiu_2023/pngs/annotations.csv")

if __name__ == "__main__":
    Beloiu_2023()