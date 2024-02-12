import glob
import pandas as pd
import os
from deepforest.utilities import read_file
import shutil

def Ventura2022():
    all_csvs = glob.glob("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/csv/*.csv")
    train_images = pd.read_csv("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/train.txt",header=None,sep=" ")[0].values
    test_images = pd.read_csv("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/test.txt",header=None,sep=" ")[0].values
    val_images = pd.read_csv("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/val.txt",header=None,sep=" ")[0].values

    df = []
    for x in all_csvs:
        pointdf = pd.read_csv(x)
        pointdf["image_path"] = "{}.tif".format(os.path.splitext(os.path.basename(x))[0])
        points = read_file(pointdf)
        df.append(points)
    annotations = pd.concat(df)
    annotations["label"] = "Tree"
    annotations["source"] = "Ventura et al. 2022" 

    annotations.loc[annotations.image_path.apply(lambda x: os.path.splitext(x)[0]).isin(train_images),"split"] = "train"
    annotations.loc[annotations.image_path.apply(lambda x: os.path.splitext(x)[0]).isin(test_images),"split"] = "test"
    annotations.loc[annotations.image_path.apply(lambda x: os.path.splitext(x)[0]).isin(val_images),"split"] = "val"

    # Split into x and y
    annotations["x"] = annotations.geometry.apply(lambda x: x.x)
    annotations["y"] = annotations.geometry.apply(lambda x: x.y)
    
    annotations["image_path"] = annotations["image_path"].apply(lambda x: os.path.join("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images",x))
    annotations.to_csv("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/annotations.csv")

    #split
    train = annotations[annotations.split=="train"]
    test = annotations[annotations.split=="test"]
    val = annotations[annotations.split=="val"]
              
    train.to_csv("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/train.csv")
    test.to_csv("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/test.csv")
    val.to_csv("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/val.csv")

if __name__ == "__main__":
    Ventura2022()