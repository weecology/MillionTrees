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
        points = read_file(x)
        points["image_path"] = os.path.splitext(os.path.basename(x))[0]
        df.append(points)
    annotations = pd.concat(df)
    annotations["label"] = "Tree"
    annotations["source"] = "Ventura et al. 2022" 

    annotations.loc[annotations.image_path.isin(train_images),"split"] = "train"
    annotations.loc[annotations.image_path.isin(test_images),"split"] = "test"
    annotations.loc[annotations.image_path.isin(val_images),"split"] = "val"

    annotations.to_csv("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/annotations.csv")

    #split
    train = annotations[annotations.image_path.isin(train_images)]
    test = annotations[annotations.image_path.isin(test_images)]
    val = annotations[annotations.image_path.isin(val_images)]
              
    train.to_csv("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/train.csv")
    test.to_csv("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/test.csv")
    val.to_csv("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/val.csv")

    # Copy to MillionTrees folder
    for image_path in annotations.image_path.unique():
        src = os.path.join("/blue/ewhite/DeepForest/TreeFormer/test_data/images/", image_path)
        dst = os.path.join("/blue/ewhite/DeepForest/MillionTrees/images/", image_path)
        shutil.copy(src, dst)

if __name__ == "__main__":
    Ventura2022()