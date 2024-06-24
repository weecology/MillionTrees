import os
import glob
import shutil
import pandas as pd
from scipy.io import loadmat

def Treeformer():
    def convert_mat(path):
        f = loadmat(x)
        points = f["image_info"][0][0][0][0][0]
        df = pd.DataFrame(points,columns=["x","y"])
        df["label"] = "Tree"
        df["source"] = "Amirkolaee et al. 2023"
        image_path = "_".join(os.path.splitext(os.path.basename(x))[0].split("_")[1:])
        image_path = "{}.jpg".format(image_path)
        image_dir = os.path.dirname(os.path.dirname(x))
        df["image_path"] = "{}/images/{}".format(image_dir, image_path)
        
        return df

    test_gt = glob.glob("/orange/ewhite/DeepForest/TreeFormer/test_data/ground_truth/*.mat")
    test_ground_truth = []
    for x in test_gt:
        df = convert_mat(x)
        test_ground_truth.append(df)
    test_ground_truth = pd.concat(test_ground_truth)

    train_gt = glob.glob("/orange/ewhite/DeepForest/TreeFormer/train_data/ground_truth/*.mat")
    train_ground_truth = []
    for x in train_gt:
        df = convert_mat(x)
        train_ground_truth.append(df)
    train_ground_truth = pd.concat(train_ground_truth)
    
    val_gt = glob.glob("/orange/ewhite/DeepForest/TreeFormer/valid_data/ground_truth/*.mat")
    val_ground_truth = []
    for x in val_gt:
        df = convert_mat(x)
        val_ground_truth.append(df)
    val_ground_truth = pd.concat(val_ground_truth)

    test_ground_truth.to_csv("/orange/ewhite/DeepForest/TreeFormer/all_images/test.csv")
    train_ground_truth.to_csv("/orange/ewhite/DeepForest/TreeFormer/all_images/train.csv")
    val_ground_truth.to_csv("/orange/ewhite/DeepForest/TreeFormer/all_images/validation.csv")

    #Label splits and recombine
    test_ground_truth["split"] = "test"
    train_ground_truth["split"] = "train"
    val_ground_truth["split"] = "validation"
    annotations = pd.concat([test_ground_truth, train_ground_truth, val_ground_truth])

    # Create wkt geometries
    annotations["geometry"] = annotations.apply(lambda x: "POINT ({} {})".format(x.x, x.y), axis=1)
    annotations.to_csv("/orange/ewhite/DeepForest/TreeFormer/all_images/annotations.csv")

if __name__ == "__main__":
    Treeformer()