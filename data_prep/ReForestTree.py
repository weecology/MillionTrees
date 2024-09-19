import pandas as pd
from deepforest.utilities import read_file
def ReForestTree():
    """This dataset used deepforest to generate predictions which were cleaned, no test data can be used"""
    annotations = pd.read_csv("/orange/ewhite/DeepForest/ReForestTree/mapping/final_dataset.csv")
    annotations["image_path"] = "/orange/ewhite/DeepForest/ReForestTree/images/" + annotations["img_path"]
    annotations["source"] = "Reiersen et al. 2022"
    annotations["label"] = "Tree"
    print("There are {} annotations in {} images".format(annotations.shape[0], len(annotations.image_path.unique())))

    # set all coordinates to int
    annotations["xmin"] = annotations["xmin"].astype(int)
    annotations["ymin"] = annotations["ymin"].astype(int)
    annotations["xmax"] = annotations["xmax"].astype(int)
    annotations["ymax"] = annotations["ymax"].astype(int)

    annotations = read_file(annotations[["image_path", "xmin", "ymin", "xmax", "ymax", "label", "source"]])
    annotations.to_csv("/orange/ewhite/DeepForest/ReForestTree/images/train.csv")

ReForestTree()
