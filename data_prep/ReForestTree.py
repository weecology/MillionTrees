import pandas as pd
def ReForestTree():
    """This dataset used deepforest to generate predictions which were cleaned, no test data can be used"""
    annotations = pd.read_csv("/blue/ewhite/DeepForest/ReForestTree/mapping/final_dataset.csv")
    annotations["image_path"] = annotations["img_path"]
    annotations["source"] = "Reiersen et al. 2022"
    annotations[""]
    print("There are {} annotations in {} images".format(annotations.shape[0], len(annotations.image_path.unique())))
    annotations.to_csv("/blue/ewhite/DeepForest/ReForestTree/images/train.csv")
    
