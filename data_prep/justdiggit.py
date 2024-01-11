import json
import pandas as pd
import random

def justdiggit():
    with open("/blue/ewhite/DeepForest/justdiggit-drone/label_sample/Annotations_trees_only.json") as jsonfile:
        data = json.load(jsonfile)    
        ids = [x["id"] for x in data["images"]]
        image_paths = [x["file_name"] for x in data["images"]]
        id_df = pd.DataFrame({"id":ids,"image_path":image_paths})
        annotation_df = []
        for row in data["annotations"]:
            b = {"id":row["id"],"xmin":row["bbox"][0],"ymin":row["bbox"][1],"xmax":row["bbox"][2],"ymax":row["bbox"][3]}
            annotation_df.append(b)
    annotation_df = pd.DataFrame(annotation_df)
    annotations = annotation_df.merge(id_df)
    annotations["label"] = "Tree"
    annotations["source"] = "Justdiggit et al. 2023"

    print("There are {} annotations in {} images".format(annotations.shape[0], len(annotations.image_path.unique())))
    images = annotations.image_path.unique()
    random.shuffle(images)
    train_images = images[0:int(len(images)*0.8)]
    train = annotations[annotations.image_path.isin(train_images)]
    test = annotations[~(annotations.image_path.isin(train_images))]    

    train.to_csv("/blue/ewhite/DeepForest/justdiggit-drone/label_sample/train.csv")
    test.to_csv("/blue/ewhite/DeepForest/justdiggit-drone/label_sample/test.csv")