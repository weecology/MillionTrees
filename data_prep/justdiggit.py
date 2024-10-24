import json
import pandas as pd
from deepforest.utilities import read_file
from deepforest.visualize import plot_results
from matplotlib import pyplot as plt

with open("/orange/ewhite/DeepForest/justdiggit-drone/label_sample/Annotations_trees_only.json") as jsonfile:
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

annotations["image_path"] = "/orange/ewhite/DeepForest/justdiggit-drone/label_sample/" + annotations["image_path"]
annotations = read_file(annotations, root_dir="/orange/ewhite/DeepForest/justdiggit-drone/label_sample/")

for image in annotations.image_path.unique():
    print(image)
    gdf = annotations[annotations.image_path == image]
    gdf.root_dir = "/orange/ewhite/DeepForest/justdiggit-drone/label_sample/"
    plot_results(gdf)
    plt.savefig("current.png")
annotations.to_csv("/orange/ewhite/DeepForest/justdiggit-drone/label_sample/train.csv")