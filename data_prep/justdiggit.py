import json
import pandas as pd
from deepforest.utilities import read_file
from deepforest.visualize import plot_results
from matplotlib import pyplot as plt
import rasterio as rio

with open("/orange/ewhite/DeepForest/justdiggit-drone/label_sample/Annotations_trees_only.json") as jsonfile:
    data = json.load(jsonfile)    
    ids = [x["id"] for x in data["images"]]
    image_paths = [x["file_name"] for x in data["images"]]
    id_dict = {key:value for key, value in zip(ids,image_paths)}
    annotation_df = []
    for row in data["annotations"]:
        polygon = row["segmentation"][0]

        # split each pair into a tuple
        as_tuples = [(polygon[i],polygon[i+1]) for i in range(0,len(polygon)-1,2)]

        # close the polygon
        as_tuples.append(as_tuples[0])

        # convert polygon to wkt format
        polygon_coordinates = ",".join(["{} {}".format(i[0],i[1]) for i in as_tuples])
        polygon = "POLYGON ((" + polygon_coordinates + "))"

        b = {"id":row["id"],"polygon":polygon}
        b["image_path"] = id_dict[row["image_id"]]
        annotation_df.append(b)
annotation_df = pd.DataFrame(annotation_df)
annotation_df["label"] = "Tree"
annotation_df["source"] = "Justdiggit 2023"

print("There are {} annotations in {} images".format(annotation_df.shape[0], len(annotation_df.image_path.unique())))

annotations = read_file(annotation_df, root_dir="/orange/ewhite/DeepForest/justdiggit-drone/label_sample/")

for image in annotations.image_path.unique():
    print(image)
    gdf = annotations[annotations.image_path == image]
    gdf.root_dir = "/orange/ewhite/DeepForest/justdiggit-drone/label_sample/"
    width, height = rio.open(gdf.root_dir + image).shape

    #plot_results(gdf, height=height, width=width)
    #plt.savefig("current.png")

annotations["image_path"] = "/orange/ewhite/DeepForest/justdiggit-drone/label_sample/" + annotations["image_path"]
annotations.to_csv("/orange/ewhite/DeepForest/justdiggit-drone/label_sample/annotations.csv")