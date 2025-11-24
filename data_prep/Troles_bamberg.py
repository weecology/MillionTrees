import pandas as pd
from deepforest import get_data
import glob
import json
import shapely.geometry
from PIL import Image
from deepforest.visualize import plot_results
from matplotlib import pyplot as plt
import geopandas as gpd

eval_set = "/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/annotations/instances_tree_eval2023.json"
train_set = "/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/annotations/instances_tree_train2023.json"
test_set1 = "/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/annotations/instances_tree_TestSet12023.json"
test_set2 = "/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/annotations/instances_tree_TestSet22023.json"

import os
import shutil

target_dir = "/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/images"

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    image_files = glob.glob("/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/**/*.tif", recursive=True)
    
    for file in image_files:
        shutil.copy(file, target_dir)
else:
    print("Target directory already exists.")
    

def create_shapely_polygons_from_coco_segmentation_json(json_file):
  """Creates Shapely polygons from COCO segmentation JSON.

  Args:
    json_file: The path to the COCO segmentation JSON file.

  Returns:
    A list of Shapely polygons.
  """

  with open(json_file, "r") as f:
    coco_data = json.load(f)

    polygons = []
    filenames = []
    image_ids = {}
    for image in coco_data["images"]:
        image_ids[image["id"]] = image["file_name"]

    for annotation in coco_data["annotations"]:
        segmentation_mask = annotation["segmentation"][0]
        pairs = [(segmentation_mask[i], segmentation_mask[i+1]) for i in range(0, len(segmentation_mask), 2)]
        polygon = shapely.geometry.Polygon(pairs)
        wkt = polygon.wkt
        filenames.append(image_ids[annotation["image_id"]])
        polygons.append(wkt)
    df = pd.DataFrame({"image_path": filenames, "geometry": polygons})

  return df

train_polygons = create_shapely_polygons_from_coco_segmentation_json(train_set)
eval_polygons = create_shapely_polygons_from_coco_segmentation_json(eval_set)
test1_polygons = create_shapely_polygons_from_coco_segmentation_json(test_set1)
test2_polygons = create_shapely_polygons_from_coco_segmentation_json(test_set2)

train_polygons["existing_split"] = "train"
eval_polygons["existing_split"] = "eval"
test1_polygons["existing_split"] = "test1"
test2_polygons["existing_split"] = "test2"

df = pd.concat([train_polygons, eval_polygons, test1_polygons, test2_polygons])
df["source"] = "Troles et al. 2024"
# Make filenames full path 
df["image_path"] = df["image_path"].apply(lambda x: os.path.join("/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/images", x))
df.to_csv("/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/annotations/annotations.csv", index=False)

sample_image = train_polygons.head()
sample_image["label"] = "Tree"
# Read and get the height and width of the image
image_path = sample_image["image_path"].iloc[0]
image = Image.open("/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/images/" + image_path)
width, height = image.size
print(f"Image size: {width} x {height}")
sample_image.root_dir = "/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/images"
sample_image["geometry"] = sample_image["geometry"].apply(shapely.wkt.loads)
gdf = gpd.GeoDataFrame(sample_image, geometry="geometry", crs="EPSG:4326")
ax = plot_results(sample_image,width=width, height=height, savedir="/orange/ewhite/DeepForest/Troles_Bamberg/", basename="sample_image")
plt.savefig("current.png")