import pandas as pd
import globox
from deepforest.utilities import xml_to_annotations
from deepforest import get_data
import glob
import json
import shapely.geometry

eval_set = "/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/annotations/instances_tree_eval2023.json"
train_set = "/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/annotations/instances_tree_train2023.json"
test_set1 = "/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/annotations/instances_tree_TestSet12023.json"
test_set2 = "/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/annotations/instances_tree_TestSet22023.json"

import os
import shutil

def copy_images(source_dir, target_dir, extensions=('.jpg', '.png', '.jpeg')):
    """
    Recursively copy images from source_dir to target_dir.
    
    Parameters:
    - source_dir: Path to the source directory.
    - target_dir: Path to the target directory.
    - extensions: A tuple of file extensions to consider as images.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(extensions):
                source_file_path = os.path.join(root, file)
                target_file_path = os.path.join(target_dir, file)
                
                # Ensure the target directory exists
                os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
                
                # Copy the file
                shutil.copy2(source_file_path, target_file_path)
                print(f"Copied: {source_file_path} to {target_file_path}")

# Example usage
source_directory = "/orange/ewhite/DeepForest/Troles_Bamberg/coco2048"
target_directory = "/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/images"

# If source dir exists, don't run
if not os.path.exists(target_directory):
    print("Copying images...")
    copy_images(source_directory, target_directory, extensions=(".tif"))

"""
# If we wanted just the bounding boxes
    coco_train = globox.AnnotationSet.from_coco(train_set)
    coco_train.save_pascal_voc("/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/annotations/instances_tree_train2023")
    xmls = glob.glob("/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/annotations/instances_tree_train2023/*.xml")
    train_records = []
    for xml in xmls:
        try:
            df = xml_to_annotations(xml)
            train_records.append(df)
        except Exception as e:
            print(f"Failed to process file: {xml}")
            print(f"Error: {e}")
    train_records = pd.concat(train_records)
"""


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

train_polygons["split"] = "train"
eval_polygons["split"] = "eval"
test1_polygons["split"] = "test1"
test2_polygons["split"] = "test2"

df = pd.concat([train_polygons, eval_polygons, test1_polygons, test2_polygons])
df["source"] = "Troles et al. 2024"
# Make filenames full path 
df["image_path"] = df["image_path"].apply(lambda x: os.path.join("/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/images", x))
df.to_csv("/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/annotations/annotations.csv", index=False)