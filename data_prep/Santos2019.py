import os
import json
import pandas as pd
from deepforest.utilities import read_file

def load_annotations_from_json(json_path, image_dir):
    """Load annotations from a single JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    image_path = os.path.join(image_dir, os.path.basename(json_path).replace(".json", ""))
    annotations = []

    for obj in data["objects"]:
        xmin = obj["points"]["exterior"][0][0]
        ymin = obj["points"]["exterior"][0][1]
        xmax = obj["points"]["exterior"][1][0]
        ymax = obj["points"]["exterior"][1][1]

        annotations.append([image_path, xmin, ymin, xmax, ymax])

    return annotations

def load_annotations(ann_dir, image_dir):
    """Load annotations from all JSON files in a directory and create a DataFrame."""
    all_annotations = []
    for json_file in os.listdir(ann_dir):
        if json_file.endswith(".json"):
            json_path = os.path.join(ann_dir, json_file)
            annotations = load_annotations_from_json(json_path, image_dir)
            all_annotations.extend(annotations)

    df = pd.DataFrame(all_annotations, columns=["image_path", "xmin", "ymin", "xmax", "ymax"])
    return df

# Load train, test, and val annotations
train_annotations = load_annotations("/orange/ewhite/DeepForest/Santos2019/train/ann", "/orange/ewhite/DeepForest/Santos2019/train/img")
test_annotations = load_annotations("/orange/ewhite/DeepForest/Santos2019/test/ann", "/orange/ewhite/DeepForest/Santos2019/test/img")
val_annotations = load_annotations("/orange/ewhite/DeepForest/Santos2019/val/ann", "/orange/ewhite/DeepForest/Santos2019/val/img")

# Concatenate all annotations
all_annotations = pd.concat([train_annotations, test_annotations, val_annotations], ignore_index=True)

# Add source field
all_annotations["source"] = "Santos et al. 2019"
all_annotations = read_file(all_annotations)
all_annotations = all_annotations[["geometry", "image_path", "source"]]
all_annotations["label"] = "Tree"

# Save to a single annotations file
output_csv_path = "/orange/ewhite/DeepForest/Santos2019/annotations.csv"
all_annotations.to_csv(output_csv_path, index=False)