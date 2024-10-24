import os
import json
import pandas as pd
import geopandas as gpd
from deepforest import utilities

def extract_annotations(json_path, image_dir):
    with open(json_path) as f:
        data = json.load(f)
    
    annotations = []
    for key, value in data.items():
        filename = value['filename']
        regions = value['regions']
        for key, value in regions.items():
            shape_attributes = value['shape_attributes']
            all_points_x = shape_attributes['all_points_x']
            all_points_y = shape_attributes['all_points_y']
            polygon = [(x, y) for x, y in zip(all_points_x, all_points_y)]
            annotations.append({
                'image_path': os.path.join(image_dir, filename),
                'polygon': polygon
            })
    
    df = pd.DataFrame(annotations)
    # Convert polygon to well known text string
    df['polygon'] = df['polygon'].apply(lambda x: f"POLYGON(({', '.join([f'{p[0]} {p[1]}' for p in x])}))")
    gdf = utilities.read_file(df)

    return gdf

# Paths to the annotation files and directories
train_json_path = '/orange/ewhite/DeepForest/OliveTrees_spain/Dataset_RGB/train/via_region_data.json'
val_json_path = '/orange/ewhite/DeepForest/OliveTrees_spain/Dataset_RGB/val/via_region_data.json'
train_image_dir = '/orange/ewhite/DeepForest/OliveTrees_spain/Dataset_RGB/train'
val_image_dir = '/orange/ewhite/DeepForest/OliveTrees_spain/Dataset_RGB/val'

# Extract annotations
train_annotations = extract_annotations(train_json_path, train_image_dir)
val_annotations = extract_annotations(val_json_path, val_image_dir)

# Concatenate annotations
all_annotations = pd.concat([train_annotations, val_annotations], ignore_index=True)

# Save to CSV
output_csv_path = '/orange/ewhite/DeepForest/OliveTrees_spain/Dataset_RGB/annotations.csv'
all_annotations["source"] = "Safonova et al. 2021"
all_annotations["label"] = "Tree"
all_annotations.to_csv(output_csv_path, index=False)