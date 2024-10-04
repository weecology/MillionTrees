import os
import pandas as pd
from deepforest.utilities import read_file
import json

# Define the directory containing the JSON files
json_dir = "/orange/ewhite/DeepForest/Guangzhou2022/GZIndividualTree_Anno"

# Initialize an empty list to store data
data = []

# Iterate over all JSON files in the directory
for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(json_dir, filename)
        # Read the JSON file

        with open(file_path, 'r') as f:
            coco_data = json.load(f)
        
        # Extract annotations
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            image_info = next(item for item in coco_data['images'] if item['id'] == image_id)
            if 'treeInstance300' in file_path:
                image_path = os.path.join(json_dir, 'train', image_info['file_name'])
            elif 'test' in file_path:
                image_path = os.path.join(json_dir, 'test', image_info['file_name'])
            else:
                image_path = os.path.join(json_dir, image_info['file_name'])
            
            if not os.path.exists(image_path):
                continue

            xmin = annotation['bbox'][0]
            ymin = annotation['bbox'][1]
            xmax = xmin + annotation['bbox'][2]
            ymax = ymin + annotation['bbox'][3]
            
            # Append the data to the list
            data.append([image_path, xmin, xmax, ymin, ymax])

# Create a DataFrame
df = pd.DataFrame(data, columns=['image_path', 'xmin', 'xmax', 'ymin', 'ymax'])
df = read_file(df)
df["label"] = "Tree"
df["source"] = "Sun et al. 2022"

# Save the DataFrame to a CSV file
df.to_csv('/orange/ewhite/DeepForest/Guangzhou2022/annotations.csv', index=False)