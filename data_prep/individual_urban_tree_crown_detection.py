import os
import pandas as pd
from deepforest.utilities import read_file

# Directory containing the txt files
directory = '/orange/ewhite/DeepForest/individual_urban_tree_crown_detection/bbox_txt'

# Initialize an empty list to store data
data = []

# Loop through all txt files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        try:
            content_df = pd.read_csv(file_path, sep=" ", header=None)
        except pd.errors.EmptyDataError:
            print(f"Empty file: {filename}")
            continue
        content_df.columns = ['xmin', 'ymin', 'xmax', 'ymax']
        # Append filename and source to the DataFrame
        content_df['filename'] = filename
        content_df['source'] = 'Zamboni et al. 2021'
        
        # Append to the data list
        data.append(content_df)

# Concatenate all DataFrames into one
df = pd.concat(data, ignore_index=True)

# Load into deepforest after concatenation
df = read_file(df)
df["label"] = "Tree"
df["filename"] = df["filename"].str.replace('.txt', '.png')
df["image_path"] = "/orange/ewhite/DeepForest/individual_urban_tree_crown_detection/rgb/" + df["filename"]

df.drop(columns=['filename'], inplace=True)
# Optionally, save the DataFrame to a CSV file
df.to_csv('/orange/ewhite/DeepForest/individual_urban_tree_crown_detection/annotations.csv', index=False)