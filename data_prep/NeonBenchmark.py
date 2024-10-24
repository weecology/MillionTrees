import glob
import os
import pandas as pd
from deepforest.utilities import read_file

# Define the base path
BASE_PATH = "/orange/ewhite/b.weinstein/NeonTreeEvaluation/hand_annotations/crops"

# Load all CSV files in the specified directory
csv_files = glob.glob(os.path.join(BASE_PATH, "*.csv"))
csv_list = []

for csv_file in csv_files:
    print(csv_file)
    df = read_file(csv_file)
    df["image_path"] = df["image_path"].apply(lambda x: os.path.join(BASE_PATH, x))
    df["source"] = "Weecology_University_Florida"
    df["label"] = "Tree"
    csv_list.append(df)

# Concatenate all CSV dataframes
annotations = pd.concat(csv_list, ignore_index=True)

# Save the combined annotations to a CSV file
output_path = "/orange/ewhite/DeepForest/NEON_benchmark/University_of_Florida.csv"

# Save the combined annotations to a CSV file
annotations.to_csv(output_path, index=False)

# Load the existing annotations file
existing_annotations_path = "/orange/ewhite/DeepForest/NEON_benchmark/NeonTreeEvaluation_annotations.csv"
existing_annotations = pd.read_csv(existing_annotations_path)

# Check for overlapping data based on a common column, e.g., 'image_path'
overlapping_data = pd.merge(annotations, existing_annotations, on='image_path', how='inner')

# Print the overlapping data
print("Overlapping data:")
print(overlapping_data)
annotations.to_csv(output_path, index=False)


