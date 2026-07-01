import os
import pandas as pd
from deepforest.utilities import read_file

# Define the base path
BASE_PATH = "/orange/ewhite/b.weinstein/NeonTreeEvaluation/hand_annotations/crops"

# hand_annotations.csv is the master annotation file: it already aggregates every
# per-site crop CSV in this directory (verified: all per-site rows are a subset of
# it, and it carries additional rows the per-site files lack). Globbing *.csv and
# concatenating therefore counted each annotation twice -- once from the per-site
# file and once from the master -- inflating the published Weecology ground truth
# with exact-duplicate boxes. Read only the master to avoid the double-count.
MASTER_CSV = os.path.join(BASE_PATH, "hand_annotations.csv")

print(MASTER_CSV)
annotations = read_file(MASTER_CSV)
annotations["image_path"] = annotations["image_path"].apply(
    lambda x: os.path.join(BASE_PATH, x))
annotations["source"] = "Weecology_University_Florida"
annotations["label"] = "Tree"

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


