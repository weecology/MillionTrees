import os
import pandas as pd
from deepforest import utilities
from PIL import Image

# Define the root directory
root_dir = "/orange/ewhite/DeepForest/WRI/WRI-labels-opensource"

# Initialize an empty list to store dataframes
dataframes = []

# Iterate through each file in the directory
for filename in os.listdir(root_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(root_dir, filename)
        
        # Load the text file into a dataframe
        df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=["index", "x", "y", "width", "height"])
        
        # Load the corresponding image to get its shape
        image_path = os.path.join(root_dir, os.path.splitext(filename)[0] + ".png")
        image = Image.open(image_path)
        original_width, original_height = image.size
        
        # Apply the coordinate transformation to each row
        df["xmin"] = (df["x"]  * original_width).astype(int)
        df["ymin"] = (df["y"]  * original_height).astype(int)
        df["xmax"] = (df["xmin"] + df["width"] * original_width).astype(int)
        df["ymax"] = (df["ymin"] + df["height"] * original_height).astype(int)

        # Add the image_path column
        df["image_path"] = image_path
        
        # Append the dataframe to the list
        df = utilities.read_file(df, root_dir)
        dataframes.append(df)

# Concatenate all dataframes
annotations_df = pd.concat(dataframes, ignore_index=True)

# Add the 'label' and 'source' columns
annotations_df["label"] = "Tree"
annotations_df["source"] = "World Resources Institute"

# Save the concatenated dataframe to annotations.csv
annotations_df.to_csv(os.path.join(root_dir, "annotations.csv"), index=False)