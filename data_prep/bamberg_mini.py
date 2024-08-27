import pandas as pd
import os
import shutil
import zipfile

df = pd.read_csv("/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/annotations/annotations.csv")

# Extract tile names from the image filenames
df['tile_name'] = df['image_path'].apply(lambda x: '_'.join(os.path.basename(x).split('_')[:-1]))

# Create a new dataframe with one example of each tile
df_unique_tiles = df.groupby('tile_name').first().reset_index()

# Save the new dataframe to a CSV file
df_mini = df[df.image_path.isin(df_unique_tiles.image_path)]

# Copy the images of the unique tiles to a new directory
source_dir = "/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/images"
target_dir = "/orange/ewhite/DeepForest/Troles_Bamberg/Bamberg_mini/images"

os.makedirs(target_dir, exist_ok=True)
df_mini.to_csv("/orange/ewhite/DeepForest/Troles_Bamberg/Bamberg_mini/annotations.csv", index=False)

for filename, row in df_unique_tiles.groupby("image_path").first().iterrows():    
    source_path = os.path.join(source_dir, filename)
    target_path = os.path.join(target_dir, filename)
    try:
        shutil.copyfile(source_path, target_path)
    except shutil.SameFileError:
        continue

# Zip the data and images
zip_filename = "/orange/ewhite/DeepForest/Troles_Bamberg/Bamberg_mini/images.zip"

with zipfile.ZipFile(zip_filename, 'w') as zipf:
    zipf.write("/orange/ewhite/DeepForest/Troles_Bamberg/Bamberg_mini/annotations.csv", arcname="annotations.csv")
    for _, row in df_unique_tiles.iterrows():
        filename = row['image_path']
        image_path = os.path.join(target_dir, filename)
        zipf.write(image_path, arcname=filename)