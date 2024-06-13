import os
import random
import shutil
from deepforest.preprocess import split_raster
import glob
import tempfile

# Get tmpdir
tmpdir = tempfile.gettempdir()

# Define the input and output directories
output_dir = '/blue/ewhite/b.weinstein/MillionTrees/images_to_annotate'

# Get a list of all image files 
#image_files = glob.glob("/orange/ewhite/NeonData/OSBS/DP3.30010.001/**/Camera/**/*.tif", recursive=True)

# Choose 100 random images
#random_images = random.sample(image_files, 100)
random_images = glob.glob("/orange/ewhite/DeepForest/unlabeled/airbornemap/*.tif")
#random_images = ["/orange/ewhite/DeepForest/Ulex_Chile_Kattenborn/data_ulex_europaeus/ulex_f4/ulex_f4_ortho.tif"]
# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each random image
for image_file in random_images:
    # Run through deepforest to split_raster
    # Replace this line with your own deepforest processing code
    processed_image = split_raster(annotations_file=None, path_to_raster=image_file, patch_size=2000, patch_overlap=0, base_dir=output_dir)

print("Images processed and saved successfully!")