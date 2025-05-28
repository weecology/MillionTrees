import os
import glob
import shutil
import zipfile
from pathlib import Path

# Base directory
base_dir = "/orange/ewhite/b.weinstein/NeonTreeEvaluation/hand_annotations"

# Get all .tif files with underscores (NEON files)
tif_files = glob.glob(os.path.join(base_dir, "**/*.tif"), recursive=True)
tif_files = [f for f in tif_files if "_" in os.path.basename(f)]

# Function to get corresponding shapefile components
def get_shapefile_components(tif_path):
    # Get base name without extension
    base_name = os.path.splitext(os.path.basename(tif_path))[0]
    dir_path = os.path.dirname(tif_path)
    
    # Find all shapefile components
    shape_extensions = [".shp", ".shx", ".dbf", ".prj", ".cpg"]
    shape_files = []
    
    for ext in shape_extensions:
        shape_path = os.path.join(dir_path, base_name + ext)
        if os.path.exists(shape_path):
            shape_files.append(shape_path)
    
    return shape_files

# Create a temporary directory for organizing files
temp_dir = os.path.join(os.getcwd(), "temp_neon_files")
os.makedirs(temp_dir, exist_ok=True)

# Process each tif file
for tif_file in tif_files:
    # Get corresponding shapefile components
    shape_files = get_shapefile_components(tif_file)
    
    # Create a subdirectory for this pair
    pair_name = os.path.splitext(os.path.basename(tif_file))[0]
    pair_dir = os.path.join(temp_dir, pair_name)
    os.makedirs(pair_dir, exist_ok=True)
    
    # Copy tif file
    shutil.copy2(tif_file, os.path.join(pair_dir, os.path.basename(tif_file)))
    
    # Copy shapefile components
    for shape_file in shape_files:
        shutil.copy2(shape_file, os.path.join(pair_dir, os.path.basename(shape_file)))

# Create zip file
zip_file = "neon_training_annotations.zip"
with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, temp_dir)
            zipf.write(file_path, arcname)

# Clean up temporary directory
shutil.rmtree(temp_dir)

# Print scp command
print("\nTo download the zip file, use this command:")
print(f"scp hpg:{os.path.join(os.getcwd(), zip_file)} .") 