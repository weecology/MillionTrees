# Yosemite Point Dataset
# Download https://drive.google.com/drive/folders/1NWAqslICPoTS8OvT8zosI0R7cmsl6x9j

import os
import numpy as np
from PIL import Image
from deepforest import utilities
from deepforest.visualize import plot_annotations
from skimage.morphology import binary_opening, disk
from skimage.measure import label, regionprops
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

Image.MAX_IMAGE_PIXELS = 1262080000

def density_to_points(density_array, min_area=1, structure_size=2):
    """Convert density raster to point coordinates using morphological operations."""
    binary_mask = density_array > 0
    structure = disk(structure_size)
    #cleaned_mask = binary_opening(binary_mask, structure)
    labeled_array = label(binary_mask)
    regions = regionprops(labeled_array)
    
    points = np.array([
        [region.centroid[1], region.centroid[0]]  # (x, y) format
        for region in regions 
        if region.area >= min_area
    ])
    
    return points.astype(int) if len(points) > 0 else np.empty((0, 2))

def process_tiles(data_array, label_array, save_dir, patch_size=800, padding=4000):
    """Process image into tiles, skip empty ones, and save with annotations."""
    height, width = data_array.shape[:2]
    
    # Remove padding
    data_array = data_array[padding:height-padding, padding:width-padding]
    label_array = label_array[padding:height-padding, padding:width-padding]
    
    height, width = data_array.shape[:2]
    os.makedirs(save_dir, exist_ok=True)
    
    tile_count = 0
    saved_tiles = []
    
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            data_patch = data_array[y:y+patch_size, x:x+patch_size]
            label_patch = label_array[y:y+patch_size, x:x+patch_size]
            
            # Convert density to points
            points = density_to_points(label_patch)
            
            # Skip empty tiles
            if len(points) == 0:
                continue
            
            # Save image
            patch_filename = f'yosemite_tile_{tile_count}.png'
            patch_path = os.path.join(save_dir, patch_filename)
            Image.fromarray(data_patch).save(patch_path)
            
            # Save annotations
            gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(points[:, 0], points[:, 1]))
            gdf["image_path"] = patch_filename
            gdf["label"] = "Tree"
            gdf = utilities.read_file(gdf, root_dir=save_dir)
            gdf["source"] = "Chen & Shang (2022)"
            
            gdf.to_csv(patch_path.replace('.png', '.csv'))
            
            saved_tiles.append(gdf)
            tile_count += 1
    
    all_sites = pd.concat(saved_tiles, ignore_index=True)
    all_sites.to_csv(os.path.join(save_dir, 'yosemite_all_annotations.csv'))

    return saved_tiles

if __name__ == '__main__':
    data_path = "/orange/ewhite/DeepForest/Yosemite/"
    save_dir = "/orange/ewhite/DeepForest/Yosemite/tiles"
    
    # Load images
    data_image = Image.open(data_path + '/z20_data.png')
    label_image = Image.open(data_path + '/z20_label.png')
    
    data_array = np.array(data_image)
    label_array = np.array(label_image)
    
    print("Processing tiles...")
    saved_tiles = process_tiles(data_array, label_array, save_dir)
    
    print(f"Saved {len(saved_tiles)} non-empty tiles")