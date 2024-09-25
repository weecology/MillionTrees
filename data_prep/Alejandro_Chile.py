import os
import geopandas as gpd
from deepforest import utilities
import pandas as pd
import rasterio
from PIL import Image

# Directory containing the shapefiles and tif files
directory = "/orange/ewhite/DeepForest/Alejandro_Chile/alejandro"

# List to store all the GeoDataFrames
gdfs = []

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".shp"):
        # Load the shapefile
        shapefile_path = os.path.join(directory, filename)
        gdf = gpd.read_file(shapefile_path)
        
        # Extract the base name (without extension)
        base_name = os.path.splitext(filename)[0].lower()
        
        # Construct the corresponding tif file name
        tif_filename = f"mos_{base_name}.tif"
        tif_path = os.path.join(directory, tif_filename)
        
        # Check if the corresponding tif file exists
        if os.path.exists(tif_path):
            # Append the GeoDataFrame to the list
            gdf["image_path"] = os.path.basename(tif_path)
            # remove multi-polygons
            gdf = gdf[~gdf.geometry.type.isin(["MultiPolygon"])]
            gdf["label"] = "Tree"
            image_annotations = utilities.read_file(gdf, root_dir=directory)
            # Just keep image_path, geometry, label columns
            image_annotations = image_annotations[["image_path", "geometry", "label"]]
            gdfs.append(image_annotations)
        else:
            print(f"Corresponding tif file for {filename} not found.")

# Concatenate all GeoDataFrames
all_annotations = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

# Remove any annotations with negative coordinates
# Remove any annotations with negative coordinates
all_annotations = all_annotations[all_annotations.geometry.apply(lambda geom: geom.bounds[0] >= 0 and geom.bounds[1] >= 0)]

# Save the concatenated annotations to a CSV file
annotations_csv_path = os.path.join(directory, "annotations.csv")
# Ensure the output directory exists
output_directory = os.path.join(directory, "png_images")
os.makedirs(output_directory, exist_ok=True)

# Process each unique image path
unique_image_paths = all_annotations["image_path"].unique()
for image_path in unique_image_paths:
    # Open the tif file
    with rasterio.open(os.path.join(directory,image_path)) as src:
        # Read the data matrix
        data = src.read()
        png_path = os.path.join(output_directory, os.path.basename(image_path).replace(".tif", ".png"))
        # Normalize the bands to 0-255
        bands = data.astype('float32')
        for i in range(bands.shape[0]):
            band_min, band_max = bands[i].min(), bands[i].max()
            bands[i] = 255 * (bands[i] - band_min) / (band_max - band_min)
        bands = bands.astype('uint8')

        transform = src.transform

        # Save the normalized image as PNG
        with rasterio.open(
            png_path,
            'w',
            driver='PNG',
            height=src.height,
            width=src.width,
            count=3,
            dtype='uint8',
            transform=transform,
        ) as dst:
            dst.write(bands[0], 1)
            dst.write(bands[1], 2)
            dst.write(bands[2], 3)

# Replace .tif with .png for the image_paths
all_annotations["image_path"] = all_annotations["image_path"].str.replace(".tif", ".png")

# Make full image_path
all_annotations["image_path"] = output_directory + "/" + all_annotations["image_path"]
all_annotations["source"] = "Alejandro_Miranda"
all_annotations.to_csv("/orange/ewhite/DeepForest/Alejandro_Chile/alejandro/annotations.csv", index=False)