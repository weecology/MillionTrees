import os
import geopandas as gpd
from shapely.geometry import box
from deepforest.utilities import read_file
import pandas as pd
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling


# Define paths
geopackage_dir = "/orange/ewhite/DeepForest/Wagner_Australia/Crowns (manual)"
cropped_plots_dir = "/orange/ewhite/DeepForest/Wagner_Australia/Cropped plots"
output_csv = "/orange/ewhite/DeepForest/Wagner_Australia/annotations.csv"

# Padding in meters
padding = 10

# Initialize an empty list to store annotations
annotations = []

# Iterate over each geopackage in the directory
for filename in os.listdir(geopackage_dir):
    if filename.endswith(".gpkg"):
        basename = os.path.splitext(filename)[0]
        geopackage_path = os.path.join(geopackage_dir, filename)
        
        basename = basename.replace(".", "")
        
        # Look for the corresponding image in the cropped plots folder
        image_path = os.path.join(cropped_plots_dir, basename + ".tif")
        if not os.path.exists(image_path):
            print(f"Image for {basename} not found in cropped plots folder.")
            continue
        
        # Read the geopackage
        gdf = gpd.read_file(geopackage_path)
        
        # Get the total bounds of the polygons and pad by a few meters
        minx, miny, maxx, maxy = gdf.total_bounds
        minx -= padding
        miny -= padding
        maxx += padding
        maxy += padding
        
        # Read the image using rasterio
        with rio.open(image_path) as src:
            # Read the first three bands
            bands = src.read([1, 2, 3])
            
            # Define the output path for the three-band image
            three_band_path = os.path.join("/orange/ewhite/DeepForest/Wagner_Australia/three_band", basename + "_three_band.tif")
            
            # Save the three-band image

            # project into utm S55

            # Save the three-band image
            with rio.open(
                three_band_path,
                'w',
                driver='GTiff',
                height=src.height,
                width=src.width,
                count=3,
                dtype=bands.dtype,
                crs=src.crs,
                transform=src.transform,
            ) as dst:
                dst.write(bands[0], 1)
                dst.write(bands[1], 2)
                dst.write(bands[2], 3)
            
            # Reproject the three-band image to UTM S55
            with rio.open(three_band_path) as src:
                transform, width, height = calculate_default_transform(
                    src.crs, 'EPSG:32755', src.width, src.height, *src.bounds)
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': 'EPSG:32755',
                    'transform': transform,
                    'width': width,
                    'height': height
                })

                reprojected_path = os.path.join("/orange/ewhite/DeepForest/Wagner_Australia/three_band", basename + "_three_band_utm.tif")
                with rio.open(reprojected_path, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rio.band(src, i),
                            destination=rio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs='EPSG:32755',
                            resampling=Resampling.nearest)
                        
                png_path = os.path.join("/orange/ewhite/DeepForest/Wagner_Australia/three_band", basename + "_three_band_utm.png")        
                # Normalize the bands to 0-255
                bands = bands.astype('float32')
                for i in range(bands.shape[0]):
                    band_min, band_max = bands[i].min(), bands[i].max()
                    bands[i] = 255 * (bands[i] - band_min) / (band_max - band_min)
                bands = bands.astype('uint8')

                # Save the normalized image as PNG
                with rio.open(
                    png_path,
                    'w',
                    driver='PNG',
                    height=src.height,
                    width=src.width,
                    count=3,
                    dtype='uint8',
                    crs='EPSG:32755',
                    transform=transform,
                ) as dst:
                    dst.write(bands[0], 1)
                    dst.write(bands[1], 2)
                    dst.write(bands[2], 3)

        # Process the annotations
        gdf = gpd.read_file(geopackage_path)
        gdf["label"] = "Tree"
        
        # project into utm S55
        gdf = gdf.to_crs("EPSG:32755")
        gdf["image_path"] = reprojected_path
        annotations_df = read_file(gdf, root_dir=os.path.dirname(three_band_path))
        annotations_df['source'] = 'Wagner et al. 2023'
        
        # save a png without geospecific information
        annotations_df["image_path"] = png_path
 
        # Append to the list of annotations
        annotations.append(annotations_df)

# Concatenate all annotations into a single DataFrame
all_annotations = gpd.GeoDataFrame(pd.concat(annotations, ignore_index=True))
# Save to CSV
all_annotations.to_csv(output_csv, index=False)