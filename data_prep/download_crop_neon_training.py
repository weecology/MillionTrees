import os
from glob import glob
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
import neonutilities as nu

# Read neon_token.txt file to get the token
def read_neon_token():
    with open("data_prep/neon_token.txt", "r") as file:
        token = file.read().strip()
    return token


def download_remote_sensing_data(data_product, site, bounds, year):
    if data_product == "lidar":
        id = "DP3.30003.001"
    elif data_product == "rgb":
        id = "DP3.30010.001"
    elif data_product == "hyperspectral":
        id = "DP3.30006.002"
    elif data_product == "CHM":
        id = "DP3.30026.001"
    else:
        raise ValueError("Unsupported data product")
    
    xmin, ymin, xmax, ymax = bounds.values[0]
    print("Downloading data for site:", site, "from", xmin, ymin, "to", xmax, ymax, "for year:", year)
    nu.by_tile_aop(dpid=id, 
                site=site, 
                easting=int(xmin),
                northing=int(ymin),
                year=year,
                token=read_neon_token(),
                include_provisional=True,
                check_size=False,
                # Save in tmpdir, going to crop and save to final location later
                savepath=os.path.join("/Users/benweinstein/Downloads/", site),
                verbose=True,
                )
    
def crop_training_data_to_shp():
    # Define input and output folders
    shapefile_folder = "/Users/benweinstein/Dropbox/Weecology/NEON/neon_training_annotations"

    # Loop through shapefiles in the folder
    for year in range(2017, 2025):
        for root, dirs, files in os.walk(shapefile_folder):
            for file in files:
                if file.endswith(".shp"):
                    # Skip if filename doesn't start with a number
                    if not file[0].isdigit():
                        continue
                    shapefile_path = os.path.join(root, file)

                    # Check if the output shapefile already exists
                    output_shapefile_path = os.path.join(root, f"{year}_{os.path.splitext(file)[0]}.shp")
                    if os.path.exists(output_shapefile_path):
                        print(f"Output shapefile already exists: {output_shapefile_path}")
                        continue

                    gdf = gpd.read_file(shapefile_path)
                    
                    # Extract UTM coordinates from shapefile path
                    shapefile_parts = os.path.basename(shapefile_path).split("_")
                    geo_index = f"{shapefile_parts[-3]}_{shapefile_parts[-2]}"

                    # Download the tile from NEON API using geo_index
                    # (Assume geo_index is the tile name, and you want RGB data for the latest year)
                    site = shapefile_parts[1]
                    data_product = "rgb"
                    # Get bounds from shapefile geometry
                    bounds = gdf.total_bounds
                    bounds_df = pd.DataFrame([bounds], columns=["minx", "miny", "maxx", "maxy"])
                    # Download tile
                    download_remote_sensing_data(site=site, data_product=data_product, bounds=bounds_df, year=year)

                    # Find the downloaded tile (assuming naming convention)
                    tile_pattern = os.path.join("/Users/benweinstein/Downloads/", site, "**", f"*{geo_index}*.tif")
                    tile_files = glob(tile_pattern, recursive=True)

                    if len(tile_files) == 0:
                        print(f"No tile found for {geo_index}")
                        continue
                    else:
                        tile_path = tile_files[0]

                    # Crop the tile by the extent of the shapefile geometry
                    with rasterio.open(tile_path) as src:
                        # Get the bounding box of the shapefile geometry
                        minx, miny, maxx, maxy = gdf.total_bounds
                        # Compute the window to read
                        window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, src.transform)
                        out_image = src.read(window=window)
                        out_transform = src.window_transform(window)
                        out_meta = src.meta.copy()
                        out_meta.update({
                            "driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform
                        })
                    # Save the cropped tile
                    basename = os.path.splitext(file)[0]
                    # Replace the year in the basename with the current year
                    new_basename = str(year) + basename[4:]
                    cropped_tile_path = os.path.join(root, f"{new_basename}.tif")
                    with rasterio.open(cropped_tile_path, "w", **out_meta) as dest:
                        dest.write(out_image)

                    # save the shapefile for that year
                    shapefile_path = os.path.join(root, f"{new_basename}.shp")
                    gdf.to_file(shapefile_path)

                    # Delete the downloaded tile
                    os.remove(tile_path)

crop_training_data_to_shp()