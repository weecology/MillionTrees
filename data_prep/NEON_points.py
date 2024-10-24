# NEON Point Locations
import geopandas as gpd
import pandas as pd
from deepforest.utilities import crop_raster, read_file
from utilities import find_sensor_path
from deepforest.visualize import plot_results
from matplotlib import pyplot as plt

import glob
import os

points = gpd.read_file("/orange/idtrees-collab/DeepTreeAttention/fe902d874c4a41e4b8e5e0ddcfc9cb92/canopy_points.shp")

os.makedirs("/orange/ewhite/MillionTrees/NEON_points/cropped_tiles", exist_ok=True)

# Only the plot locations
plot_counts = points['plotID'].value_counts()
filtered_plots = plot_counts[plot_counts > 5].index
filtered_points = points[points['plotID'].isin(filtered_plots)]

annotations = []
for plot_id in filtered_points['plotID'].unique():
    print(plot_id)
    plot_points = filtered_points[filtered_points['plotID'] == plot_id]
    rgb_pool = glob.glob("/orange/ewhite/NeonData/*/DP3.30010.001/**/Camera/**/*.tif", recursive=True)
    bounds = plot_points.total_bounds

    # add small buffer of 2 meters on all sides
    bounds[0] -= 2
    bounds[1] -= 2
    bounds[2] += 2
    bounds[3] += 2

    sensor_path = find_sensor_path(bounds=bounds, lookup_pool=rgb_pool)
    #if not os.path.exists(f"/orange/ewhite/MillionTrees/NEON_points/cropped_tiles/{plot_id}.tif"):
    try:
        cropped_tile = crop_raster(rgb_path=sensor_path, bounds=plot_points.total_bounds, savedir="/orange/ewhite/MillionTrees/NEON_points/cropped_tiles", filename=plot_id)
    except:
        print(f"Failed to crop {plot_id}")
        continue

    xmin, ymin, xmax, ymax = bounds
    plot_points['x'] = plot_points.geometry.x
    plot_points['y'] = plot_points.geometry.y
    plot_points['x'] -= xmin
    plot_points['y'] -= ymin

    # scale by resolution
    resolution = 0.1
    plot_points['x'] /= resolution
    plot_points['y'] /= resolution
    
    plot_points['image_path'] = cropped_tile

    plotdf = plot_points[["image_path", "x", "y"]]
    gdf = read_file(plotdf)
    gdf["source"] = "NEON_points"
    gdf["label"] = "Tree"
    gdf.root_dir = "/orange/ewhite/MillionTrees/NEON_points/cropped_tiles"
    gdf["score"] = 1
    plot_results(gdf)

    annotations.append(gdf)

pd.concat(annotations).to_csv("/orange/ewhite/MillionTrees/NEON_points/annotations.csv", index=False)
