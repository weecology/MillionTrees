# NEON Point Locations
import geopandas as gpd
import pandas as pd
import rasterio
from deepforest.utilities import crop_raster, read_file
from utilities import find_sensor_path
from deepforest.visualize import plot_results
from matplotlib import pyplot as plt

import glob
import os

points = gpd.read_file("/blue/ewhite/b.weinstein/DeepTreeAttention/fe902d874c4a41e4b8e5e0ddcfc9cb92/canopy_points.shp")

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

    # Search with a small 2 m buffer so edge plots still match a sensor tile,
    # but crop to the exact point bounds (the buffer must not leak into the
    # pixel-coordinate origin, or every point ends up shifted).
    search_bounds = bounds.copy()
    search_bounds[0] -= 2
    search_bounds[1] -= 2
    search_bounds[2] += 2
    search_bounds[3] += 2

    sensor_path = find_sensor_path(bounds=search_bounds, lookup_pool=rgb_pool)
    try:
        cropped_tile = crop_raster(rgb_path=sensor_path, bounds=bounds, savedir="/orange/ewhite/MillionTrees/NEON_points/cropped_tiles", filename=plot_id)
    except:
        print(f"Failed to crop {plot_id}")
        continue

    # Derive pixel coords from the written crop's geotransform so the origin and
    # north-up row direction always match the saved image exactly.
    with rasterio.open(cropped_tile) as ds:
        cols, rows = ~ds.transform * (plot_points.geometry.x.values, plot_points.geometry.y.values)
    plot_points['x'] = cols
    plot_points['y'] = rows

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
