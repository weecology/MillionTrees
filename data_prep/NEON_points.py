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
import shutil

points = gpd.read_file("/blue/ewhite/b.weinstein/DeepTreeAttention/fe902d874c4a41e4b8e5e0ddcfc9cb92/canopy_points.shp")

# Regenerate from scratch so stale crops (e.g. the old tight point-bounds tiles)
# don't survive the rerun.
cropped_dir = "/orange/ewhite/MillionTrees/NEON_points/cropped_tiles"
if os.path.exists(cropped_dir):
    shutil.rmtree(cropped_dir)
os.makedirs(cropped_dir, exist_ok=True)

viz_dir = "/orange/ewhite/MillionTrees/NEON_points/viz"
os.makedirs(viz_dir, exist_ok=True)

# Only the plot locations
plot_counts = points['plotID'].value_counts()
filtered_plots = plot_counts[plot_counts > 5].index
filtered_points = points[points['plotID'].isin(filtered_plots)]

# Standard NEON base plots are 40 x 40 m. Cropping to the tight point bounds
# collapses to the point cluster (often <20 m), producing tiny tiles, while the
# full sensor tile (1 km, 10000 px) is mostly unlabeled trees. Crop a fixed
# 40 m window centered on each plot instead: consistent size for standard plots,
# and expanded to the point span on larger contributed plots so no point is clipped.
PLOT_SIZE_M = 40.0

rgb_pool = glob.glob("/orange/ewhite/NeonData/*/DP3.30010.001/**/Camera/**/*.tif", recursive=True)

annotations = []
for plot_id in filtered_points['plotID'].unique():
    print(plot_id)
    plot_points = filtered_points[filtered_points['plotID'] == plot_id]
    point_bounds = plot_points.total_bounds

    # Build a window centered on the plot, at least PLOT_SIZE_M per axis but
    # never smaller than the point span (so larger plots stay fully covered).
    cx = (point_bounds[0] + point_bounds[2]) / 2
    cy = (point_bounds[1] + point_bounds[3]) / 2
    half_w = max(PLOT_SIZE_M, point_bounds[2] - point_bounds[0]) / 2
    half_h = max(PLOT_SIZE_M, point_bounds[3] - point_bounds[1]) / 2
    bounds = [cx - half_w, cy - half_h, cx + half_w, cy + half_h]

    # Search with a small 2 m buffer so edge plots still match a sensor tile,
    # but crop to the exact window bounds (the buffer must not leak into the
    # pixel-coordinate origin, or every point ends up shifted).
    search_bounds = list(bounds)
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

    # reset_index so read_file's internally reindexed geometry aligns back onto
    # these rows; a non-contiguous index leaves trailing geometries as None.
    plotdf = plot_points[["image_path", "x", "y"]].reset_index(drop=True)
    gdf = read_file(plotdf, root_dir=cropped_dir, label="Tree")
    gdf["source"] = "NEON_points"
    gdf["label"] = "Tree"
    gdf.root_dir = "/orange/ewhite/MillionTrees/NEON_points/cropped_tiles"
    gdf["score"] = 1

    # Save a handful of QA overlays so the new tile size can be eyeballed,
    # closing figures each iteration so they don't accumulate over all plots.
    if len(annotations) < 15:
        plot_results(gdf, savedir=viz_dir, basename=plot_id, show=False)
    plt.close("all")

    annotations.append(gdf)

pd.concat(annotations).to_csv("/orange/ewhite/MillionTrees/NEON_points/annotations.csv", index=False)
