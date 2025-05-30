import os
import glob
import pandas as pd
import geopandas as gpd
from deepforest.preprocess import split_raster, read_file
import rasterio as rio
from deepforest.visualize import plot_results

def process_bohlman_bci(geometry):
    # Directories
    raster_dir = "/orange/ewhite/DeepForest/BohlmanBCI"
    output_dir = os.path.join(raster_dir, "crops")
    os.makedirs(output_dir, exist_ok=True)

    # Find all .tif files
    tif_files = glob.glob(os.path.join(raster_dir, "*.tif"))

    all_annotations = []

    for tif_path in tif_files:
        basename = os.path.splitext(os.path.basename(tif_path))[0]
        # Assume shapefile with same basename
        shp_path = os.path.join(raster_dir, f"{basename}_{geometry}.shp")
        
        if not os.path.exists(shp_path):
            print(f"Skipping {tif_path}, no matching shapefile found.")
            continue

        print(f"Processing {tif_path} and {shp_path}")

        with rio.open(tif_path) as src:
            # Set the CRS to EPSG:26717 and save as _projected.tif
            projected_path = os.path.join(raster_dir, f"{basename}_projected.tif")
            profile = src.profile.copy()
            profile.update(crs="EPSG:26717")
            with rio.open(projected_path, "w", **profile) as dst:
                for i in range(1, src.count + 1):
                    dst.write(src.read(i), i)
            tif_path = projected_path
        
        gdf = gpd.read_file(shp_path)
        
        if geometry == 'crowns':
            gdf = gdf[gdf.geometry.type == "Polygon"].copy()

        gdf["image_path"] = os.path.basename(projected_path)
        gdf["label"] = "tree"

        # Read as MillionTrees annotation
        # remove crs
        gdf = gdf.set_crs(epsg=26717, allow_override=True)

        annotation = read_file(gdf, root_dir=raster_dir)

        # Split raster into patches
        crop_annotations = split_raster(
            annotation,
            path_to_raster=tif_path,
            patch_size=1500,
            allow_empty=False,
            base_dir=output_dir
        )
        all_annotations.append(crop_annotations)


    # Combine and finalize
    annotations = pd.concat(all_annotations)
    annotations["image_path"] = annotations["image_path"].apply(lambda x: os.path.join(output_dir, x))
    annotations["source"] = "Bohlman et al. 2008"

    # Save
    output_csv = os.path.join(output_dir, f"annotations_{geometry}.csv")
    annotations.to_csv(output_csv, index=False)
    print(f"Annotations saved to {output_csv}")

if __name__ == "__main__":
    process_bohlman_bci(geometry="crowns")
    process_bohlman_bci(geometry="points")

    df = read_file("/orange/ewhite/DeepForest/BohlmanBCI/crops/annotations_points.csv")
    # Get the first image
    df = df[df["image_path"] == df["image_path"].iloc[0]]
    df.root_dir = "/orange/ewhite/DeepForest/BohlmanBCI/crops"
    plot_results(df)