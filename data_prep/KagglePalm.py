import os
import pandas as pd
import geopandas as gpd
from shapely import wkt
from preprocess_polygons import split_raster_with_polygons

ANNOTATION_CSV = "/orange/ewhite/DeepForest/KagglePalm/Palm-Counting-349images/annotations.csv"
SAVE_DIR = "/orange/ewhite/DeepForest/KagglePalm/Palm-Counting-349images/crops"
OUTPUT_CSV = "/orange/ewhite/DeepForest/KagglePalm/Palm-Counting-349images/crops/annotations.csv"
PATCH_SIZE = 1500


def tile_kaggle_palm():
    os.makedirs(SAVE_DIR, exist_ok=True)

    df = pd.read_csv(ANNOTATION_CSV)
    df["geometry"] = df["geometry"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry")
    # Keep the full path for opening, use basename for matching in split_raster_with_polygons
    gdf["image_path"] = gdf["image_path"].apply(os.path.basename)

    # Build a map from basename -> full path using original df
    basename_to_fullpath = dict(
        zip(df["image_path"].apply(os.path.basename), df["image_path"])
    )

    all_annotations = []
    for image_name in gdf["image_path"].unique():
        image_gdf = gdf[gdf["image_path"] == image_name].copy()
        full_path = basename_to_fullpath[image_name]

        try:
            tiled = split_raster_with_polygons(
                image_gdf,
                path_to_raster=full_path,
                patch_size=PATCH_SIZE,
                patch_overlap=0,
                allow_empty=False,
                save_dir=SAVE_DIR,
            )
            all_annotations.append(tiled)
        except Exception as e:
            print(f"Skipping {image_name}: {e}")

    if not all_annotations:
        raise RuntimeError("No annotations produced — check image paths and annotation file.")

    combined = pd.concat(all_annotations, ignore_index=True)
    combined["image_path"] = combined["image_path"].apply(
        lambda x: os.path.join(SAVE_DIR, os.path.basename(x))
    )
    combined["source"] = "Kaggle_Palm_Counting"
    combined["label"] = "Tree"
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(combined)} annotations across {combined['image_path'].nunique()} tiles to {OUTPUT_CSV}")
    return combined


if __name__ == "__main__":
    tile_kaggle_palm()
