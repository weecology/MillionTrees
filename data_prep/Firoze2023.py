import os
import json
import pandas as pd
import geopandas as gpd
from shapely import wkt
from preprocess_polygons import split_raster_with_polygons

ANNOTATION_CSV = "/orange/ewhite/DeepForest/Firoze2023/annotations.csv"
SAVE_DIR = "/orange/ewhite/DeepForest/Firoze2023/crops"
OUTPUT_CSV = "/orange/ewhite/DeepForest/Firoze2023/crops/annotations.csv"
PATCH_SIZE = 1500


def extract_bounding_boxes_from_labelme(folder_path):
    data = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            json_path = os.path.join(folder_path, filename)
            with open(json_path, 'r') as file:
                labelme_data = json.load(file)
                image_path = os.path.join(folder_path, labelme_data['imagePath'])

                polygons = []
                for shape in labelme_data['shapes']:
                    points = shape['points']
                    # Add the first to the end to close the polygon
                    points.append(points[0])
                    wkt_polygon = 'POLYGON(({}))'.format(', '.join(['{} {}'.format(p[0], p[1]) for p in points]))
                    polygons.append([image_path, wkt_polygon])

                # Create dataframe with image path
                df = pd.DataFrame(polygons, columns=['image_path', 'geometry'])
                data.append(df)

    annotations = pd.concat(data)

    return annotations


def tile_firoze():
    os.makedirs(SAVE_DIR, exist_ok=True)

    df = pd.read_csv(ANNOTATION_CSV)
    df["geometry"] = df["geometry"].apply(wkt.loads)
    df["geometry"] = df["geometry"].apply(lambda g: g.buffer(0) if not g.is_valid else g)
    gdf = gpd.GeoDataFrame(df, geometry="geometry")
    gdf["image_path"] = gdf["image_path"].apply(os.path.basename)

    all_annotations = []
    for image_name in gdf["image_path"].unique():
        image_gdf = gdf[gdf["image_path"] == image_name].copy()
        # Reconstruct full path from basename
        full_path = df.loc[gdf["image_path"] == image_name, "image_path"].iloc[0]

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
    combined["source"] = "Firoze et al. 2023"
    combined["label"] = "Tree"
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(combined)} annotations across {combined['image_path'].nunique()} tiles to {OUTPUT_CSV}")
    return combined


if __name__ == "__main__":
    tile_firoze()
