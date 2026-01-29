import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import geopandas as gpd
from shapely.geometry import Polygon
from deepforest.utilities import read_file
from deepforest.preprocess import split_raster
import rasterio


def load_json_annotations(json_dir):
    """Load all .json annotation files, split images into chunks, and return cropped annotations."""
    crop_dir = os.path.join(json_dir, "crops")
    os.makedirs(crop_dir, exist_ok=True)
    
    json_files = glob.glob(os.path.join(json_dir, "annotation*.json"))
    all_crop_annotations = []
    
    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            image_name = os.path.basename(json_file).replace(".json", ".png")
            number = image_name.replace("annotation_", "").replace(".png", "")
            rgb_filename = read_rgb_channels(json_dir, number)
            
            # Create annotations for this image
            annotation_list = []
            for poly in data.get("Trees", []):
                # Swap x and y for each coordinate pair
                swapped_poly = [(y, x) for x, y in poly]
                polygon = Polygon(swapped_poly)
                annotation_list.append({
                    "image_path": os.path.basename(rgb_filename),
                    "geometry": polygon,
                    "label": "tree"
                })
            
            if not annotation_list:
                continue
            
            gdf = gpd.GeoDataFrame(annotation_list, geometry="geometry")
            gdf["source"] = "Li et al. 2023"
            
            # Standardize to expected annotation format
            ann = read_file(gdf, root_dir=json_dir)
            ann.root_dir = json_dir
            
            # Match CRS from raster to annotations (if available)
            try:
                with rasterio.open(rgb_filename) as src:
                    raster_crs = src.crs
                    if raster_crs is not None:
                        ann.crs = raster_crs
            except Exception:
                # PNG files may not have CRS, continue without it
                pass
            
            # Split raster into smaller chunks and get cropped annotations
            print(f"Splitting {rgb_filename} into smaller chunks...")
            ann.reset_index(drop=True, inplace=True)
            try:
                crop_annotations = split_raster(
                    ann,
                    path_to_raster=rgb_filename,
                    save_dir=crop_dir,
                    patch_size=1000,
                    patch_overlap=0,
                    allow_empty=False
                )
            except Exception as e:
                print(f"Error splitting {rgb_filename}: {e}")
                continue
            
            # Update image paths to full paths for cropped images
            crop_annotations["image_path"] = crop_annotations["image_path"].apply(
                lambda x: os.path.join(crop_dir, x)
            )
            
            all_crop_annotations.append(crop_annotations)
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    if not all_crop_annotations:
        return pd.DataFrame()
    
    df = pd.concat(all_crop_annotations, ignore_index=True)
    df.to_csv(os.path.join(crop_dir, "annotations.csv"), index=False)
    
    return df

def read_rgb_channels(image_dir, base_name):
    """Read red, green, blue channel pngs and stack into an RGB image using PIL."""
    r = Image.open(os.path.join(image_dir, f"red_{base_name}.png")).convert("L")
    g = Image.open(os.path.join(image_dir, f"green_{base_name}.png")).convert("L")
    b = Image.open(os.path.join(image_dir, f"blue_{base_name}.png")).convert("L")
    rgb = Image.merge("RGB", (r, g, b))
    filename = os.path.join(image_dir, f"rgb_{base_name}.png")
    rgb.save(filename)
    return filename

def plot_example(df, idx=0):
    """Plot an example image with polygon overlays using GeoPandas."""
    row = df.iloc[idx]
    img = np.array(Image.open(row["image_path"]))
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(img)
    # Plot all polygons for this image
    df[df["image_path"] == row["image_path"]].plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2)
    plt.title(row["image_path"])
    plt.axis("off")
    plt.show()
    fig.savefig("current.png")

if __name__ == "__main__":
    json_dir = "/orange/ewhite/DeepForest/TreeCountSegHeight/extracted_data_2aux_v4_cleaned_centroid_raw 2"
    df = load_json_annotations(json_dir)

    if not df.empty:
        print(df.head())
        plot_example(df, 0)
    else:
        print("No annotations generated.")


