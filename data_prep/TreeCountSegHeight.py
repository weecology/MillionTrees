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


def load_json_annotations(json_dir):
    """Load all .json annotation files and return a GeoDataFrame with polygons."""
    annotation_list = []
    json_files = glob.glob(os.path.join(json_dir, "annotation*.json"))
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
        image_name = os.path.basename(json_file).replace(".json", ".png")
        number = image_name.replace("annotation_", "").replace(".png", "")
        filename = read_rgb_channels(json_dir, number)
        print(filename)
        for poly in data.get("Trees", []):
            # Swap x and y for each coordinate pair
            swapped_poly = [(y, x) for x, y in poly]
            polygon = Polygon(swapped_poly)
            annotation_list.append({
            "image_path": filename,
            "geometry": polygon,
            "label": "tree"
            })
    
    gdf = gpd.GeoDataFrame(annotation_list, geometry="geometry")
    gdf["source"] = "Li et al. 2023"
    gdf["image_path"] = gdf["image_path"].apply(lambda x: os.path.join(json_dir, x))
    # Set to pandas dataframe with wkt geometry
    gdf["geometry"] = gdf["geometry"].to_wkt()
    df = pd.DataFrame(gdf)
    df.to_csv(os.path.join(json_dir, "annotations.csv"), index=False)

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

    # add source columns
    print(df.head())
    #plot_example(df, 0)


