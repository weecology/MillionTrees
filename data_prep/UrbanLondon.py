import geopandas as gpd
from deepforest.utilities import read_file
import rasterio as rio
import os
from shapely.geometry import box
import pandas as pd

train_crowns = gpd.read_file("/orange/ewhite/DeepForest/UrbanLondon/crowns/tiles_0.25m_160_20_0_train_crowns.shp")
test_crowns = gpd.read_file("/orange/ewhite/DeepForest/UrbanLondon/crowns/tiles_0.25m_160_20_0_test_crowns.shp")

source = "Zuniga-Gonzalez et al. 2023"

def get_image_annotations(tif_path, crowns):
    with rio.open(tif_path) as src:
        bounds = src.bounds
        bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        intersected = gpd.overlay(crowns, gpd.GeoDataFrame({'geometry': [bbox]}, crs=crowns.crs), how='intersection')
        clipped = gpd.clip(intersected, bbox)
        if clipped.empty:
            return None
        clipped["image_path"] = os.path.basename(tif_path)
        clipped['label'] = 'Tree'
        clipped = read_file(clipped, root_dir=os.path.dirname(tif_path))
        
        return clipped

tif_dir = "/orange/ewhite/DeepForest/UrbanLondon/rgb"
tif_files = [os.path.join(tif_dir, f) for f in os.listdir(tif_dir) if f.endswith('.tif')]

train_annotations = pd.concat([get_image_annotations(tif, train_crowns) for tif in tif_files], ignore_index=True)
test_annotations = pd.concat([get_image_annotations(tif, test_crowns) for tif in tif_files], ignore_index=True)
test_annotations["existing_split"] = "test"
train_annotations["existing_split"] = "train"

annotations = pd.concat([train_annotations, test_annotations], ignore_index=True)
annotations['source'] = source
annotations['image_path'] = tif_dir + "/" + annotations['image_path']
annotations["image_path"] = annotations["image_path"].str.replace("tif", "png")
annotations.to_csv("/orange/ewhite/DeepForest/UrbanLondon/annotations.csv", index=False)