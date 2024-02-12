import glob
import os
import pandas as pd
from deepforest.utilities import read_file
from deepforest.preprocess import split_raster
import geopandas as gpd
import rasterio as rio
from rasterio.plot import show
from matplotlib import pyplot as plt
from shapely.geometry import box
import shutil

def Jansen_2023():
    shps = glob.glob("/blue/ewhite/DeepForest/Jansen_2023/images/*.shp")
    images = glob.glob("/blue/ewhite/DeepForest/Jansen_2023/images/*.tif")

    split_annotations = []
    for shp in shps:
        print(shp)
        gdf = gpd.read_file(shp)
        gdf = gdf[~(gdf.geometry.type=="MultiPolygon")]
        image = shp.replace("_Labels.shp", "_RGB.tif")  # Generate corresponding image path
        gdf["image_path"] = image
        gdf["label"] = "Tree"

        # Confirm overlap
        #src_bounds = rio.open(image).bounds
        #fig, ax = plt.subplots(figsize=(10, 10))
        #gpd.GeoSeries(box(*src_bounds)).plot(color="red", alpha=0.3, ax=ax)
        #gdf.plot(ax=ax, alpha=0.3)
        #plt.savefig("fig.png")

        annotations = read_file(input=gdf)

        split_annotations_1 = split_raster(
            annotations,
            path_to_raster=image,
            patch_size=2000,
            allow_empty=False, 
            base_dir="/blue/ewhite/DeepForest/Jansen_2023/pngs"
        )
        split_annotations.append(split_annotations_1)

    split_annotations = pd.concat(split_annotations)
    split_annotations = split_annotations[~(split_annotations.geometry.geom_type=="MultiPolygon")]
    
    # Add full path to images
    split_annotations["image_path"] = split_annotations.image_path.apply(lambda x: "/blue/ewhite/DeepForest/Jansen_2023/pngs/{}".format(x))

    # Split train test based on image path
    split_images = split_annotations.image_path.unique()
    train_images = split_images[0:int(len(split_images) * 0.8)]
    test_images = [x for x in split_images if x not in train_images]

    split_annotations["split"] = "train"
    split_annotations.loc[split_annotations.image_path.isin(test_images), "split"] = "test"
    split_annotations["source"] = "Jansen et al. 2023"
    split_annotations.to_csv("/blue/ewhite/DeepForest/Jansen_2023/pngs/annotations.csv")

if __name__ == "__main__":
    Jansen_2023()