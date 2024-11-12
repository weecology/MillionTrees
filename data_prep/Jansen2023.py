import glob
import pandas as pd
from deepforest.utilities import read_file
from deepforest.visualize import plot_results
from deepforest.preprocess import split_raster
import geopandas as gpd
import rasterio as rio
from rasterio.plot import show
from matplotlib import pyplot as plt
import os
from shapely.geometry import box

def Jansen_2023():
    shps = glob.glob("/orange/ewhite/DeepForest/Jansen_2023/images/*.shp")

    split_annotations = []
    for shp in shps:
        print(shp)
        gdf = gpd.read_file(shp)
        gdf = gdf[~(gdf.geometry.type=="MultiPolygon")]
        image = shp.replace("_Labels.shp", "_RGB.tif")  # Generate corresponding image path
        gdf["image_path"] = image
        gdf["label"] = "Tree"

        annotations = read_file(input=gdf)

        split_annotations_1 = split_raster(
            annotations,
            path_to_raster=image,
            patch_size=2000,
            allow_empty=False, 
            base_dir="/orange/ewhite/DeepForest/Jansen_2023/pngs"
        )
        split_annotations.append(split_annotations_1)

    split_annotations = pd.concat(split_annotations)
    split_annotations = split_annotations[~(split_annotations.geometry.geom_type=="MultiPolygon")]
    
    # view sample images
    split_annotations.root_dir = "/orange/ewhite/DeepForest/Jansen_2023/pngs"

    # Plot a sample image
    sample = split_annotations.sample(1)
    sample.root_dir = "/orange/ewhite/DeepForest/Jansen_2023/pngs"
    width, height = rio.open(os.path.join(sample.root_dir, sample.image_path.values[0])).shape
    plot_results(sample, height=height, width=width)
    plt.savefig("current.png")
    
    # Add full path to images
    split_annotations["image_path"] = split_annotations.image_path.apply(lambda x: "/orange/ewhite/DeepForest/Jansen_2023/pngs/{}".format(x))
    split_annotations["source"] = "Jansen et al. 2023"
    split_annotations.to_csv("/orange/ewhite/DeepForest/Jansen_2023/pngs/annotations.csv")

if __name__ == "__main__":
    Jansen_2023()