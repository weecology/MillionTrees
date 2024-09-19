import geopandas as gpd
import glob
import pandas as pd
from utilities import *
from deepforest.utilities import crop_raster, geo_to_image_coordinates
import rasterio as rio

def HemmingSchroeder():
    """Polygon annotations over TEAK NEON sites"""
    
    # Read the polygon annotations from the shapefile
    annotations = gpd.read_file("/orange/ewhite/DeepForest/HemmingSchroeder/data/training/trees_2017_training_filtered_labeled.shp")
    
    # Find RGB images for each plot location
    rgb_pool = glob.glob("/orange/ewhite/NeonData/*/DP3.30010.001/**/Camera/**/*.tif", recursive=True)
    plot_locations = gpd.read_file("/orange/ewhite/DeepForest/HemmingSchroeder/data/training/training_sample_sites.shp")
    plot_locations = plot_locations[plot_locations.sample_id.isin(annotations.sampleid)]
    
    # Generate images and annotations for each plot location
    year_annotations = []
    for plot_location in plot_locations.geometry:
        # Get RGB paths for all years
        rgb_paths = find_sensor_path(bounds=plot_location.bounds, lookup_pool=rgb_pool, all_years=True)
        sampleid = plot_locations[plot_locations.geometry==plot_location]["sample_id"].values[0]
        sample_annotations = annotations[annotations.sampleid==sampleid]
        
        for rgb_path in rgb_paths:
            year = year_from_tile(rgb_path)
            basename = "{}_{}".format(sampleid, year)
            year_annotation = sample_annotations.copy()
            year_annotation["image_path"] = "{}.tif".format(basename)
            
            left, bottom, right, top = plot_location.bounds
            left = left - 5
            bottom = bottom - 5
            right = right + 5 
            top = top + 5

            # Crop the raster based on plot location
            raster_path = crop_raster(
                bounds=(left, bottom, right, top),
                rgb_path=rgb_path,
                savedir="/blue/ewhite/DeepForest/HemmingSchroeder/data/training/images/",
                filename=basename
            )
            
            # Open the cropped raster
            src = rio.open(raster_path)
            
            # Convert the annotation coordinates from geographic to image coordinates
            image_coordinates = geo_to_image_coordinates(year_annotation, src.bounds, src.res[0])
            year_annotations.append(image_coordinates)

    year_annotations = pd.concat(year_annotations)
    
    # Remove basenames with FullSite
    year_annotations = year_annotations[~(year_annotations.image_path.str.contains("FullSite"))]


    # Concatenate the training and test annotations and save to a CSV file
    year_annotations["image_path"] = "images/" + year_annotations["image_path"]
    year_annotations.to_csv("/orange/ewhite/DeepForest/HemmingSchroeder/annotations.csv")

if __name__ == "__main__":
    HemmingSchroeder()