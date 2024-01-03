# Utilities Module
import argparse
import yaml
import json
import xml.etree.ElementTree as ET
import pandas as pd
import glob
import geopandas as gpd
from deepforest.utilities import shapefile_to_annotations
import os
import re
import rasterio
import rasterstats
import math
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling


def read_config(config_path):
    """Read config yaml file"""
    #Allow command line to override 
    parser = argparse.ArgumentParser("DeepTreeAttention config")
    parser.add_argument('-d', '--my-dict', type=json.loads, default=None)
    args = parser.parse_known_args()
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    except Exception as e:
        raise FileNotFoundError("There is no config at {}, yields {}".format(
            config_path, e))
    
    #Update anything in argparse to have higher priority
    if args[0].my_dict:
        for key, value in args[0].my_dict:
            config[key] = value
        
    return config

def read_xml_Beloiu(path):
    tree = ET.parse(path)
    root = tree.getroot()

    # Initialize lists to store data
    filename_list = []
    name_list = []
    xmin_list = []
    ymin_list = []
    xmax_list = []
    ymax_list = []

    # Iterate through each 'object' element
    for obj in root.findall('.//object'):
        filename = root.find('.//filename').text
        name = obj.find('name').text
        xmin = float(obj.find('bndbox/xmin').text)
        ymin = float(obj.find('bndbox/ymin').text)
        xmax = float(obj.find('bndbox/xmax').text)
        ymax = float(obj.find('bndbox/ymax').text)
        
        # Append data to lists
        filename_list.append(filename)
        name_list.append(name)
        xmin_list.append(xmin)
        ymin_list.append(ymin)
        xmax_list.append(xmax)
        ymax_list.append(ymax)

    # Create a DataFrame
    data = {
        'image_path': filename_list,
        'name': name_list,
        'xmin': xmin_list,
        'ymin': ymin_list,
        'xmax': xmax_list,
        'ymax': ymax_list
    }

    df = pd.DataFrame(data)

    return df


def read_Siberia_points():
    shps = glob.glob("/blue/ewhite/DeepForest/Siberia/labels/*.shp")
    annotations = []
    for path in shps:
        ID = os.path.basename(path).split("_")[0]
        df = shapefile_to_annotations(
            path,
            rgb="/blue/ewhite/DeepForest/Siberia/orthos/{}_RGB_orthomosaic.tif".format(ID))
        annotations.append(df)
    annotations = pd.concat(annotations)

    annotations["source"] = "Kruse et al. 2021"
    
    return annotations

def bounds_to_geoindex(bounds):
    """Convert an extent into NEONs naming schema
    Args:
        bounds: list of top, left, bottom, right bounds, usually from geopandas.total_bounds
    Return:
        geoindex: str {easting}_{northing}
    """
    easting = int(np.mean([bounds[0], bounds[2]]))
    northing = int(np.mean([bounds[1], bounds[3]]))

    easting = math.floor(easting / 1000) * 1000
    northing = math.floor(northing / 1000) * 1000

    geoindex = "{}_{}".format(easting, northing)

    return geoindex

def find_sensor_path(lookup_pool, shapefile=None, bounds=None, geo_index=None, all_years=False):
    """Find a hyperspec path based on the shapefile using NEONs schema
    Args:
        bounds: Optional: list of top, left, bottom, right bounds, usually from geopandas.total_bounds. Instead of providing a shapefile
        lookup_pool: glob string to search for matching files for geoindex
    Returns:
        match: full path to sensor tile, if all years, a list of paths
    """
    if not geo_index:
        if shapefile:
            basename = os.path.splitext(os.path.basename(shapefile))[0]
            geo_index = re.search("(\d+_\d+)_image", basename).group(1)
        else:
            geo_index = bounds_to_geoindex(bounds=bounds) 
    
    match = [x for x in lookup_pool if geo_index in x]
    
    if len(match) == 0:
        raise ValueError("No matches for geoindex {} in sensor pool".format(geo_index))                    
        
    #Get most recent year or all years
    if all_years:
        # No duplicate years
        years = [year_from_tile(x) for x in match]
        year_df = pd.DataFrame({"year":years,"tiles":match})
        all_year_match = year_df.groupby("year").apply(lambda x: x.head(1)).tiles.values

        return all_year_match
    else:        
        match.sort()
        match = match[::-1]
        
        return match[0]
    
def year_from_tile(path):
    return path.split("/")[-8]
    