import glob
import geopandas as gpd
from deepforest.utilities import shapefile_to_annotations 
from deepforest.preprocess import split_raster
from shapely.geometry import Point
import os
import pandas as pd
import yaml
import argparse
import math
import numpy as np
import re
from utilities import crop_raster, geo_to_image_coordinates, read_file
import json

def NEON_Trees():
    """Transform raw NEON data into clean shapefile   
    Args:
        config: DeepTreeAttention config dict, see config.yml
    """
    field = pd.read_csv("/orange/ewhite/DeepForest/NEON_Trees/vst_nov_2023.csv")
    field["individual"] = field["individualID"]
    field = field[~field.growthForm.isin(["liana","small shrub"])]
    field = field[~field.growthForm.isnull()]
    field = field[~field.plantStatus.isnull()]        
    field = field[field.plantStatus.str.contains("Live")]  
    field.shape
    field = field[field.stemDiameter > 10]
    
    # Recover canopy position from any eventID
    groups = field.groupby("individual")
    shaded_ids = []
    for name, group in groups:
        shaded = any([x in ["Full shade", "Mostly shaded"] for x in group.canopyPosition.values])
        if shaded:
            if any([x in ["Open grown", "Full sun"] for x in group.canopyPosition.values]):
                continue
            else:
                shaded_ids.append(group.individual.unique()[0])
        
    field = field[~(field.individual.isin(shaded_ids))]
    field = field[(field.height > 3) | (field.height.isnull())]

    # Most Recent Year
    field = field.groupby("individual").apply(lambda x: x.sort_values(["eventID"],ascending=False).head(1)).reset_index(drop=True)
    field = field[~(field.eventID.str.contains("2014"))]
    field = field[~field.height.isnull()]

    # Remove multibole
    field = field[~(field.individual.str.contains('[A-Z]$',regex=True))]

    # List of hand cleaned errors
    known_errors = ["NEON.PLA.D03.OSBS.03422","NEON.PLA.D03.OSBS.03422","NEON.PLA.D03.OSBS.03382", "NEON.PLA.D17.TEAK.01883"]
    field = field[~(field.individual.isin(known_errors))]
    field = field[~(field.plotID == "SOAP_054")]

    #Create shapefile
    field["geometry"] = [Point(x,y) for x,y in zip(field["itcEasting"], field["itcNorthing"])]
    shp = gpd.GeoDataFrame(field)

    # CHM Filter
    CHM_pool = glob.glob("/orange/ewhite/NeonData/**/CanopyHeightModelGtif/*.tif",recursive=True)
    rgb_pool = glob.glob("/orange/ewhite/NeonData/**/DP3.30010.001/**/Camera/**/*.tif",recursive=True)

    #shp = CHM.filter_CHM(shp, CHM_pool)

    # BLAN has some data in 18N UTM, reproject to 17N update columns
    BLAN_errors = shp[(shp.siteID == "BLAN") & (shp.utmZone == "18N")]
    BLAN_errors.set_crs(epsg=32618, inplace=True)
    BLAN_errors.to_crs(32617,inplace=True)
    BLAN_errors["utmZone"] = "17N"
    BLAN_errors["itcEasting"] = BLAN_errors.geometry.apply(lambda x: x.coords[0][0])
    BLAN_errors["itcNorthing"] = BLAN_errors.geometry.apply(lambda x: x.coords[0][1])

    # reupdate
    shp.loc[BLAN_errors.index] = BLAN_errors

    # Oak Right Lab has no AOP data
    shp = shp[~(shp.siteID.isin(["PUUM","ORNL"]))]

    # Create a unique subplot ID
    shp = shp[~(shp["subplotID"].isnull())]
    shp["subID"] = shp["plotID"] + "_" + shp["subplotID"].astype(int).astype("str") 

    # For each subplot crop image and gather annotations
    plot_shp = gpd.read_file("/orange/ewhite/DeepForest/NEON_Trees/All_Neon_TOS_Polygon_V4.shp")
    annotations = []
    for plot in shp.plotID:
        subplot_annotations = shp[shp.plotID==plot]
        bounds = plot_shp[plot_shp.plotID==plot]
        if bounds.empty:
            continue
        utmZone = bounds.utmZone.unique()[0] 
        if utmZone == "6N":
            epsg = 32606
        elif utmZone=="5N":
            epsg=32605
        else:
            epsg = "326{}".format(int(utmZone[:2]))
        bounds = bounds.to_crs(epsg).total_bounds
        try:
            sensor_path = find_sensor_path(bounds=list(bounds), lookup_pool=rgb_pool)
            year = year_from_tile(sensor_path)
            crop_raster(
                bounds=bounds,
                rgb_path=sensor_path,
                savedir="/orange/ewhite/DeepForest/NEON_Trees/images/",
                filename="{}_{}".format(plot, year))
        except:
            continue
        annotations.append(subplot_annotations)
    
    #Split into train and test
    annotations = pd.concat(annotations)
    subplots = annotations.subID.unique()

    return shp

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
    
def main():
    config = read_config("/home/b.weinstein/DeepTreeAttention/config.yml")
    shp = NEON_Trees()
    shp.to_file("NEON_Trees.shp")

if __name__ == "__main__":
    main()