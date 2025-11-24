import argparse
import yaml
import json
import pandas as pd
import glob
import geopandas as gpd
import os
import re
import rasterio
import math
import numpy as np
from deepforest.utilities import xml_to_annotations
import warnings
import shapely
import re

# Utilities Module
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
    
def determine_geometry_type(df, verbose=True):
    """Determine the geometry type of a geodataframe
    Args:
        df: a pandas dataframe
    Returns:
        geometry_type: a string of the geometry type
    """
    columns = df.columns
    if "geometry" in columns:
        geometry_type = df.geometry.type.unique()[0]
        if geometry_type == "Point":
            return 'point'
        if geometry_type == "Polygon":
            if (df.geometry.area == df.envelope.area).all():
                return 'box'
            else:
                return 'polygon'

    if "Polygon" in columns:
        raise ValueError("Polygon column is capitalized, please change to lowercase")
    
    if "xmin" in columns and "ymin" in columns and "xmax" in columns and "ymax" in columns:
        geometry_type = "box"
    elif "polygon" in columns:
        geometry_type = "polygon"
    elif "x" in columns and "y" in columns:
        geometry_type = 'point'
    else:
        raise ValueError("Could not determine geometry type from columns {}".format(columns))

    # Report number of annotations, unique images and geometry type
    if verbose:
        print("Found {} annotations in {} unique images with {} geometry type".format(
            df.shape[0], df.image_path.unique().shape[0], geometry_type))
    
    return geometry_type

def infer_existing_split_from_path_string(path: str):
    """
    Infer existing split from a path string.
    Any indication of validation/val/eval/holdout/test maps to 'test'.
    Train remains None.
    """
    if not isinstance(path, str):
        return None
    s = path.lower()
    # Split into tokens by common path separators
    tokens = re.split(r"[\\/]", s)
    # Simple checks for common folder/file indicators
    test_indicators = ["val", "valid", "validation", "eval", "holdout", "test"]
    if any(tok in test_indicators for tok in tokens):
        return "test"
    # Also check word boundaries in the whole string to catch e.g. *_val_*
    if re.search(r"\b(val|valid|validation|eval|holdout|test)\b", s):
        return "test"
    # If it explicitly says train, treat as not test (return None)
    if "train" in tokens or re.search(r"\btrain\b", s):
        return None
    return None

def tag_existing_split(df: pd.DataFrame, path_columns: list[str] | None = None) -> pd.DataFrame:
    """
    Add an 'existing split' column to df if any path-like columns are present.
    We mark rows as 'test' when path hints at validation/test-style folders.
    """
    columns_to_check = path_columns or ["image_path", "filename", "rgb", "image"]
    for col in columns_to_check:
        if col in df.columns:
            df = df.copy(deep=True)
            df["existing split"] = df[col].astype(str).apply(infer_existing_split_from_path_string)
            break
    return df

def read_file(input, rgb):
    """Read a file and return a geopandas dataframe. This is the main entry point for reading annotations into deepforest.
    Args:
        input: a path to a file or a pandas dataframe
        rgb: a path to an RGB image

    Returns:
        df: a geopandas dataframe with the properly formatted geometry column
    """
    # read file
    if isinstance(input, str):
        if input.endswith(".csv"):
            df = pd.read_csv(input)
        elif input.endswith(".shp") or input.endswith(".gpkg"):
            df = shapefile_to_annotations(input, rgb=rgb)
        elif input.endswith(".xml"):
            df = xml_to_annotations(input)
        else:
            raise ValueError("File type {} not supported. DeepForest currently supports .csv, .shp or .xml files. See https://deepforest.readthedocs.io/en/latest/annotation.html ".format(os.path.splitext(input)[1]))
    else:
        if type(input) == pd.DataFrame:
            df = input.copy(deep=True)
        elif type(input) == gpd.GeoDataFrame:
            return shapefile_to_annotations(input,rgb=rgb)
        else:
            raise ValueError("Input must be a path to a file, geopandas or a pandas dataframe")

    if type(df) == pd.DataFrame:
        # If the geometry column is present, convert to geodataframe directly
        if "geometry" in df.columns:
            df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
            df.crs = None
        else:
            # Detect geometry type
            geom_type = determine_geometry_type(df)

            # Check for uppercase names and set to lowercase
            df.columns = [x.lower() for x in df.columns]

            # convert to geodataframe
            if geom_type == "box":
                df['geometry'] = df.apply(
                        lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
            elif geom_type == "polygon":
                df['geometry'] = gpd.GeoSeries.from_wkt(df["polygon"])
            elif geom_type == "point":
                df["geometry"] = [shapely.geometry.Point(x, y)
                for x, y in zip(df.x.astype(float), df.y.astype(float))]
            else:
                raise ValueError("Geometry type {} not supported".format(geom_type))

    # convert to geodataframe
    df = gpd.GeoDataFrame(df, geometry='geometry')
    
    # remove any of the csv columns
    df = df.drop(columns=["polygon", "x", "y","xmin","ymin","xmax","ymax"], errors="ignore")
    
    # Try to infer existing data split (val/test -> 'test') from path-like columns
    # We intentionally keep this simple and fail fast.
    for col in ["image_path", "filename", "rgb", "image"]:
        if col in df.columns:
            df["existing split"] = df[col].astype(str).apply(infer_existing_split_from_path_string)
            break
                        
    return df

def geo_to_image_coordinates(gdf, image_bounds, image_resolution):
    """
    Convert from projected coordinates to image coordinates
    Args:
        gdf: a pandas type dataframe with columns: name, xmin, ymin, xmax, ymax. Name is the relative path to the root_dir arg.
        image_bounds: bounds of the image
        image_resolution: resolution of the image
    Returns:
        gdf: a geopandas dataframe with the transformed to image origin. CRS is removed
        """
    
    if len(image_bounds) != 4:
        raise ValueError("image_bounds must be a tuple of (left, bottom, right, top)")

    transformed_gdf = gdf.copy(deep=True)
    # unpack image bounds
    left, bottom, right, top = image_bounds
    
    transformed_gdf.geometry = transformed_gdf.geometry.translate(xoff=-left, yoff=-top)
    transformed_gdf.geometry = transformed_gdf.geometry.scale(xfact=1/image_resolution, yfact=-1/image_resolution, origin=(0,0))
    transformed_gdf.crs = None

    return transformed_gdf

def shapefile_to_annotations(shapefile,
                             rgb=None,
                             root_dir=None):
    """
    Convert a shapefile of annotations into annotations csv file for DeepForest training and evaluation
    
    Args:
        shapefile: Path to a shapefile on disk. If a label column is present, it will be used, else all labels are assumed to be "Tree"
        rgb: Path to the RGB image on disk
        root_dir: Optional directory to prepend to the image_path column        
    Returns:
        results: a pandas dataframe
    """
    # Read shapefile
    if isinstance(shapefile, str):
        gdf = gpd.read_file(shapefile)
    else:
        gdf = shapefile.copy(deep=True)

    if rgb is None:
        if "image_path" not in gdf.columns:
            raise ValueError("No image_path column found in shapefile, please specify rgb path")
        else:
            rgb = gdf.image_path.unique()[0]
            print("Found image_path column in shapefile, using {}".format(rgb))

    # Determine geometry type and report to user
    if gdf.geometry.type.unique().shape[0] > 1:
        raise ValueError(
            "Multiple geometry types found in shapefile. Please ensure all geometries are of the same type.")
    else:
        geometry_type = gdf.geometry.type.unique()[0]
        print("Geometry type of shapefile is {}".format(geometry_type))

    # raster bounds
    if root_dir:
        rgb = os.path.join(root_dir, rgb)
    with rasterio.open(rgb) as src:
        left, bottom, right, top = src.bounds
        resolution = src.res[0]
        raster_crs = src.crs

    # Check matching the crs
    if gdf.crs is not None:
        if raster_crs is None:
            raise ValueError("Shapefile has a crs, but raster does not. Please add a crs to the raster.")
        if not gdf.crs.to_string() == raster_crs.to_string():
             warnings.warn("The shapefile crs {} does not match the image crs {}".format(
                gdf.crs.to_string(), src.crs.to_string()), UserWarning)

    if src.crs is not None:
        print("CRS of shapefile is {}".format(src.crs))
        gdf = geo_to_image_coordinates(gdf, src.bounds, src.res[0])

    # check for label column
    if "label" not in gdf.columns:
        gdf["label"] = "Tree"
        warnings.warn("No label column found in shapefile. Please add a column named 'label' to your shapefile.", UserWarning)
    else:
        gdf["label"] = gdf["label"]
  
    # add filename
    gdf["image_path"] = os.path.basename(rgb)

    return gdf

def crop_raster(bounds, rgb_path=None, savedir=None, filename=None, driver="GTiff"):
    """
    Crop a raster to a bounding box, save as projected or unprojected crop
    Args:
        bounds: a tuple of (left, bottom, right, top) bounds
        rgb_path: path to the rgb image
        savedir: directory to save the crop
        filename: filename to save the crop "{}.tif".format(filename)"
        driver: rasterio driver to use, default to GTiff, can be 'GTiff' for projected data or 'PNG' unprojected data
    Returns:
        filename: path to the saved crop, if savedir specified
        img: a numpy array of the crop, if savedir not specified
    """
    left, bottom, right, top = bounds
    src = rasterio.open(rgb_path)
    if src.crs is None:
        # Read unprojected data using PIL and crop numpy array
        img = np.array(Image.open(rgb_path))
        img = img[bottom:top, left:right, :]
        img = np.rollaxis(img, 2, 0)
        cropped_transform = None
        if driver == "GTiff":
            warnings.warn(
                "Driver {} not supported for unprojected data, setting to 'PNG',".format(
                    driver), UserWarning)
            driver = "PNG"
    else:
        # Read projected data using rasterio and crop
        img = src.read(window=rasterio.windows.from_bounds(
            left, bottom, right, top, transform=src.transform))
        cropped_transform = rasterio.windows.transform(
            rasterio.windows.from_bounds(left,
                                         bottom,
                                         right,
                                         top,
                                         transform=src.transform), src.transform)
    if img.size == 0:
        raise ValueError("Bounds {} does not create a valid crop for source {}".format(
            bounds, src.transform))
    if savedir:
        res = src.res[0]
        height = (top - bottom) / res
        width = (right - left) / res

        # Write the cropped image to disk with transform
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        if driver == "GTiff":
            filename = "{}/{}.tif".format(savedir, filename)
            with rasterio.open(filename,
                               "w",
                               driver="GTiff",
                               height=height,
                               width=width,
                               count=img.shape[0],
                               dtype=img.dtype,
                               transform=cropped_transform) as dst:
                dst.write(img)
        elif driver == "PNG":
            # PNG driver does not support transform
            filename = "{}/{}.png".format(savedir, filename)
            with rasterio.open(filename,
                               "w",
                               driver="PNG",
                               height=height,
                               width=width,
                               count=img.shape[0],
                               dtype=img.dtype) as dst:
                dst.write(img)
        else:
            raise ValueError("Driver {} not supported".format(driver))

    if savedir:
        return filename
    else:
        return img
