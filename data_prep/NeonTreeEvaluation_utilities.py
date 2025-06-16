from deepforest.utilities import read_file
import glob
import os
import pandas as pd
import neonutilities as nu
import geopandas as gpd
from rasterio.windows import from_bounds
import numpy as np
from deepforest.utilities import read_file
import os

# Read neon_token.txt file to get the token
def read_neon_token():
    with open("data_prep/neon_token.txt", "r") as file:
        token = file.read().strip()
    return token
    
# Set the token for neonutilities
def lookup_plot_bounds(plot_file, plotID):

    # project to the utmZone of the plot
    plot_bounds = plot_file[plot_file["plotID"] == plotID].geometry
    if plot_bounds.empty:
        raise ValueError(f"Plot ID {plotID} not found in the plot file.")
    
    utm_zone = plot_file[plot_file["plotID"] == plotID].utmZone.values[0][:-1]
    plot_bounds = plot_bounds.to_crs(f"EPSG:326{utm_zone}")
    
    return plot_bounds.bounds

def download_remote_sensing_data(data_product, site, bounds, year):
    if data_product == "lidar":
        id = "DP1.30003.001"
    elif data_product == "rgb":
        id = "DP3.30010.001"
    elif data_product == "hyperspectral":
        id = "DP3.30006.002"
    elif data_product == "CHM":
        id = "DP3.30015.001"
    else:
        raise ValueError("Unsupported data product")
    
    xmin, ymin, xmax, ymax = bounds.values[0]
    print("Downloading data for site:", site, "from", xmin, ymin, "to", xmax, ymax, "for year:", year)
    
    geo_index = f"{int(xmin // 1000 * 1000)}_{int(ymin // 1000 * 1000)}"
    existing_tiles = glob.glob(os.path.join("/orange/ewhite/NeonData/", site, f"{id}/**/*"), recursive=True)
    existing_tiles = [x for x in existing_tiles if geo_index in x]
    existing_tiles = [x for x in existing_tiles if str(year) in x]
    if len(existing_tiles) > 0:
        print(f"Data for {site}, {geo_index}, {year} already exists, skipping download.")
        return existing_tiles[0]
    else:
        download_tiles = nu.by_tile_aop(dpid=id, 
                    site=site, 
                    easting=int(xmin),
                    northing=int(ymin),
                    year=year,
                    token=read_neon_token(),
                    include_provisional=True,
                    check_size=False,
                    # Save in tmpdir, going to crop and save to final location later
                    savepath=os.path.join("/orange/ewhite/NeonData/", site),
                    verbose=True,
                    )

def run(): 
    BENCHMARK_PATH = "/orange/idtrees-collab/NeonTreeEvaluation/"
    NEON_plot_file = "/home/b.weinstein/TreeSegmentation/data/NEONFieldSites/All_NEON_TOS_Plots_V7/All_NEON_TOS_Plot_Polygons_V7.shp"
    tifs = glob.glob(BENCHMARK_PATH + "evaluation/RGB/*.tif")
    xmls = [os.path.splitext(os.path.basename(x))[0] for x in tifs] 
    xmls = [os.path.join(BENCHMARK_PATH, "annotations", x) + ".xml" for x in xmls] 

    #Load and format xmls, not every RGB image has an annotation
    annotation_list = []   
    for xml_path in xmls:
        try:
            annotation = read_file(xml_path)
        except:
            continue
        annotation_list.append(annotation)
    benchmark_annotations = pd.concat(annotation_list, ignore_index=True)
    plot_file = gpd.read_file(NEON_plot_file)

    # For each plotID, crop the remote sensing data
    benchmark_annotations["plotID"] = benchmark_annotations["image_path"].apply(lambda x: "_".join(x.split("_")[0:2]))
    plotIDs = benchmark_annotations["plotID"].unique()
    data_products = ["rgb","CHM","lidar"] # ADD "hyperspectral" when needed
    for year in [2024, 2023, 2022, 2021, 2020, 2019, 2018]:
        for data_product in data_products:
            print(f"Processing data product: {data_product}")
            for plotID in plotIDs:
                site = plotID.split("_")[0]
                try:
                    bounds = lookup_plot_bounds(plot_file, plotID)
                except ValueError as e:
                    continue
                download_remote_sensing_data(site=site, data_product=data_product, bounds=bounds, year=year)

 

def match_xml_to_tif(xml_path, tifs):
     ## match xml to tif

    neon_training_annotations = "neon_training_annotations"
    annotation_folder = "NeonTreeEvaluation/annotations"

    for tif_folder in os.listdir(neon_training_annotations):
        tif_folder_path = os.path.join(neon_training_annotations, tif_folder)
        if not os.path.isdir(tif_folder_path):
            continue
        tif_files = glob.glob(os.path.join(tif_folder_path, "*.tif"))
        for tif_path in tif_files:
            tif_name = os.path.splitext(os.path.basename(tif_path))[0]
            xml_path = os.path.join(annotation_folder, f"{tif_name}.xml")
            if os.path.exists(xml_path):
                gdf = read_file(xml_path, rgb=tif_path)
                shapefile_path = os.path.join(tif_folder_path, f"{tif_name}.shp")
                gdf.to_file(shapefile_path)


run()