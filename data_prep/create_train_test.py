from utilities import *
import random
import glob
from scipy.io import loadmat
import geopandas as gpd
import rasterio
from deepforest.utilities import shapefile_to_annotations, xml_to_annotations, crop_raster, geo_to_image_coordinates, read_file
from deepforest.preprocess import split_raster
from shapely.geometry import Point, box
import CHM
import os
import pandas as pd
import shutil



def justdiggit():
    with open("/blue/ewhite/DeepForest/justdiggit-drone/label_sample/Annotations_trees_only.json") as jsonfile:
        data = json.load(jsonfile)    
        ids = [x["id"] for x in data["images"]]
        image_paths = [x["file_name"] for x in data["images"]]
        id_df = pd.DataFrame({"id":ids,"image_path":image_paths})
        annotation_df = []
        for row in data["annotations"]:
            b = {"id":row["id"],"xmin":row["bbox"][0],"ymin":row["bbox"][1],"xmax":row["bbox"][2],"ymax":row["bbox"][3]}
            annotation_df.append(b)
    annotation_df = pd.DataFrame(annotation_df)
    annotations = annotation_df.merge(id_df)
    annotations["label"] = "Tree"
    annotations["source"] = "Justdiggit et al. 2023"

    print("There are {} annotations in {} images".format(annotations.shape[0], len(annotations.image_path.unique())))
    images = annotations.image_path.unique()
    random.shuffle(images)
    train_images = images[0:int(len(images)*0.8)]
    train = annotations[annotations.image_path.isin(train_images)]
    test = annotations[~(annotations.image_path.isin(train_images))]    

    train.to_csv("/blue/ewhite/DeepForest/justdiggit-drone/label_sample/train.csv")
    test.to_csv("/blue/ewhite/DeepForest/justdiggit-drone/label_sample/test.csv")

def ReForestTree():
    """This dataset used deepforest to generate predictions which were cleaned, no test data can be used"""
    annotations = pd.read_csv("/blue/ewhite/DeepForest/ReForestTree/mapping/final_dataset.csv")
    annotations["image_path"] = annotations["img_path"]
    print("There are {} annotations in {} images".format(annotations.shape[0], len(annotations.image_path.unique())))
    annotations.to_csv("/blue/ewhite/DeepForest/ReForestTree/images/train.csv")

def Hickman2021():
    rgb = "/blue/ewhite/DeepForest/Hickman2021/RCD105_MA14_21_orthomosaic_20141023_reprojected_full_res_crop1.tif"
    shp = "/blue/ewhite/DeepForest/Hickman2021/manual_crowns_sepilok.shp"
    
    annotations = shapefile_to_annotations(shp, rgb=rgb)
    split_annotations_1 = split_raster(
        annotations,
        path_to_raster="RCD105_MA14_21_orthomosaic_20141023_reprojected_full_res_crop1.tif",
        patch_size=2000,
        allow_empty=False,
        base_dir="/blue/ewhite/DeepForest/Hickman2021/images/")
    
    rgb = "/blue/ewhite/DeepForest/Hickman2021/RCD105_MA14_21_orthomosaic_20141023_reprojected_full_res_crop1.tif"
    annotations = shapefile_to_annotations(shp, rgb=rgb)
    split_annotations_1 = split_raster(
        annotations,
        path_to_raster="RCD105_MA14_21_orthomosaic_20141023_reprojected_full_res_crop1.tif",
        patch_size=2000,
        allow_empty=False,
        base_dir="/blue/ewhite/DeepForest/Hickman2021/images/")
    
def HemmingSchroeder():
    annotations = gpd.read_file("/blue/ewhite/DeepForest/HemmingSchroeder/data/training/trees_2017_training_filtered_labeled.shp")
    # There a many small plots, each with a handful of annotations
    plots = annotations.sampleid.unique()
    train_plots = annotations.sampleid.drop_duplicates().sample(frac=0.9)

    train = annotations[annotations.sampleid.isin(train_plots)]
    test = annotations[~(annotations.sampleid.isin(train_plots))]

    #Need to generate images from NEON data archive base on plot locations
    rgb_pool = glob.glob("/orange/ewhite/NeonData/*/DP3.30010.001/**/Camera/**/*.tif", recursive=True)
    plot_locations = gpd.read_file("/blue/ewhite/DeepForest/HemmingSchroeder/data/training/training_sample_sites.shp")
    plot_locations = plot_locations[plot_locations.sample_id.isin(annotations.sampleid)]
    
    year_annotations = []
    for plot_location in plot_locations.geometry:
            # Get rgb path for all years
            rgb_paths = find_sensor_path(bounds = plot_location.bounds, lookup_pool=rgb_pool, all_years=True)
            sampleid = plot_locations[plot_locations.geometry==plot_location]["sample_id"].values[0]
            sample_annotations = annotations[annotations.sampleid==sampleid]

            for rgb_path in rgb_paths:
                year = year_from_tile(rgb_path)
                basename = "{}_{}".format(sampleid, year)
                year_annotation = sample_annotations.copy()
                year_annotation["image_path"] = basename
                crop_raster(
                    bounds=plot_location.bounds,
                    sensor_path=rgb_path,
                    savedir="/blue/ewhite/DeepForest/HemmingSchroeder/data/training/images/",
                    basename=basename
                    )
                year_annotations.append(year_annotation)

    year_annotations = pd.concat(year_annotations)
    # Remove basenames with FullSite
    year_annotations = year_annotations[~(year_annotations.basename.str.contains("FullSite"))]
    train_annotations = year_annotations[year_annotations.sampleid.isin(train.sampleid)]
    test_annotations = year_annotations[year_annotations.sampleid.isin(test.sampleid)]

    train_annotations.to_csv("/blue/ewhite/DeepForest/HemmingSchroeder/train.csv")
    test_annotations.to_csv("/blue/ewhite/DeepForest/HemmingSchroeder/test.csv")

#def ForestGEO():
#   """Point data from within the NEON sites"""
#    OSBS = gpd.read_file("/orange/idtrees-collab/megaplot/OSBS_megaplot.shp")
#    SERC = gpd.read_file("/orange/idtrees-collab/megaplot/SERC_megaplot.shp")
#    HARV = gpd.read_file("/orange/idtrees-collab/megaplot/HARV_megaplot.shp")

#    #Split into iamges. 
#    shapefile_to_annotations()

def NEON_Trees():
    """Transform raw NEON data into clean shapefile   
    Args:
        config: DeepTreeAttention config dict, see config.yml
    """
    field = pd.read_csv("/blue/ewhite/DeepForest/NEON_Trees/vst_nov_2023.csv")
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
    plot_shp = gpd.read_file("/blue/ewhite/DeepForest/NEON_Trees/All_Neon_TOS_Polygon_V4.shp")
    annotations = []
    for plot in shp.plotID:
        subplot_annotations = shp[shp.plotID==plot]
        bounds = plot_shp[plot_shp.plotID==plot]
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
            crop_raster(
                bounds=bounds,
                sensor_path=sensor_path,
                savedir="/blue/ewhite/DeepForest/NEON_Trees/images/",
                basename=plot)
        except:
            continue
        annotations.append(subplot_annotations)
    
    #Split into train and test
    annotations = pd.concat(annotations)
    subplots = annotations.subID.unique()

    return shp

# Uncomment to regenerate each dataset
#Beloiu_2023()
#Siberia_polygons()
#justdiggit()
#ReForestTree()
#Treeformer()
#Ventura()
#Cloutier2023()
#HemmingSchroeder()
#ForestGEO()
#NEON_Trees()
#Ryoungseob_2023()
#Hickman2021()
