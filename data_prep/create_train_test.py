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

def Beloiu_2023():
    xmls = glob.glob("/blue/ewhite/DeepForest/Beloiu_2023/labels/*")
    annotations = []
    for path in xmls:
        df = read_xml_Beloiu(path)
        gdf = read_file(df)
        annotations.append(gdf)

    annotations = pd.concat(annotations)
    annotations["label"] = "Tree"
    annotations["source"] = "Beloiu et al. 2023"
    print("There are {} annotations in {} images".format(annotations.shape[0], len(annotations.image_path.unique())))
    images = annotations.image_path.unique()

    # Convert projected tiff to png
    for image in images:
        with rasterio.open(os.path.join("/blue/ewhite/DeepForest/Beloiu_2023/images",image)) as src:
            bounds = src.bounds
            res = src.res[0]

        # read tif and save as png
        filename = crop_raster(
            bounds=bounds,
            rgb_path=os.path.join("/blue/ewhite/DeepForest/Beloiu_2023/images",image),
            savedir="/blue/ewhite/DeepForest/Beloiu_2023/pngs/",
            filename=os.path.splitext(os.path.basename(image))[0],
            driver="PNG"
            )

    #Set image path to png
    annotations["image_path"] = annotations.image_path.apply(lambda x: os.path.splitext(os.path.basename(x))[0] + ".png")
    random.shuffle(images)
    train_images = images[0:int(len(images)*0.9)]
    train = annotations[annotations.image_path.isin(train_images)]
    test = annotations[~(annotations.image_path.isin(train_images))]
    
    train.to_csv("/blue/ewhite/DeepForest/Beloiu_2023/images/train.csv")
    test.to_csv("/blue/ewhite/DeepForest/Beloiu_2023/images/test.csv")

    # Move all data to the common images dir
    for image_path in annotations.image_path.unique():
        src = os.path.join("/blue/ewhite/DeepForest/Beloiu_2023/pngs/", image_path)
        dst = os.path.join("/blue/ewhite/DeepForest/MillionTrees/images/", image_path)
        shutil.copy(src, dst)
    
    train.to_csv("/blue/ewhite/DeepForest/MillionTrees/annotations/Beloiu_2023_train.csv")
    test.to_csv("/blue/ewhite/DeepForest/MillionTrees/annotations/Beloiu_2023_test.csv")

def Siberia_polygons():
    shps = glob.glob("/blue/ewhite/DeepForest/Siberia/vanGeffen-etal_2021b_shapefiles_allfiles/vanGeffen_et_al_SiDroForest_Individual_Polygon_Labelled/*.shp")
    annotations = []
    cropped_images = []
    # There were several .tif files that did not have the correct crs compared to the .shp, read them in and covert them to the correct crs
    for path in shps:
        print(path)
        ID = os.path.basename(path).split("_")[0]
        rgb_path = "/blue/ewhite/DeepForest/Siberia/orthos/{}_RGB_orthomosaic.tif".format(ID)
        df = gpd.read_file(path)
        df["image_path"] = rgb_path
        df["label"] = "Tree"
        df = read_file(input=df)
        src = rasterio.open(rgb_path)

        if src.count == 4:
            # Remove alpha channel
            new_rgb_path = "/blue/ewhite/DeepForest/Siberia/orthos/{}_RGB_orthomosaic_corrected.tif".format(ID)
            with rasterio.open(rgb_path) as src:
                kwargs = src.meta.copy()
                kwargs.update(count=3)
                with rasterio.open(new_rgb_path, 'w', **kwargs) as dst:
                    dst.write(src.read()[:3,:,:])
        elif src.crs != df.crs:
            dst_crs = df.crs
            new_rgb_path = "/blue/ewhite/DeepForest/Siberia/orthos/{}_RGB_orthomosaic_corrected.tif".format(ID)

            with rasterio.open(rgb_path) as src:
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds)
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': dst_crs,
                    'transform': transform,
                    'width': width,
                    'height': height
                })

                with rasterio.open(new_rgb_path, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.nearest)
        else:
            new_rgb_path = rgb_path
            
        
        df = df[df.Type=="Tree"]
        if df.empty:
            continue
        df["source"] = "Kruse et al. 2021"
        df["original_image"] = new_rgb_path

        # Crop to bounded area with a small buffer
        buffered_bounds = box(*df.total_bounds).bounds

        # Translate to crop coordinates, destroys the CRS
        df.geometry = df.geometry.translate(xoff=-buffered_bounds[0], yoff=-buffered_bounds[1])
        filename = cropped_raster = crop_raster(
            bounds=buffered_bounds,
            rgb_path=new_rgb_path,
            savedir="/blue/ewhite/DeepForest/Siberia/images/",
            filename=ID,
            driver="PNG"
            )
        df["image_path"] = os.path.basename(filename)
        cropped_images.append(cropped_raster)
        df.crs = None
        
        annotations.append(df)

    annotations = pd.concat(annotations)
    images = annotations.image_path.unique()
    train_images = images[0:int(len(images)*0.8)]
    test_images = [x for x in images if x not in train_images]
    split_train_annotations = annotations[annotations.image_path.isin(train_images)]
    split_test_annotations = annotations[~(annotations.image_path.isin(train_images))]

    split_train_annotations.to_csv("/blue/ewhite/DeepForest/Siberia/images/train.csv")
    split_test_annotations.to_csv("/blue/ewhite/DeepForest/Siberia/images/test.csv")

    # Move all data to the common images dir
    for image_path in split_test_annotations.image_path.unique():
        src = os.path.join("/blue/ewhite/DeepForest/Siberia/images/", image_path)
        dst = os.path.join("/blue/ewhite/DeepForest/MillionTrees/images/", image_path)
        shutil.copy(src, dst)
    
    for image_path in split_train_annotations.image_path.unique():
        src = os.path.join("/blue/ewhite/DeepForest/Siberia/images/", image_path)
        dst = os.path.join("/blue/ewhite/DeepForest/MillionTrees/images/", image_path)
        shutil.copy(src, dst)

    split_train_annotations.to_csv("/blue/ewhite/DeepForest/MillionTrees/annotations/Siberia_train.csv")
    split_test_annotations.to_csv("/blue/ewhite/DeepForest/MillionTrees/annotations/Siberia_test.csv")

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

def Treeformer():
    def convert_mat(path):
        f = loadmat(x)
        points = f["image_info"][0][0][0][0][0]
        df = pd.DataFrame(points,columns=["x","y"])
        df["label"] = "Tree"
        df["source"] = "Amirkolaee et al. 2023"
        image_path = "_".join(os.path.splitext(os.path.basename(x))[0].split("_")[1:])
        image_path = "{}.jpg".format(image_path)
        image_dir = os.path.dirname(os.path.dirname(x))
        df["image_path"] = image_path
        return df

    test_gt = glob.glob("/blue/ewhite/DeepForest/TreeFormer/test_data/ground_truth/*.mat")
    test_ground_truth = []
    for x in test_gt:
        df = convert_mat(x)
        test_ground_truth.append(df)
    test_ground_truth = pd.concat(test_ground_truth)

    train_gt = glob.glob("/blue/ewhite/DeepForest/TreeFormer/train_data/ground_truth/*.mat")
    train_ground_truth = []
    for x in train_gt:
        df = convert_mat(x)
        train_ground_truth.append(df)
    train_ground_truth = pd.concat(train_ground_truth)
    
    val_gt = glob.glob("/blue/ewhite/DeepForest/TreeFormer/valid_data/ground_truth/*.mat")
    val_ground_truth = []
    for x in val_gt:
        df = convert_mat(x)
        val_ground_truth.append(df)
    val_ground_truth = pd.concat(val_ground_truth)

    test_ground_truth.to_csv("/blue/ewhite/DeepForest/TreeFormer/all_images/test.csv")
    train_ground_truth.to_csv("/blue/ewhite/DeepForest/TreeFormer/all_images/train.csv")
    val_ground_truth.to_csv("/blue/ewhite/DeepForest/TreeFormer/all_images/validation.csv")

    # Copy to MillionTrees folder
    for image_path in test_ground_truth.image_path.unique():
        src = os.path.join("/blue/ewhite/DeepForest/TreeFormer/test_data/images/", image_path)
        dst = os.path.join("/blue/ewhite/DeepForest/MillionTrees/images/", image_path)
        shutil.copy(src, dst)
    
    for image_path in train_ground_truth.image_path.unique():
        src = os.path.join("/blue/ewhite/DeepForest/TreeFormer/train_data/images/", image_path)
        dst = os.path.join("/blue/ewhite/DeepForest/MillionTrees/images/", image_path)
        shutil.copy(src, dst)
    
    for image_path in val_ground_truth.image_path.unique():
        src = os.path.join("/blue/ewhite/DeepForest/TreeFormer/valid_data/images/", image_path)
        dst = os.path.join("/blue/ewhite/DeepForest/MillionTrees/images/", image_path)
        shutil.copy(src, dst)
    
    test_ground_truth.to_csv("/blue/ewhite/DeepForest/MillionTrees/annotations/TreeFormer_test.csv")
    train_ground_truth.to_csv("/blue/ewhite/DeepForest/MillionTrees/annotations/TreeFormer_train.csv")
    val_ground_truth.to_csv("/blue/ewhite/DeepForest/MillionTrees/annotations/TreeFormer_validation.csv")

def Ventura():
    """In the current conception, using all Ventura data and not comparing against the train-test split"""
    all_csvs = glob.glob("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/csv/*.csv")
    train_images = pd.read_csv("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/train.txt",header=None,sep=" ")[0].values
    test_images = pd.read_csv("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/test.txt",header=None,sep=" ")[0].values
    val_images = pd.read_csv("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/val.txt",header=None,sep=" ")[0].values

    df = []
    for x in all_csvs:
        points = pd.read_csv(x)
        points["image_path"] = os.path.splitext(os.path.basename(x))[0]
        df.append(points)
    annotations = pd.concat(df)
    annotations["label"] = "Tree"

    train = annotations[annotations.image_path.isin(train_images)]
    test = annotations[annotations.image_path.isin(test_images)]
    val = annotations[annotations.image_path.isin(val_images)]
              
    train.to_csv("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/train.csv")
    test.to_csv("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/test.csv")
    val.to_csv("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/val.csv")

def Cloutier2023():
    # Zone 3 is test, Zone 1 and 2 is train. Intentionally vary window size.

    drone_flights = glob.glob("/blue/ewhite/DeepForest/Cloutier2023/**/*.tif",recursive=True)
    zone1 = "/blue/ewhite/DeepForest/Cloutier2023/quebec_trees_dataset_2021-06-17/Z1_polygons.gpkg"
    zone2 = "/blue/ewhite/DeepForest/Cloutier2023/quebec_trees_dataset_2021-06-17/Z2_polygons.gpkg"
    zone3 = "/blue/ewhite/DeepForest/Cloutier2023/quebec_trees_dataset_2021-06-17/Z3_polygons.gpkg"
    
    train = []
    test = []
    for flight in drone_flights:
            train_zone1 = shapefile_to_annotations(zone1, rgb=flight)
            split_annotations_1 = split_raster(train_zone1, path_to_raster=flight, patch_size=1000, allow_empty=True, base_dir="/blue/ewhite/DeepForest/Cloutier2023/images/")
            train_zone2 = shapefile_to_annotations(zone2, rgb=flight)
            split_annotations_2 = split_raster(train_zone2, path_to_raster=flight, patch_size=2000, allow_empty=True, base_dir="/blue/ewhite/DeepForest/Cloutier2023/images/")
            test_zone3 = shapefile_to_annotations(zone3, rgb=flight)
            split_annotations_3 = split_raster(test_zone3, path_to_raster=flight, patch_size=1500, allow_empty=True, base_dir="/blue/ewhite/DeepForest/Cloutier2023/images/")
            train.append(split_annotations_1)
            train.append(split_annotations_2)
            test.append(split_annotations_3)
    train = pd.concat(train)
    test = pd.concat(test)

    test.to_csv("/blue/ewhite/DeepForest/Cloutier2023/images/test.csv")
    train.to_csv("/blue/ewhite/DeepForest/Cloutier2023/images/train.csv")

def Jansen_2023():
    shps = glob.glob("/blue/ewhite/DeepForest/Jansen_2023/images/*.shp")
    images = "/blue/ewhite/DeepForest/Jansen_2023/*.tif"

    for image in images:
        basename = os.path.splitext(os.path.basename(image))[0]
        matching_shp = [x for x in shps if basename in shps]
        annotations = shapefile_to_annotations(matching_shp, rgb=image)
        split_annotations_1 = split_raster(
            annotations,
            path_to_raster=image,
            patch_size=1000,
            allow_empty=False,
            base_dir="/blue/ewhite/DeepForest/Hickman2021/images/")
        
        split_annotations_1.to_csv("/blue/ewhite/DeepForest/Hickman2021/images/train.csv")
        

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
   
def generate_NEON_benchmark():
    BENCHMARK_PATH = "/orange/idtrees-collab/NeonTreeEvaluation/"
    tifs = glob.glob(BENCHMARK_PATH + "evaluation/RGB/*.tif")
    xmls = [os.path.splitext(os.path.basename(x))[0] for x in tifs] 
    xmls = [os.path.join(BENCHMARK_PATH, "annotations", x) + ".xml" for x in xmls] 
    
    #Load and format xmls, not every RGB image has an annotation
    annotation_list = []   
    for xml_path in xmls:
        try:
            annotation = xml_to_annotations(xml_path)
        except:
            continue
        annotation_list.append(annotation)
    benchmark_annotations = pd.concat(annotation_list, ignore_index=True)

    benchmark_annotations["source"] = "NEON_benchmark"
    for image_path in benchmark_annotations.image_path.unique():
        dst = os.path.join(BENCHMARK_PATH, "evaluation/RGB/", image_path)
        shutil.copy(dst, "/blue/ewhite/DeepForest/NEON_benchmark/images/")

    benchmark_annotations.to_csv("/blue/ewhite/DeepForest/NEON_benchmark/images/test.csv")

    # Copy images to test location
    benchmark_annotations["source"] = "NEON_benchmark"
    for image_path in benchmark_annotations.image_path.unique():
        dst = os.path.join(BENCHMARK_PATH, "evaluation/RGB/", image_path)
        shutil.copy(dst, "/blue/ewhite/DeepForest/NEON_benchmark/images/")

  
    ## Train annotations ##

    BASE_PATH = "/orange/ewhite/b.weinstein/NeonTreeEvaluation/hand_annotations/"
    #convert hand annotations from xml into retinanet format
    xmls = glob.glob(BENCHMARK_PATH + "annotations/" + "*.xml")
    annotation_list = []
    for xml in xmls:
        #check if it is in the directory
        image_name = "{}.tif".format(os.path.splitext(os.path.basename(xml))[0])
        if os.path.exists(os.path.join(BASE_PATH,image_name)):
            print(xml)
            annotation = xml_to_annotations(xml)
            annotation_list.append(annotation)
        
    #Collect hand annotations
    annotations = pd.concat(annotation_list, ignore_index=True)      
    
    #collect shapefile annotations
    shps = glob.glob(BASE_PATH + "*.shp")
    shps_tifs = glob.glob(BASE_PATH + "*.tif")
    shp_results = []
    for shp in shps: 
        print(shp)
        rgb = "{}.tif".format(os.path.splitext(shp)[0])
        shp_df = shapefile_to_annotations(shp, rgb)
        shp_df = pd.DataFrame(shp_df)        
        shp_results.append(shp_df)
    
    shp_results = pd.concat(shp_results,ignore_index=True)
    annotations = pd.concat([annotations,shp_results])
    
    #force dtype
    annotations.xmin = annotations.xmin.astype(int)
    annotations.ymin = annotations.ymin.astype(int)
    annotations.xmax = annotations.xmax.astype(int)
    annotations.ymax = annotations.ymax.astype(int)
    
    annotations.to_csv(BASE_PATH + "hand_annotations.csv",index=False)
    
    #Collect tiles
    xmls = glob.glob(BASE_PATH + "*.xml")
    xmls = [os.path.splitext(os.path.basename(x))[0] for x in xmls] 
    raster_list = [BASE_PATH + x + ".tif" for x in xmls] 
    raster_list = raster_list + shps_tifs 
    
    cropped_annotations = [ ]
    
    for raster in raster_list:
        try:
            annotations_df= split_raster(path_to_raster=raster,
                                            annotations_file=BASE_PATH + "hand_annotations.csv",
                                            save_dir=BASE_PATH + "crops/",
                                            patch_size=400,
                                            patch_overlap=0.05)
        except ValueError:
            continue
        cropped_annotations.append(annotations_df)
    
    ##Gather annotation files into a single file
    train_annotations = pd.concat(cropped_annotations, ignore_index=True)   
    
    #Ensure column order
    train_annotations.to_csv("/blue/ewhite/DeepForest/NEON_benchmark/images/train.csv",index=False, header=True)
   
    train_annotations["source"] = "NEON_benchmark"
    for image_path in train_annotations.image_path.unique():
        dst = os.path.join(BASE_PATH, "crops", image_path)
        shutil.copy(dst, "/blue/ewhite/DeepForest/NEON_benchmark/images/")

def Ryoungseob_2023():
    xmls = glob.glob("/blue/ewhite/DeepForest/Ryoungseob_2023/train_datasets/annotations/*.xml")
    
    #Load and format xmls
    annotation_list = []   
    for xml_path in xmls:
        try:
            annotation = xml_to_annotations(xml_path)
        except:
            continue
        annotation_list.append(annotation)
    annotations = pd.concat(annotation_list, ignore_index=True)      
    annotations["label"] = "Tree"
    annotations["source"] = "Kwon et al. 2023"  
    
    # Train only
    annotations.to_csv("/blue/ewhite/DeepForest/Ryoungseob_2023/train_datasets/images/train.csv")

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
Siberia_polygons()
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
#generate_NEON_benchmark()