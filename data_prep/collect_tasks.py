# Collect individual datasets into boxes, polygon and point datasets
import pandas as pd
import os
import shutil
import geopandas as gpd
import zipfile
from deepforest.visualize import plot_results
from deepforest.utilities import read_file
import cv2

TreeBoxes = [
    "/orange/ewhite/DeepForest/Ryoungseob_2023/train_datasets/images/train.csv",
    "/orange/ewhite/DeepForest/Velasquez_urban_trees/tree_canopies/nueva_carpeta/annotations.csv",
    '/orange/ewhite/DeepForest/individual_urban_tree_crown_detection/annotations.csv',
    '/orange/ewhite/DeepForest/Radogoshi_Sweden/annotations.csv',
    "/orange/ewhite/DeepForest/WRI/WRI-labels-opensource/annotations.csv",
    "/orange/ewhite/DeepForest/Guangzhou2022/annotations.csv",
    "/orange/ewhite/DeepForest/NEON_benchmark/NeonTreeEvaluation_annotations.csv",
    "/orange/ewhite/DeepForest/NEON_benchmark/University_of_Florida.csv",
    '/orange/ewhite/DeepForest/ReForestTree/images/train.csv']

TreePoints = [
    "/orange/ewhite/DeepForest/TreeFormer/all_images/annotations.csv",
    "/orange/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/annotations.csv",
    "/orange/ewhite/MillionTrees/NEON_points/annotations.csv",
    "/orange/ewhite/DeepForest/Tonga/annotations.csv"]

TreePolygons = [
    "/orange/ewhite/DeepForest/Jansen_2023/pngs/annotations.csv",
    "/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/annotations/annotations.csv",
    "/orange/ewhite/DeepForest/Cloutier2023/images/annotations.csv",
    "/orange/ewhite/DeepForest/Firoze2023/annotations.csv",
    "/orange/ewhite/DeepForest/Wagner_Australia/annotations.csv",
    "/orange/ewhite/DeepForest/Alejandro_Chile/alejandro/annotations.csv",
    "/orange/ewhite/DeepForest/UrbanLondon/annotations.csv",
    "/orange/ewhite/DeepForest/OliveTrees_spain/Dataset_RGB/annotations.csv",
    "/orange/ewhite/DeepForest/Araujo_2020/annotations.csv",
    "/orange/ewhite/DeepForest/justdiggit-drone/label_sample/annotations.csv",
    "/orange/ewhite/DeepForest/BCI/BCI_50ha_2020_08_01_crownmap_raw/annotations.csv"
    ]

# Current errors
# Hickman

# Combine box datasets
TreeBoxes_datasets = []
for dataset in TreeBoxes:
    TreeBoxes_datasets.append(pd.read_csv(dataset))

# Make a random split
TreeBoxes_datasets = pd.concat(TreeBoxes_datasets)
train_images = TreeBoxes_datasets.image_path.drop_duplicates().sample(frac=0.8)
TreeBoxes_datasets.loc[TreeBoxes_datasets.image_path.isin(train_images), "split"] = "train"
TreeBoxes_datasets.loc[~TreeBoxes_datasets.image_path.isin(train_images), "split"] = "test"

train = TreeBoxes_datasets[TreeBoxes_datasets.split=="train"]
test = TreeBoxes_datasets[TreeBoxes_datasets.split=="test"]
TreeBoxes_datasets = TreeBoxes_datasets.rename(columns={"image_path":"filename"})

# Make xmin, ymin, xmax, ymax columns from geometry
TreeBoxes_datasets[["xmin","ymin","xmax","ymax"]] = gpd.GeoSeries.from_wkt(TreeBoxes_datasets["geometry"]).bounds

# Combine point datasets
TreePoints_datasets = []
for dataset in TreePoints:
    TreePoints_datasets.append(pd.read_csv(dataset))
TreePoints_datasets = pd.concat(TreePoints_datasets)
train_images = TreePoints_datasets.image_path.drop_duplicates().sample(frac=0.8)
TreePoints_datasets.loc[TreePoints_datasets.image_path.isin(train_images), "split"] = "train"
TreePoints_datasets.loc[~TreePoints_datasets.image_path.isin(train_images), "split"] = "test"
TreePoints_datasets = TreePoints_datasets.rename(columns={"image_path":"filename"})

# Make x,y columns from geometry
TreePoints_datasets["x"] = gpd.GeoSeries.from_wkt(TreePoints_datasets["geometry"]).centroid.x
TreePoints_datasets["y"] = gpd.GeoSeries.from_wkt(TreePoints_datasets["geometry"]).centroid.y

# Combine polygon datasets
TreePolygons_datasets = []
for dataset in TreePolygons:
    TreePolygons_datasets.append(pd.read_csv(dataset))
TreePolygons_datasets = pd.concat(TreePolygons_datasets)
train_images = TreePolygons_datasets.image_path.drop_duplicates().sample(frac=0.8)
TreePolygons_datasets.loc[TreePolygons_datasets.image_path.isin(train_images), "split"] = "train"
TreePolygons_datasets.loc[~TreePolygons_datasets.image_path.isin(train_images), "split"] = "test"
TreePolygons_datasets["polygon"] = gpd.GeoDataFrame(TreePolygons_datasets.geometry).to_wkt()

train = TreePolygons_datasets[TreePolygons_datasets.split=="train"]
test = TreePolygons_datasets[TreePolygons_datasets.split=="test"]
TreePolygons_datasets = TreePolygons_datasets.rename(columns={"image_path":"filename"})


# Assert that none of the geometry columns have bounds greater than 20,000, checking for non-geographic coordinates
assert TreeBoxes_datasets.xmin.max() < 20000
assert TreeBoxes_datasets.ymin.max() < 20000
assert TreeBoxes_datasets.xmax.max() < 20000
assert TreeBoxes_datasets.ymax.max() < 20000

assert TreePoints_datasets.x.max() < 20000
assert TreePoints_datasets.y.max() < 30000

# Create release txt
with open("/orange/ewhite/DeepForest/MillionTrees/TreeBoxes_v0.0/RELEASE_v0.0.txt", "w") as outfile:
    outfile.write("Initial debug")

# Create release txt
with open("/orange/ewhite/DeepForest/MillionTrees/TreePolygons_v0.0/RELEASE_v0.0.txt", "w") as outfile:
    outfile.write("Initial debug")

# Create release txt
with open("/orange/ewhite/DeepForest/MillionTrees/TreePoints_v0.0/RELEASE_v0.0.txt", "w") as outfile:
    outfile.write("Initial debug")

# Copy images
for image in TreeBoxes_datasets.filename.unique():
    destination = "/orange/ewhite/DeepForest/MillionTrees/TreeBoxes_v0.0/images/"
    if not os.path.exists(destination + os.path.basename(image)):
        shutil.copy(image, destination)

for image in TreePoints_datasets.filename.unique():
    destination = "/orange/ewhite/DeepForest/MillionTrees/TreePoints_v0.0/images/"
    if not os.path.exists(destination + os.path.basename(image)):
        shutil.copy(image, destination)

for image in TreePolygons_datasets.filename.unique():
    destination = "/orange/ewhite/DeepForest/MillionTrees/TreePolygons_v0.0/images/"
    if not os.path.exists(destination + os.path.basename(image)):
        shutil.copy(image, destination)

# change filenames to relative path
TreeBoxes_datasets["filename"] = TreeBoxes_datasets["filename"].apply(os.path.basename)
TreePoints_datasets["filename"] = TreePoints_datasets["filename"].apply(os.path.basename)
TreePolygons_datasets["filename"] = TreePolygons_datasets["filename"].apply(os.path.basename)

# Save splits

# Clean the columns
Boxes_columns = ["xmin","ymin","xmax","ymax","filename","split","source"]
TreeBoxes_datasets = TreeBoxes_datasets[Boxes_columns]

Polygons_columns = ["polygon","filename","split","source"]
TreePolygons_datasets = TreePolygons_datasets[Polygons_columns]

Points_columns = ["x","y","filename","split","source"]
TreePoints_datasets = TreePoints_datasets[Points_columns]

# Make sure there are no duplicates
TreeBoxes_datasets = TreeBoxes_datasets.drop_duplicates()
TreePolygons_datasets = TreePolygons_datasets.drop_duplicates()
TreePoints_datasets = TreePoints_datasets.drop_duplicates()

TreePolygons_datasets.to_csv("/orange/ewhite/DeepForest/MillionTrees/TreePolygons_v0.0/official.csv", index=False)
TreePoints_datasets.to_csv("/orange/ewhite/DeepForest/MillionTrees/TreePoints_v0.0/official.csv", index=False)
TreeBoxes_datasets.to_csv("/orange/ewhite/DeepForest/MillionTrees/TreeBoxes_v0.0/official.csv", index=False)

# Create github test versions by taking one image and annotation from each dataset
# Create directories for mini datasets
# Delete directories if they already exist
if os.path.exists("/orange/ewhite/DeepForest/MillionTrees/MiniTreeBoxes_v0.0/images"):
    shutil.rmtree("/orange/ewhite/DeepForest/MillionTrees/MiniTreeBoxes_v0.0/images")
if os.path.exists("/orange/ewhite/DeepForest/MillionTrees/MiniTreePoints_v0.0/images"):
    shutil.rmtree("/orange/ewhite/DeepForest/MillionTrees/MiniTreePoints_v0.0/images")
if os.path.exists("/orange/ewhite/DeepForest/MillionTrees/MiniTreePolygons_v0.0/images"):
    shutil.rmtree("/orange/ewhite/DeepForest/MillionTrees/MiniTreePolygons_v0.0/images")
    
os.makedirs("/orange/ewhite/DeepForest/MillionTrees/MiniTreeBoxes_v0.0/images", exist_ok=True)
os.makedirs("/orange/ewhite/DeepForest/MillionTrees/MiniTreePoints_v0.0/images", exist_ok=True)
os.makedirs("/orange/ewhite/DeepForest/MillionTrees/MiniTreePolygons_v0.0/images", exist_ok=True)

# Create mini versions of the datasets
mini_TreeBoxes_datasets = TreeBoxes_datasets.groupby("source").first().reset_index(drop=True)
mini_TreePoints_datasets = TreePoints_datasets.groupby("source").first().reset_index(drop=True)
mini_TreePolygons_datasets = TreePolygons_datasets.groupby("source").first().reset_index(drop=True)

# Get the filenames from the mini datasets
mini_TreeBoxes_filenames = mini_TreeBoxes_datasets["filename"].tolist()
mini_TreePoints_filenames = mini_TreePoints_datasets["filename"].tolist()
mini_TreePolygons_filenames = mini_TreePolygons_datasets["filename"].tolist()

# Select all annotations from the mini datasets
mini_TreeBoxes_annotations = TreeBoxes_datasets[TreeBoxes_datasets["filename"].isin(mini_TreeBoxes_filenames)]
mini_TreePoints_annotations = TreePoints_datasets[TreePoints_datasets["filename"].isin(mini_TreePoints_filenames)]
mini_TreePolygons_annotations = TreePolygons_datasets[TreePolygons_datasets["filename"].isin(mini_TreePolygons_filenames)]

# Create release txt for mini datasets
with open("/orange/ewhite/DeepForest/MillionTrees/MiniTreeBoxes_v0.0/RELEASE_v0.0.txt", "w") as outfile:
    outfile.write("Initial debug")

# Create release txt for mini datasets
with open("/orange/ewhite/DeepForest/MillionTrees/MiniTreePolygons_v0.0/RELEASE_v0.0.txt", "w") as outfile:
    outfile.write("Initial debug")

# Create release txt for mini datasets
with open("/orange/ewhite/DeepForest/MillionTrees/MiniTreePoints_v0.0/RELEASE_v0.0.txt", "w") as outfile:
    outfile.write("Initial debug")

# Create zip files for mini datasets
mini_TreeBoxes_annotations.to_csv("/orange/ewhite/DeepForest/MillionTrees/MiniTreeBoxes_v0.0/official.csv", index=False)
mini_TreePoints_annotations.to_csv("/orange/ewhite/DeepForest/MillionTrees/MiniTreePoints_v0.0/official.csv", index=False)
mini_TreePolygons_annotations.to_csv("/orange/ewhite/DeepForest/MillionTrees/MiniTreePolygons_v0.0/official.csv", index=False)

# Copy images for mini datasets
for image in mini_TreeBoxes_filenames:
    destination = "/orange/ewhite/DeepForest/MillionTrees/MiniTreeBoxes_v0.0/images/"
    if not os.path.exists(destination + os.path.basename(image)):
        shutil.copy("/orange/ewhite/DeepForest/MillionTrees/TreeBoxes_v0.0/images/" + image, destination)

for image in mini_TreePoints_filenames:
    destination = "/orange/ewhite/DeepForest/MillionTrees/MiniTreePoints_v0.0/images/"
    if not os.path.exists(destination + os.path.basename(image)):
        shutil.copy("/orange/ewhite/DeepForest/MillionTrees/TreePoints_v0.0/images/" + image, destination)

for image in mini_TreePolygons_filenames:
    destination = "/orange/ewhite/DeepForest/MillionTrees/MiniTreePolygons_v0.0/images/"
    if not os.path.exists(destination + os.path.basename(image)):
        shutil.copy("/orange/ewhite/DeepForest/MillionTrees/TreePolygons_v0.0/images/" + image, destination)

# Write examples from the mini datasets to the MillionTrees doc folder
mini_TreeBoxes_annotations.root_dir = "/orange/ewhite/DeepForest/MillionTrees/MiniTreeBoxes_v0.0/images/"
mini_TreePoints_annotations.root_dir = "/orange/ewhite/DeepForest/MillionTrees/MiniTreePoints_v0.0/images/"
mini_TreePolygons_annotations.root_dir = "/orange/ewhite/DeepForest/MillionTrees/MiniTreePolygons_v0.0/images/"

mini_TreeBoxes_annotations["label"] = "Tree"
mini_TreePoints_annotations["label"] = "Tree"
mini_TreePolygons_annotations["label"] = "Tree"

mini_TreeBoxes_annotations["score"] = None
mini_TreePoints_annotations["score"] = None
mini_TreePolygons_annotations["score"] = None

for source, group in mini_TreeBoxes_annotations.groupby("source"):
    group["image_path"] = group["filename"]
    group = read_file(group, root_dir="/orange/ewhite/DeepForest/MillionTrees/MiniTreeBoxes_v0.0/images/")
    group.root_dir = "/orange/ewhite/DeepForest/MillionTrees/MiniTreeBoxes_v0.0/images/"
    # remove any spaces in source
    source = source.replace(" ", "_")
    plot_results(results=group, savedir="/home/b.weinstein/MillionTrees/docs/public/", basename=source)

for source, group in mini_TreePoints_annotations.groupby("source"):
    group["image_path"] = group["filename"]
    group = read_file(group, root_dir="/orange/ewhite/DeepForest/MillionTrees/MiniTreePoints_v0.0/images/")
    group.root_dir = "/orange/ewhite/DeepForest/MillionTrees/MiniTreePoints_v0.0/images/"
    source = source.replace(" ", "_")
    plot_results(group, savedir="/home/b.weinstein/MillionTrees/docs/public/", basename=source)

for source, group in mini_TreePolygons_annotations.groupby("source"):
    group["image_path"] = group["filename"]
    group = read_file(group, root_dir="/orange/ewhite/DeepForest/MillionTrees/MiniTreePolygons_v0.0/images/")
    group.root_dir = "/orange/ewhite/DeepForest/MillionTrees/MiniTreePolygons_v0.0/images/"
    height, width, channels = cv2.imread("/orange/ewhite/DeepForest/MillionTrees/MiniTreePolygons_v0.0/images/" + group.image_path.iloc[0]).shape
    source = source.replace(" ", "_")
    plot_results(group, savedir="/home/b.weinstein/MillionTrees/docs/public/", basename=source, height=height, width=width)

# Zip the files
def zip_directory(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                zipf.write(os.path.join(root, file),
                            os.path.relpath(os.path.join(root, file), 
                                            os.path.join(folder_path, '..')))

zip_directory("/orange/ewhite/DeepForest/MillionTrees/TreeBoxes_v0.0", "/orange/ewhite/DeepForest/MillionTrees/TreeBoxes_v0.0.zip")
zip_directory("/orange/ewhite/DeepForest/MillionTrees/TreePoints_v0.0", "/orange/ewhite/DeepForest/MillionTrees/TreePoints_v0.0.zip")
zip_directory("/orange/ewhite/DeepForest/MillionTrees/TreePolygons_v0.0", "/orange/ewhite/DeepForest/MillionTrees/TreePolygons_v0.0.zip")


zip_directory("/orange/ewhite/DeepForest/MillionTrees/MiniTreeBoxes_v0.0", "/orange/ewhite/DeepForest/MillionTrees/MiniTreeBoxes_v0.0.zip")
zip_directory("/orange/ewhite/DeepForest/MillionTrees/MiniTreePoints_v0.0", "/orange/ewhite/DeepForest/MillionTrees/MiniTreePoints_v0.0.zip")
zip_directory("/orange/ewhite/DeepForest/MillionTrees/MiniTreePolygons_v0.0", "/orange/ewhite/DeepForest/MillionTrees/MiniTreePolygons_v0.0.zip")
