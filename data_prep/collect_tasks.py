# Collect individual datasets into boxes, polygon and point datasets
import pandas as pd
import os
import shutil
import geopandas as gpd

TreeBoxes = ["/orange/ewhite/DeepForest/Ryoungseob_2023/train_datasets/images/train.csv"]
TreePoints = ["/orange/ewhite/DeepForest/TreeFormer/all_images/annotations.csv","/orange/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/annotations.csv"]
TreePolygons = ["/orange/ewhite/DeepForest/Jansen_2023/pngs/annotations.csv","/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/annotations/annotations.csv"]

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
Boxes_columns = ["xmin","ymin","xmax","ymax","filename","split","source","resolution"]
TreeBoxes_datasets = TreeBoxes_datasets[Boxes_columns]

Polygons_columns = ["polygon","filename","split","source","resolution"]
TreePolygons_datasets = TreePolygons_datasets[Polygons_columns]

Points_columns = ["x","y","filename","split","source","resolution"]
TreePoints_datasets = TreePoints_datasets[Points_columns]

TreePolygons_datasets.to_csv("/orange/ewhite/DeepForest/MillionTrees/TreePolygons_v0.0/official.csv", index=False)
TreePoints_datasets.to_csv("/orange/ewhite/DeepForest/MillionTrees/TreePoints_v0.0/official.csv", index=False)
TreeBoxes_datasets.to_csv("/orange/ewhite/DeepForest/MillionTrees/TreeBoxes_v0.0/official.csv", index=False)

# Zip the files
shutil.make_archive("/orange/ewhite/DeepForest/MillionTrees/TreeBoxes_v0.0", 'zip', "/orange/ewhite/DeepForest/MillionTrees/TreeBoxes_v0.0")
shutil.make_archive("/orange/ewhite/DeepForest/MillionTrees/TreePoints_v0.0", 'zip', "/orange/ewhite/DeepForest/MillionTrees/TreePoints_v0.0")
shutil.make_archive("/orange/ewhite/DeepForest/MillionTrees/TreePolygons_v0.0", 'zip', "/orange/ewhite/DeepForest/MillionTrees/TreePolygons_v0.0")


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
mini_TreeBoxes_datasets = TreeBoxes_datasets.sample(n=1)
mini_TreePoints_datasets = TreePoints_datasets.sample(n=1)
mini_TreePolygons_datasets = TreePolygons_datasets.sample(n=1)

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


shutil.make_archive("/orange/ewhite/DeepForest/MillionTrees/MiniTreeBoxes_v0.0", 'zip', "/orange/ewhite/DeepForest/MillionTrees/MiniTreeBoxes_v0.0")
shutil.make_archive("/orange/ewhite/DeepForest/MillionTrees/MiniTreePoints_v0.0", 'zip', "/orange/ewhite/DeepForest/MillionTrees/MiniTreePoints_v0.0")
shutil.make_archive("/orange/ewhite/DeepForest/MillionTrees/MiniTreePolygons_v0.0", 'zip', "/orange/ewhite/DeepForest/MillionTrees/MiniTreePolygons_v0.0")
