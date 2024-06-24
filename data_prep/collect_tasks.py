# Collect individual datasets into boxes, polygon and point datasets
import pandas as pd
import os
import shutil

TreeBoxes = ["/orange/ewhite/DeepForest/Beloiu_2023/pngs/annotations.csv","/orange/ewhite/DeepForest/Ryoungseob_2023/train_datasets/images/train.csv"]
TreePoints = ["/orange/ewhite/DeepForest/TreeFormer/all_images/annotations.csv","/orange/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/annotations.csv"]
TreePolygons = ["/orange/ewhite/DeepForest/Jansen_2023/pngs/annotations.csv"]

# Combine box datasets
TreeBoxes_datasets = []
for dataset in TreeBoxes:
    TreeBoxes_datasets.append(pd.read_csv(dataset))

TreeBoxes_datasets = pd.concat(TreeBoxes_datasets)
train_images = TreeBoxes_datasets.image_path.drop_duplicates().sample(frac=0.8)
TreeBoxes_datasets.loc[TreeBoxes_datasets.image_path.isin(train_images), "split"] = "train"
TreeBoxes_datasets.loc[~TreeBoxes_datasets.image_path.isin(train_images), "split"] = "test"

train = TreeBoxes_datasets[TreeBoxes_datasets.split=="train"]
test = TreeBoxes_datasets[TreeBoxes_datasets.split=="test"]
train.to_csv("/orange/ewhite/DeepForest/MillionTrees/TreeBoxes_v0.0/random_train.csv", index=False)
test.to_csv("/orange/ewhite/DeepForest/MillionTrees/TreeBoxes_v0.0/random_test.csv", index=False)
TreeBoxes_datasets = TreeBoxes_datasets.rename(columns={"image_path":"filename"})

TreeBoxes_datasets.to_csv("/orange/ewhite/DeepForest/MillionTrees/TreeBoxes_v0.0/metadata.csv", index=False)

# Combine point datasets
TreePoints_datasets = []
for dataset in TreePoints:
    TreePoints_datasets.append(pd.read_csv(dataset))
TreePoints_datasets = pd.concat(TreePoints_datasets)
train_images = TreePoints_datasets.image_path.drop_duplicates().sample(frac=0.8)
TreePoints_datasets.loc[TreePoints_datasets.image_path.isin(train_images), "split"] = "train"
TreePoints_datasets.loc[~TreePoints_datasets.image_path.isin(train_images), "split"] = "test"

train = TreePoints_datasets[TreePoints_datasets.split=="train"]
test = TreePoints_datasets[TreePoints_datasets.split=="test"]
train.to_csv("/orange/ewhite/DeepForest/MillionTrees/TreePoints_v0.0/random_train.csv", index=False)
test.to_csv("/orange/ewhite/DeepForest/MillionTrees/TreePoints_v0.0/random_test.csv", index=False)
TreePoints_datasets = TreePoints_datasets.rename(columns={"image_path":"filename"})
TreePoints_datasets.to_csv("/orange/ewhite/DeepForest/MillionTrees/TreePoints_v0.0/metadata.csv", index=False)

# Combine polygon datasets
TreePolygons_datasets = []
for dataset in TreePolygons:
    TreePolygons_datasets.append(pd.read_csv(dataset))
TreePolygons_datasets = pd.concat(TreePolygons_datasets)
train_images = TreePolygons_datasets.image_path.drop_duplicates().sample(frac=0.8)
TreePolygons_datasets.loc[TreePolygons_datasets.image_path.isin(train_images), "split"] = "train"
TreePolygons_datasets.loc[~TreePolygons_datasets.image_path.isin(train_images), "split"] = "test"

train = TreePolygons_datasets[TreePolygons_datasets.split=="train"]
test = TreePolygons_datasets[TreePolygons_datasets.split=="test"]
train.to_csv("/orange/ewhite/DeepForest/MillionTrees/TreePolygons_v0.0/random_train.csv", index=False)
test.to_csv("/orange/ewhite/DeepForest/MillionTrees/TreePolygons_v0.0/random_test.csv", index=False)
TreePolygons_datasets = TreePolygons_datasets.rename(columns={"image_path":"filename"})
TreePolygons_datasets.to_csv("/orange/ewhite/DeepForest/MillionTrees/TreePolygons_v0.0/metadata.csv", index=False)

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
