# Collect individual datasets into boxes, polygon and point datasets
import pandas as pd

TreeBoxes = ["/blue/ewhite/DeepForest/Beloiu_2023/pngs/annotations.csv","/blue/ewhite/DeepForest/Ryoungseob_2023/train_datasets/images/train.csv"]

TreePoints = ["/blue/ewhite/DeepForest/TreeFormer/all_images/annotations.csv","/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/annotations.csv"]

TreePolygons = ["/blue/ewhite/DeepForest/Jansen_2023/pngs/annotations.csv","/blue/ewhite/DeepForest/Siberia/annotations.csv","/blue/ewhite/DeepForest/Cloutier2023/images/annotations.csv"]

# Combine box datasets
TreeBoxes_datasets = []
for dataset in TreeBoxes:
    TreeBoxes_datasets.append(pd.read_csv(dataset))

pd.concat(TreeBoxes_datasets).to_csv("/blue/ewhite/DeepForest/MillionTrees/TreeBoxes.csv")

# Combine point datasets
TreePoints_datasets = []
for dataset in TreePoints:
    TreePoints_datasets.append(pd.read_csv(dataset))
pd.concat(TreePoints_datasets).to_csv("/blue/ewhite/DeepForest/MillionTrees/TreePoints.csv")

# Combine polygon datasets
TreePolygons_datasets = []
for dataset in TreePolygons:
    TreePolygons_datasets.append(pd.read_csv(dataset))
pd.concat(TreePolygons_datasets).to_csv("/blue/ewhite/DeepForest/MillionTrees/TreePolygons.csv")
