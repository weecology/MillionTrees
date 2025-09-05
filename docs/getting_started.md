# Getting Started

## Installation

We highly recommend using conda to isolate a python environment

For example

```
conda create -n MillionTrees python=3.11
```

```
pip install MillionTrees
```

**Note:** On some older systems, there can occasionally be conflicts between torch 2.0 and numpy 2.0. If you encounter errors on loading MillionTrees, try:

```
pip install numpy==1.26.4
```

### Install unsupervised submodule

MillionTrees comes with tens of millions of weakly labeled tree detections. There are extra dependencies for downloading and filtering these datasets.

```
pip install milliontrees[unsupervised]
```

For more information, refer to the section on unsupervised datasets.

##  Dataset structure
The aim of the package is to provide a single interface to load data directly into pytorch without needing to deal with the details of the data format. Users download the data and yield training and evaluation examples from the dataloaders.

```python
from milliontrees.common.data_loaders import get_train_loader
from milliontrees.datasets.TreeBoxes import TreeBoxesDataset

# Download the data; this will take a while. By default sources containing
# 'unsupervised' are excluded. You can override using include/exclude patterns.
dataset = TreeBoxesDataset(download=True)

train_dataset = dataset.get_subset("train")

# View the first image in the dataset
metadata, image, targets = train_dataset[0]
print(f"Metadata length: {len(metadata)}")
print(f"Image shape: {image.shape}, Image type: {type(image)}")
print(f"Targets keys: {targets.keys()}, Label type: {type(targets)}")
```

### Include/Exclude sources

You can select sources to include or exclude using wildcard patterns. Patterns use shell-style matching (e.g., `*unsupervised*`, `NEON_*`). If both are provided, inclusion is applied first, then exclusion.

```python
from milliontrees.datasets.TreeBoxes import TreeBoxesDataset
from milliontrees.datasets.TreePoints import TreePointsDataset
from milliontrees.datasets.TreePolygons import TreePolygonsDataset

# Exclude any source that contains 'unsupervised' (default behavior):
ds_default = TreeBoxesDataset(download=False)

# Explicitly include only NEON sources and exclude a subset
ds_boxes = TreeBoxesDataset(
    download=False,
    include_sources=["NEON*", "Urban*"],
    exclude_sources=["*unsupervised*", "NEON_TestSite*"]
)

# Same API across geometries
ds_points = TreePointsDataset(
    download=False,
    include_sources="*Africa*",
)

ds_polygons = TreePolygonsDataset(
    download=False,
    exclude_sources="*benchmark_old*",
)
```

### Dataloaders

Datasets are batched into lists of target dictionaries, tensors of images, and tensors of metadata.
Each target dictionary contains tensors of the ground truth with the keys dict_keys
(['y', 'labels']). 'y' differs among the TreeGeometry datasets.

```python
train_loader = get_train_loader("standard", train_dataset, batch_size=2)

# Show one batch of the loader
for metadata, image, targets in train_loader:
    print("Targets is a list of dictionaries with the following keys: ", targets[0].keys())
    print(f"Image shape: {image.shape}, Image type: {type(image)}")
    print(f"Annotation shape of the first image: {targets[0]['y'].shape}")
    break
```