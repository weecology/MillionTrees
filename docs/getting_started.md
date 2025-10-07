# Getting Started

## Installation

```
pip install MillionTrees
```

**Note:** On some older systems, there can occasionally be conflicts between torch 2.0 and numpy 2.0. If you encounter errors on loading MillionTrees, try downgrading numpy to pre2.0. 

```
pip install numpy==1.26.4
```

### Optional: Unsupervised submodule

MillionTrees comes with tens of millions of weakly labeled tree detections. There are extra dependencies for downloading and filtering these datasets. These data are large and optional, only needed for training large scale models.

```
pip install milliontrees[unsupervised]
```

For more information, refer to the section on unsupervised datasets.

## Mini Datasets for Development

**Recommended for Development**: Before working with the full datasets (which can be several GB), we recommend starting with the mini versions for development and testing. Mini datasets contain a small subset of the data but maintain the same structure and format.

```python
from milliontrees import get_dataset

# Download a mini version of TreeBoxes (~few MB instead of ~several GB)
dataset = get_dataset('TreeBoxes', download=True, mini=True)

# This works the same as the full dataset but much faster to download
train_dataset = dataset.get_subset("train")
print(f"Mini dataset size: {len(train_dataset)} images")
```

Mini datasets are available for all three dataset types:
- `TreeBoxes` → `MiniTreeBoxes` 
- `TreePoints` → `MiniTreePoints`
- `TreePolygons` → `MiniTreePolygons`

Once you've developed and tested your code with the mini datasets, simply remove `mini=True` to use the full datasets for training and evaluation.

##  Dataset structure

The aim of the package is to provide a single interface to load data directly into pytorch without needing to deal with the details of the data format. Users download the data and yield training and evaluation examples from the dataloaders.

```python
from milliontrees.common.data_loaders import get_train_loader
from milliontrees.datasets.TreeBoxes import TreeBoxesDataset

# Download the data; this will take a while. By default sources containing
# 'unsupervised' are excluded. You can override using include/exclude patterns.
# For development, consider using mini=True for faster downloads
dataset = TreeBoxesDataset(download=True)  # Add mini=True for development

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