# Getting Started

## Installation

```
pip install milliontrees
```

**Note:** On some older systems, there can occasionally be conflicts between torch 2.0 and numpy 2.0. If you encounter errors on loading MillionTrees, try downgrading numpy to pre2.0. 

```
pip install numpy==1.26.4
```

## Dataset release sizes

Before working with the full datasets (which can be several GB), use a smaller release size. All sizes keep the same dataloader API and on-disk layout.

| Flag | Archive | Images per source | Split CSVs |
|------|---------|-------------------|------------|
| `mini=True` | `MiniTree*` | 3 | `random` only |
| `small=True` | `SmallTree*` | Up to 50 | All split schemes |
| (default) | `Tree*` | Full release | All split schemes |

```python
from milliontrees import get_dataset

# Fastest smoke tests
dataset = get_dataset('TreeBoxes', download=True, mini=True)

# Medium subset with zeroshot / crossgeometry support
dataset = get_dataset('TreePoints', download=True, small=True, split_scheme='zeroshot')

train_dataset = dataset.get_subset("train")
print(f"Dataset size: {len(train_dataset)} images")
```

Mini and small archives are published for all three geometry datasets (`TreeBoxes`, `TreePoints`, `TreePolygons`). Set neither flag for the full release when training at scale.

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
if "tree_coverage_mask" in targets:
    print(f"Coverage mask shape: {targets['tree_coverage_mask'].shape}")
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
(['y', 'labels', 'tree_coverage_mask']). 'y' differs among the TreeGeometry datasets.

```python
train_loader = get_train_loader("standard", train_dataset, batch_size=2)

# Show one batch of the loader
for metadata, image, targets in train_loader:
    print("Targets is a list of dictionaries with the following keys: ", targets[0].keys())
    print(f"Image shape: {image.shape}, Image type: {type(image)}")
    print(f"Annotation shape of the first image: {targets[0]['y'].shape}")
    if "tree_coverage_mask" in targets[0]:
        print(f"Coverage mask shape of the first image: {targets[0]['tree_coverage_mask'].shape}")
    break
```

### Orientation Helpers for New Users

All dataset constructors support a `verbose` flag (default `True`) that prints:
- dataset/version and local path
- train/test image counts
- annotation and source counts
- active include/exclude source filters

```python
from milliontrees import get_dataset

# Keep verbose=True while exploring, disable later if needed
dataset = get_dataset("TreeBoxes", download=True, mini=True, verbose=True)
```

MillionTrees also includes onboarding visualization helpers:

```python
from milliontrees import get_dataset
from milliontrees.common.onboarding import (
    get_latest_release_sizes,
    plot_release_size_summary,
    save_sample_visualization,
)

dataset = get_dataset("TreePoints", download=True, mini=True)
save_sample_visualization(dataset, "treepoints_sample.png", split="train", index=0)

sizes = get_latest_release_sizes()
plot_release_size_summary("current_data_sizes.png", sizes)
```

## Why are there incomplete annotations? Why aren't all annotations perfect?

The MillionTrees dataset is designed to meet the need for a global source while acknowledging the limitations of data collection for training and evaluation data. To capture the breadth of resolutions, backgrounds, and trees at a global extent, we must accommodate a range of annotation geometries and annotation approaches. Rather than reduce the breadth of the dataset in favor of a few ideal datasets, we believe that benchmark datasets should include a wide array of situations accompanied with tools to navigate the differences among datasets. Our hope is that the standardization and centralization of the data will help transcend current limitations in model development. For example, current models cannot easily cross among annotation inputs, a typical neural network uses boxes, or points, or polygons, but not all three. Some models may be much more sensitive to incomplete annotation approaches, requiring them to be trained with less diverse, but higher quality data. The MillionTrees benchmark is designed to bring focus to these challenges in applied machine learning. We opted to include all together, because we hope to put the need first and build architectures that meet that need. 
