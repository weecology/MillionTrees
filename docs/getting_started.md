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

## Load the data

The aim of the package is to provide a single interface to load data directly into pytorch without needing to deal with the details of the data format. Users download the data and yield training and evaluation examples from the dataloaders.

```
from milliontrees.common.data_loaders import get_train_loader
from milliontrees.datasets.TreeBoxes import TreeBoxesDataset

# Download the data; this will take a while
dataset = TreeBoxesDataset(download=True)

train_dataset = dataset.get_subset("train")

# View the first image in the dataset
metadata, image, targets = train_dataset[0]
print(f"Metadata length: {len(metadata)}")
print(f"Image shape: {image.shape}, Image type: {type(image)}")
print(f"Targets keys: {targets.keys()}, Label type: {type(targets)}")
```

### Dataloaders

Datasets are batched into lists of target dictionaries, tensors of images, and tensors of metadata.
Each target dictionary contains tensors of the ground truth with the keys dict_keys
(['y', 'labels']). 'y' differs among the TreeGeometry datasets.

```
train_loader = get_train_loader("standard", train_dataset, batch_size=2)

# Show one batch of the loader
for metadata, image, targets in train_loader:
    print("Targets is a list of dictionaries with the following keys: ", targets[0].keys())
    print(f"Image shape: {image.shape}, Image type: {type(image)}")
    print(f"Annotation shape of the first image: {targets[0]['y'].shape}")
```