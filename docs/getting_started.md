# Getting Started

The MillionTrees package is a collection of tree detection datasets. These datasets are organized by annotation geometry, "TreePointsDataset", "TreeBoxesDataset", "TreePolygonDataset". Each of these datasets contain images from many source projects. 

## Download

MillionTrees datasets can be download directly from python

```
dataset = TreePointsDataset(download=True, root_dir=<directory to save data>) 
```

## Visualize

```
for image, label, metadata in dataset:
    plot_points(image, label)
```

## Train
```

```

*Note* To install the train dependencies, please run pip install MillionTrees[train]. These are solely for the reproducible examples.

## Evaluate

## Submit