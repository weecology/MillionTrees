from milliontrees.datasets.TreePolygons import TreePolygonsDataset
from milliontrees.datasets.TreePoints import TreePointsDataset
from milliontrees.datasets.TreeBoxes import TreeBoxesDataset
from milliontrees.common.data_loaders import get_train_loader

import torch

def test_TreePolygons_release():
    dataset = TreePolygonsDataset(download=False, root_dir="/orange/ewhite/DeepForest/MillionTrees/")
    train_dataset = dataset.get_subset("train")
        
    for metadata, image, targets in train_dataset:
        y = targets["y"]
        labels = targets["labels"]
        assert image.shape == (3, 448, 448)
        assert image.dtype == torch.float32
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert y[0].shape == (448, 448)
        assert metadata.shape[0] == 2
    
    train_loader = get_train_loader('standard', train_dataset, batch_size=2)
    for metadata, x, targets in train_loader:
        y = targets["y"]
        assert x.shape == (2, 3, 448, 448)
        assert x.dtype == torch.float32
        assert x.min() >= 0.0 and x.max() <= 1.0
        assert y[0].shape == (1,448, 448)
        assert len(metadata) == 2
        break

def test_TreePoints_release():
    dataset = TreePointsDataset(download=False, root_dir="/orange/ewhite/DeepForest/MillionTrees/")
    train_dataset = dataset.get_subset("train")
    
    for metadata, image, targets in train_dataset:
        points = targets["points"]
        labels = targets["labels"]
        assert image.shape == (3, 448, 448)
        assert image.dtype == torch.float32
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert points.shape[1] == 2
        assert metadata.shape[0] == 2
    
    train_loader = get_train_loader('standard', train_dataset, batch_size=2)
    for metadata, x, targets in train_loader:
        points = targets["points"]
        assert x.shape == (2, 3, 448, 448)
        assert x.dtype == torch.float32
        assert x.min() >= 0.0 and x.max() <= 1.0
        assert points.shape[1] == 2
        assert len(metadata) == 2
        break

def test_TreeBoxes_release():
    dataset = TreeBoxesDataset(download=False, root_dir="/orange/ewhite/DeepForest/MillionTrees/")
    train_dataset = dataset.get_subset("train")
    
    for metadata, image, targets in train_dataset:
        boxes = targets["boxes"]
        labels = targets["labels"]
        assert image.shape == (3, 448, 448)
        assert image.dtype == torch.float32
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert boxes.shape[1] == 4
        assert metadata.shape[0] == 2
    
    train_loader = get_train_loader('standard', train_dataset, batch_size=2)
    for metadata, x, targets in train_loader:
        boxes = targets["boxes"]
        assert x.shape == (2, 3, 448, 448)
        assert x.dtype == torch.float32
        assert x.min() >= 0.0 and x.max() <= 1.0
        assert boxes.shape[1] == 4
        assert len(metadata) == 2
        break