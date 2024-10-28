from milliontrees.datasets.TreeBoxes import TreeBoxesDataset
from milliontrees.common.data_loaders import get_train_loader, get_eval_loader

import torch
import pytest
import os
import pandas as pd
import numpy as np

# Check if running on hipergator
if os.path.exists("/orange"):
    on_hipergator = True
else:
    on_hipergator = False

# Test structure without real annotation data to ensure format is correct
def test_TreeBoxes_generic(dataset):
    dataset = TreeBoxesDataset(download=False, root_dir=dataset) 
    for metadata, image, targets in dataset:
        boxes, labels = targets["boxes"], targets["labels"]
        assert image.shape == (100, 100, 3)
        assert image.dtype == np.float32
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert boxes.shape == (2, 4)
        assert labels.shape == (2,)
        assert metadata.shape == (2,2)
        break

    train_dataset = dataset.get_subset("train")
     
    for metadata, image, targets in train_dataset:
        boxes, labels = targets["boxes"], targets["labels"]
        assert image.shape == (3, 448, 448)
        assert image.dtype == torch.float32
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert torch.is_tensor(boxes)
        assert boxes.shape == (2,4)
        assert len(labels) == 2
        assert metadata.shape == (2,2)
        break

@pytest.mark.parametrize("batch_size", [1, 2])
def test_get_train_dataloader(dataset, batch_size):
    dataset = TreeBoxesDataset(download=False, root_dir=dataset) 
    train_dataset = dataset.get_subset("train")
    train_loader = get_train_loader('standard', train_dataset, batch_size=batch_size)
    for metadata, x, targets in train_loader:
        y = targets[0]["boxes"]
        assert torch.is_tensor(targets[0]["boxes"])
        assert x.shape == (batch_size, 3, 448, 448)
        assert x.dtype == torch.float32
        assert x.min() >= 0.0 and x.max() <= 1.0
        assert y.shape[1] == 4
        assert len(metadata) == batch_size
        break

def test_get_test_dataloader(dataset, batch_size):
    dataset = TreeBoxesDataset(download=False, root_dir=dataset) 
    test_dataset = dataset.get_subset("test")
    
    for metadata, image, targets in test_dataset:
        boxes, labels = targets["boxes"], targets["labels"]
        assert image.shape == (100, 100, 3)
        assert image.dtype == np.float32
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert boxes.shape == (2, 4)
        assert labels.shape == (2,)
        assert metadata.shape == (2,2)
        break

    test_loader = get_eval_loader('standard', test_dataset, batch_size=batch_size)
    for metadata, x, targets in test_loader:
        y = targets[0]["boxes"]
        assert torch.is_tensor(targets[0]["boxes"])
        assert x.shape == (batch_size, 3, 448, 448)
        assert x.dtype == torch.float32
        assert x.min() >= 0.0 and x.max() <= 1.0
        assert y.shape[1] == 4
        assert len(metadata) == batch_size
        break

# Test structure with real annotation data to ensure format is correct
# Do not run on github actions, long running.
@pytest.mark.skipif(not on_hipergator, reason="Do not run on github actions")
def test_TreeBoxes_release():
    # Lookup size of the train dataset on disk
    dataset = TreeBoxesDataset(download=False, root_dir="/orange/ewhite/DeepForest/MillionTrees/")
    train_dataset = dataset.get_subset("train")
     
    for metadata, image, targets in train_dataset:
        boxes = targets["boxes"]
        labels = targets["labels"]
        assert image.shape == (3, 448, 448)
        assert image.dtype == torch.float32
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert boxes.shape[1] == 4
        assert metadata.shape[1] == 2
    
    train_loader = get_train_loader('standard', train_dataset, batch_size=2)
    for metadata, x, targets in train_loader:
        y = targets[0]["boxes"]
        assert torch.is_tensor(targets[0]["boxes"])
        assert x.shape == (2, 3, 448, 448)
        assert x.dtype == torch.float32
        assert x.min() >= 0.0 and x.max() <= 1.0
        assert y.shape[1] == 4
        assert len(metadata) == 2
        break

def test_TreeBoxes_download(tmpdir):
    dataset = TreeBoxesDataset(download=True, root_dir=tmpdir)
    train_dataset = dataset.get_subset("train")
     
    for metadata, image, targets in train_dataset:
        boxes = targets["boxes"]
        assert image.shape == (3, 448, 448)
        assert image.dtype == torch.float32
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert boxes.shape[1] == 4
        assert metadata.shape[1] == 1
        break