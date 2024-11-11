from milliontrees.datasets.TreePoints import TreePointsDataset
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
def test_TreePoints_generic(dataset):
    dataset = TreePointsDataset(download=False, root_dir=dataset) 
    for metadata, image, targets in dataset:
        points, labels = targets["y"], targets["labels"]
        assert image.shape == (100, 100, 3)
        assert image.dtype == np.float32
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert points.shape == (2, 2)
        assert labels.shape == (2,)
        assert metadata.shape == (2,)
        break

    train_dataset = dataset.get_subset("train")
     
    for metadata, image, targets in train_dataset:
        points, labels = targets["y"], targets["labels"]
        assert image.shape == (3, 448, 448)
        assert image.dtype == torch.float32
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert torch.is_tensor(points)
        assert points.shape == (2, 2)
        assert len(labels) == 2
        assert metadata.shape == (2,)
        break

@pytest.mark.parametrize("batch_size", [1, 2])
def test_get_train_dataloader(dataset, batch_size):
    dataset = TreePointsDataset(download=False, root_dir=dataset) 
    train_dataset = dataset.get_subset("train")
    train_loader = get_train_loader('standard', train_dataset, batch_size=batch_size)
    for metadata, x, targets in train_loader:
        y = targets[0]["y"]
        assert torch.is_tensor(targets[0]["y"])
        assert x.shape == (batch_size, 3, 448, 448)
        assert x.dtype == torch.float32
        assert x.min() >= 0.0 and x.max() <= 1.0
        assert y.shape[1] == 2
        assert len(metadata) == batch_size
        break

def test_get_test_dataloader(dataset):
    dataset = TreePointsDataset(download=False, root_dir=dataset) 
    test_dataset = dataset.get_subset("test")
    
    for metadata, image, targets in test_dataset:
        points, labels = targets["y"], targets["labels"]
        assert image.shape == (3,448, 448)
        assert image.dtype == torch.float32
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert points.shape == (2, 2)
        assert labels.shape == (2,)
        assert metadata.shape == (2,)
        break
    
    # Assert that test_dataset[0] == "image3.jpg"
    metadata, image, targets = test_dataset[0]
    assert metadata[1] == 1
    assert metadata[0] == "image3.jpg"

    test_loader = get_eval_loader('standard', test_dataset, batch_size=2)
    for metadata, x, targets in test_loader:
        y = targets[0]["y"]
        assert torch.is_tensor(targets[0]["y"])
        assert x.shape == (2, 3, 448, 448)
        assert x.dtype == torch.float32
        assert x.min() >= 0.0 and x.max() <= 1.0
        assert y.shape[1] == 2
        assert len(metadata) == 2
        break

def test_TreePoints_eval(dataset):
    dataset = TreePointsDataset(download=False, root_dir=dataset) 
    test_dataset = dataset.get_subset("test")
    test_loader = get_eval_loader('standard', test_dataset, batch_size=2)

    all_y_pred = []
    all_y_true = []
    all_metadata = []
    # Get predictions for the full test set
    for metadata, x, y_true in test_loader:
        y_pred = [{'y': torch.tensor([[30, 70]]), 'label': torch.tensor([0]), 'score': torch.tensor([0.54])} for _ in range(x.shape[0])]
        # Accumulate y_true, y_pred, metadata
        all_y_pred.append(y_pred)
        all_y_true.append(y_true)
        all_metadata.append(metadata)

    # Evaluate
    eval_results, eval_string = dataset.eval(all_y_pred, all_y_true, all_metadata)

    assert len(eval_results) 
    assert "detection_acc_avg" in eval_results.keys()

# Test structure with real annotation data to ensure format is correct
# Do not run on github actions, long running.
@pytest.mark.skipif(not on_hipergator, reason="Do not run on github actions")
def test_TreePoints_release():
    # Lookup size of the train dataset on disk
    dataset = TreePointsDataset(download=False, root_dir="/orange/ewhite/DeepForest/MillionTrees/")
    train_dataset = dataset.get_subset("train")
     
    for metadata, image, targets in train_dataset:
        points = targets["y"]
        labels = targets["labels"]
        assert image.shape == (3, 448, 448)
        assert image.dtype == torch.float32
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert points.shape[1] == 2
        assert metadata.shape[0] == 2
    
    train_loader = get_train_loader('standard', train_dataset, batch_size=2)
    for metadata, x, targets in train_loader:
        y = targets[0]["y"]
        assert torch.is_tensor(targets[0]["y"])
        assert x.shape == (2, 3, 448, 448)
        assert x.dtype == torch.float32
        assert x.min() >= 0.0 and x.max() <= 1.0
        assert y.shape[1] == 2
        assert len(metadata) == 2
        break

def test_TreePoints_download(tmpdir):
    dataset = TreePointsDataset(download=True, root_dir=tmpdir)
    train_dataset = dataset.get_subset("train")
     
    for metadata, image, targets in train_dataset:
        points = targets["points"]
        assert image.shape == (3, 448, 448)
        assert image.dtype == torch.float32
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert points.shape[1] == 2
        assert metadata.shape[0] == 2
        break