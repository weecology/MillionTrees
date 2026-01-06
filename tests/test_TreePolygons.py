from milliontrees.datasets.TreePolygons import TreePolygonsDataset
from milliontrees.common.data_loaders import get_train_loader, get_eval_loader

import torch
import pytest
import numpy as np
from shapely import from_wkt
import requests

# Test structure without real annotation data to ensure format is correct
def test_TreePolygons_generic(dataset):
    ds = TreePolygonsDataset(download=False, root_dir=dataset, version="0.0") 
    for metadata, image, targets in ds:
        masks = targets["y"]
        labels = targets["labels"]
        boxes = targets["bboxes"]
        assert image.shape == (100, 100, 3)
        assert image.dtype == np.float32
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert masks.shape == (1, 100, 100)
        assert len(labels) == 1
        assert boxes.shape == (1,4)
        assert metadata.shape == (2,)
        break

    train_dataset = ds.get_subset("train")
     
    for metadata, image, targets in train_dataset:
        masks = targets["y"]
        boxes = targets["bboxes"]
        labels = targets["labels"]
        assert image.shape == (3, 448, 448)
        assert image.dtype == torch.float32
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert masks.shape == (1,448, 448)
        assert len(labels) == 1
        assert boxes.shape == (1,4)
        assert metadata.shape == (2,)
        break

@pytest.mark.parametrize("batch_size", [1, 2])
def test_get_train_dataloader(dataset, batch_size):
    ds = TreePolygonsDataset(download=False, root_dir=dataset, version="0.0") 
    train_dataset = ds.get_subset("train")
    train_loader = get_train_loader('standard', train_dataset, batch_size=batch_size)
    for metadata, x, targets in train_loader:
        for i in targets:
            assert i["y"].shape == (1,448, 448)
            assert i["bboxes"].shape == (1,4)
            assert i["labels"].shape == (1,)
        
        assert x.shape == (batch_size, 3, 448, 448)
        assert x.dtype == torch.float32
        assert x.min() >= 0.0 and x.max() <= 1.0
        assert len(metadata) == batch_size

def test_get_test_dataloader(dataset):
    ds = TreePolygonsDataset(download=False, root_dir=dataset, version="0.0") 
    test_dataset = ds.get_subset("test")
    
    for metadata, image, targets in test_dataset:
        assert targets["y"].shape == (1,448, 448)
        assert targets["labels"].shape == torch.Size([1])
        assert targets["bboxes"].shape == torch.Size([1, 4])
        assert image.shape == (3,448, 448)
        assert image.dtype == torch.float32
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert metadata.shape == torch.Size([2])
    
    test_loader = get_eval_loader('standard', test_dataset, batch_size=1)
    for metadata, x, targets in test_loader:
        for i in targets:
            masks = i["y"]
            boxes = i["bboxes"]
            labels = i["labels"]
            assert masks.shape == (1,448, 448)
            assert labels.shape == torch.Size([1])
            assert boxes.shape == torch.Size([1, 4])

        assert x.shape == (1, 3, 448, 448)
        assert x.dtype == torch.float32
        assert x.min() >= 0.0 and x.max() <= 1.0
        assert len(metadata) == 1

def test_TreePolygons_eval(dataset):
    ds = TreePolygonsDataset(download=False, root_dir=dataset, version="0.0") 
    test_dataset = ds.get_subset("test")
    test_loader = get_eval_loader('standard', test_dataset, batch_size=2)

    all_y_pred = []
    all_y_true = []
    
    # Get predictions for the full test set
    for metadata, x, y_true in test_loader:
        # Construct the mask for true positive for image1.jpg
        polygon = from_wkt("POLYGON((10 15, 50 15, 50 55, 10 55, 10 15))")
        height, width = x.shape[2], x.shape[3]
        pred_mask = ds.create_polygon_mask(width=width, height=height, vertices=polygon)
        pred_box = torch.tensor([10, 15, 50, 55]).unsqueeze(0)
        batch = [{'y': torch.tensor([pred_mask]), 'bboxes':pred_box,'labels': torch.tensor([0]), 'scores': torch.tensor([0.54])}]
        
        # Accumulate y_true, y_pred, metadata
        all_y_pred.extend(batch)
        all_y_true.extend(y_true)

    # Evaluate
    eval_results, eval_string = ds.eval(y_pred=all_y_pred,y_true=all_y_true, metadata=test_dataset.metadata_array)
    
    # The above example has one true positive and two false negatives = 0.33 accuracy and recall
    eval_results["accuracy"]["mask_acc_avg"] == 0.33
    assert len(eval_results) 
    assert "accuracy" in eval_results.keys()
    assert "recall" in eval_results.keys()

def test_TreePolygons_download_url(dataset):
    ds = TreePolygonsDataset(download=False, root_dir=dataset, version="0.0")
    for version in ds._versions_dict.keys():
        print(version)
        # Confirm url can be downloaded
        url = ds._versions_dict[version]['download_url']
        # If the url is not accessible, skip the test
        if url == "":
            continue
        response = requests.head(url, allow_redirects=True)
        assert response.status_code == 200, f"URL {url} is not accessible"