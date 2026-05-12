from pathlib import Path

from milliontrees.datasets.TreeBoxes import TreeBoxesDataset
from milliontrees.common.data_loaders import get_train_loader, get_eval_loader
from milliontrees.common.metrics.all_metrics import MaskAwareDetectionPrecision

import torch
import pytest
import numpy as np
import requests
import os

# Test structure without real annotation data to ensure format is correct
def test_TreeBoxes_generic(dataset):
    ds = TreeBoxesDataset(download=False, root_dir=dataset,version="0.0") 
    for metadata, image, targets in ds:
        boxes, labels = targets["y"], targets["labels"]
        assert image.shape == (100, 100, 3)
        assert image.dtype == np.float32
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert boxes.shape == (2, 4)
        assert labels.shape == (2,)
        assert metadata.shape == (2,)
        break

    train_dataset = ds.get_subset("train")
     
    for metadata, image, targets in train_dataset:
        boxes, labels = targets["y"], targets["labels"]
        assert image.shape == (3, 448, 448)
        assert image.dtype == torch.float32
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert torch.is_tensor(boxes)
        assert boxes.shape == (2,4)
        assert len(labels) == 2
        assert metadata.shape == (2,)
        break


def test_TreeBoxes_full_dataset_iteration(dataset):
    ds = TreeBoxesDataset(download=False, root_dir=dataset, version="0.0")
    assert len(ds) == len(ds._input_array) == 4

    for idx in range(len(ds)):
        metadata, image, targets = ds[idx]
        boxes, labels = targets["y"], targets["labels"]
        assert image.shape == (100, 100, 3)
        assert image.dtype == np.float32
        assert boxes.shape[1] == 4
        assert len(labels) == len(boxes)
        assert metadata.shape == (2,)


# confirm that we can change target name is needed
def test_get_dataset_with_geometry_name(dataset):
    ds = TreeBoxesDataset(download=False, root_dir=dataset, geometry_name="boxes",version="0.0") 
    train_dataset = ds.get_subset("train")

    for metadata, image, targets in train_dataset:
        boxes, labels = targets["boxes"], targets["labels"]
        break

    # Test the dataloader
    train_loader = get_train_loader('standard', train_dataset, batch_size=2)

    all_y_pred = []
    all_y_true = []
    # Get predictions for the full test set
    for metadata, x, y_true in train_loader:
        labels = torch.zeros(x.shape[0])
        pred_tensor = [[30, 70, 35, 75]]
        scores = torch.stack([torch.tensor(0.54) for x in range(len(pred_tensor))])
        y_pred = [{'boxes': torch.tensor(pred_tensor), 'label': labels, 'scores': scores} for _ in range(x.shape[0])]
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    # Concat and Evaluate
    eval_results, eval_string = ds.eval(y_pred=all_y_pred,y_true=all_y_true, metadata=train_dataset.metadata_array)

@pytest.mark.parametrize("batch_size", [1, 2])
def test_get_train_dataloader(dataset, batch_size):
    ds = TreeBoxesDataset(download=False, root_dir=dataset, version="0.0") 
    train_dataset = ds.get_subset("train")
    train_loader = get_train_loader('standard', train_dataset, batch_size=batch_size)
    for metadata, x, targets in train_loader:
        y = targets[0]["y"]
        assert torch.is_tensor(targets[0]["y"])
        assert x.shape == (batch_size, 3, 448, 448)
        assert x.dtype == torch.float32
        assert x.min() >= 0.0 and x.max() <= 1.0
        assert y.shape[1] == 4
        assert len(metadata) == batch_size
        break

def test_get_test_dataloader(dataset):
    ds = TreeBoxesDataset(download=False, root_dir=dataset,version="0.0") 
    test_dataset = ds.get_subset("test")
    
    for metadata, image, targets in test_dataset:
        boxes, labels = targets["y"], targets["labels"]
        assert image.shape == (3,448, 448)
        assert image.dtype == torch.float32
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert boxes.shape == (2, 4)
        assert labels.shape == (2,)
        assert metadata.shape == (2,)
        break
    
    # Assert that test_dataset[0] == "image3.jpg"
    metadata, image, targets = test_dataset[0]
    assert metadata[1] == 0
    assert ds._filename_id_to_code[int(metadata[0])] == "image3.jpg"

    test_loader = get_eval_loader('standard', test_dataset, batch_size=2)
    for metadata, x, targets in test_loader:
        y = targets[0]["y"]
        assert torch.is_tensor(targets[0]["y"])
        assert x.shape == (2, 3, 448, 448)
        assert x.dtype == torch.float32
        assert x.min() >= 0.0 and x.max() <= 1.0
        assert y.shape[1] == 4
        assert metadata.shape[0] == 2
        break
    
@pytest.mark.parametrize("pred_tensor", [[[134, 156, 313, 336]], [[30, 70, 35, 75],[30, 20, 35, 55]]], ids=["single", "multiple"])
def test_TreeBoxes_eval(dataset, pred_tensor):
    ds = TreeBoxesDataset(download=False, root_dir=dataset,version="0.0") 
    test_dataset = ds.get_subset("test")
    test_loader = get_eval_loader('standard', test_dataset, batch_size=2)

    all_y_pred = []
    all_y_true = []
    # Get predictions for the full test set
    for metadata, x, y_true in test_loader:
        labels = torch.zeros(x.shape[0])
        scores = torch.stack([torch.tensor(0.54) for x in range(len(pred_tensor))])
        y_pred = [{'y': torch.tensor(pred_tensor), 'label': labels, 'scores': scores} for _ in range(x.shape[0])]
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    # Concat and Evaluate
    eval_results, eval_string = ds.eval(y_pred=all_y_pred,y_true=all_y_true, metadata=test_dataset.metadata_array)

    if pred_tensor == [[134, 156, 313, 336]]:
        # Greedy 1:1 matching at IoU 0.4: one image scores 0, the other 0.5 → mean 0.25
        assert eval_results["accuracy"]["detection_accuracy_avg"] == 0.25

    assert len(eval_results)
    assert "accuracy" in eval_results.keys()
    assert "recall" in eval_results.keys()
    assert "maskaware_precision" in eval_results.keys()
    assert "merge_commission" in eval_results.keys()

def test_TreeBoxes_eval_visualization(dataset, tmp_path):
    ds = TreeBoxesDataset(download=False, root_dir=dataset, version="0.0")
    test_dataset = ds.get_subset("test")
    test_loader = get_eval_loader("standard", test_dataset, batch_size=2)
    all_y_pred, all_y_true = [], []
    pred_tensor = [[30, 70, 35, 75]]
    for _, x, y_true in test_loader:
        labels = torch.zeros(x.shape[0])
        scores = torch.stack(
            [torch.tensor(0.54) for _ in range(len(pred_tensor))])
        y_pred = [{
            "y": torch.tensor(pred_tensor),
            "labels": labels,
            "scores": scores,
        } for _ in range(x.shape[0])]
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    viz_root = tmp_path / "viz"
    out, _ = ds.eval(
        y_pred=all_y_pred,
        y_true=all_y_true,
        metadata=test_dataset.metadata_array,
        viz_dir=str(viz_root),
        viz_n_per_source=2,
    )
    paths = out["eval_visualization_paths"]
    assert len(paths) >= 1
    for p in paths:
        assert Path(p).is_file()


def test_maskaware_precision_ignores_unmatched_tree_pixels():
    metric = MaskAwareDetectionPrecision(geometry_name="y",
                                         iou_threshold=0.5,
                                         tree_fraction_threshold=0.5)
    gt = [{'y': torch.tensor([[10., 10., 20., 20.]])}]
    pred = [{
        'y': torch.tensor([[10., 10., 20., 20.], [30., 30., 40., 40.]]),
        'scores': torch.tensor([0.9, 0.9]),
    }]
    tree_mask = torch.zeros((64, 64), dtype=torch.uint8)
    tree_mask[30:40, 30:40] = 1
    gt[0]["tree_coverage_mask"] = tree_mask

    score = metric.compute(pred, gt)[metric.agg_metric_field]
    assert score == pytest.approx(1.0)


def test_maskaware_precision_counts_background_unmatched():
    metric = MaskAwareDetectionPrecision(geometry_name="y",
                                         iou_threshold=0.5,
                                         tree_fraction_threshold=0.5)
    gt = [{'y': torch.tensor([[10., 10., 20., 20.]])}]
    pred = [{
        'y': torch.tensor([[10., 10., 20., 20.], [30., 30., 40., 40.]]),
        'scores': torch.tensor([0.9, 0.9]),
    }]
    gt[0]["tree_coverage_mask"] = torch.zeros((64, 64), dtype=torch.uint8)

    score = metric.compute(pred, gt)[metric.agg_metric_field]
    assert score == pytest.approx(0.5)


def test_maskaware_precision_keeps_duplicate_predictions_as_false_positives():
    metric = MaskAwareDetectionPrecision(geometry_name="y",
                                         iou_threshold=0.5,
                                         tree_fraction_threshold=0.5)
    gt = [{'y': torch.tensor([[10., 10., 20., 20.]])}]
    pred = [{
        'y': torch.tensor([[10., 10., 20., 20.], [10., 10., 20., 20.]]),
        'scores': torch.tensor([0.9, 0.8]),
    }]
    gt[0]["tree_coverage_mask"] = torch.ones((64, 64), dtype=torch.uint8)

    score = metric.compute(pred, gt)[metric.agg_metric_field]
    assert score == pytest.approx(0.5)


def test_maskaware_precision_falls_back_without_tree_mask():
    metric = MaskAwareDetectionPrecision(geometry_name="y",
                                         iou_threshold=0.5,
                                         tree_fraction_threshold=0.5)
    gt = [{'y': torch.tensor([[10., 10., 20., 20.]])}]
    pred = [{
        'y': torch.tensor([[10., 10., 20., 20.], [30., 30., 40., 40.]]),
        'scores': torch.tensor([0.9, 0.9]),
    }]

    score = metric.compute(pred, gt)[metric.agg_metric_field]
    assert score == pytest.approx(0.5)


def test_TreeBoxes_download_url(dataset):
    ds = TreeBoxesDataset(download=False, root_dir=dataset, version="0.0")
    for version in ds._versions_dict.keys():
        print(version)
        # Confirm url can be downloaded
        url = ds._versions_dict[version]['download_url']
        # If the url is not accessible, skip the test
        if url == "":
            continue
        response = requests.head(url, allow_redirects=True)
        assert response.status_code == 200, f"URL {url} is not accessible"

