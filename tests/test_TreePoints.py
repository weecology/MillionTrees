from milliontrees.datasets.TreePoints import TreePointsDataset
from milliontrees.common.data_loaders import get_train_loader, get_eval_loader
from milliontrees.common.metrics.all_metrics import (
    CountingError,
    MaskAwareKeypointPrecision,
)

import torch
import pytest
import numpy as np
import requests
import os

# Test structure without real annotation data to ensure format is correct
@pytest.mark.parametrize("split_scheme", ["within-distribution", "out-of-distribution", "crossgeometry"])
def test_TreePoints_small(dataset, split_scheme):
    ds = TreePointsDataset(
        download=False,
        root_dir=dataset,
        version="0.0",
        small=True,
        split_scheme=split_scheme,
    )
    assert str(ds._data_dir).endswith("SmallTreePoints_v0.0")
    train_dataset = ds.get_subset("train")
    assert len(train_dataset) > 0 or len(ds.get_subset("test")) > 0


def test_TreePoints_generic(dataset):
    ds = TreePointsDataset(download=False, root_dir=dataset, version="0.0") 
    for metadata, image, targets in ds:
        points, labels = targets["y"], targets["labels"]
        assert image.shape == (100, 100, 3)
        assert image.dtype == np.float32
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert points.shape == (2, 2)
        assert labels.shape == (2,)
        assert metadata.shape == (2,)
        break

    train_dataset = ds.get_subset("train")
     
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
    ds = TreePointsDataset(download=False, root_dir=dataset, version="0.0") 
    train_dataset = ds.get_subset("train")
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
    ds = TreePointsDataset(download=False, root_dir=dataset, version="0.0") 
    test_dataset = ds.get_subset("test")
    
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
    assert metadata[1] == 0
    assert ds._filename_id_to_code[int(metadata[0])] == "image3.jpg"

    test_loader = get_eval_loader('standard', test_dataset, batch_size=1)
    for metadata, x, targets in test_loader:
        y = targets[0]["y"]
        assert torch.is_tensor(targets[0]["y"])
        assert x.shape == (1, 3, 448, 448)
        assert x.dtype == torch.float32
        assert x.min() >= 0.0 and x.max() <= 1.0
        assert y.shape[1] == 2
        assert len(metadata) == 1
        break

@pytest.mark.parametrize("pred_tensor", [[[134, 156]], [[30, 70],[35, 55]]], ids=["single", "multiple"])
def test_TreePoints_eval(dataset, pred_tensor):
    ds = TreePointsDataset(download=False, root_dir=dataset, version="0.0") 
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

    # Evaluate
    eval_results, eval_string = ds.eval(all_y_pred, all_y_true, test_dataset.metadata_array)

def test_maskaware_keypoint_precision_ignores_unmatched_tree_pixels():
    metric = MaskAwareKeypointPrecision(geometry_name="y",
                                        distance_threshold=0.02,
                                        image_size=448,
                                        tree_fraction_threshold=0.5)
    gt = [{'y': torch.tensor([[10., 10.]])}]
    pred = [{
        'y': torch.tensor([[10., 10.], [30., 30.]]),
        'scores': torch.tensor([0.9, 0.9]),
    }]
    tree_mask = torch.zeros((64, 64), dtype=torch.uint8)
    tree_mask[30, 30] = 1
    gt[0]["tree_coverage_mask"] = tree_mask

    score = metric.compute(pred, gt)[metric.agg_metric_field]
    assert score == pytest.approx(1.0)


def test_maskaware_keypoint_precision_counts_background_unmatched():
    metric = MaskAwareKeypointPrecision(geometry_name="y",
                                        distance_threshold=0.02,
                                        image_size=448,
                                        tree_fraction_threshold=0.5)
    gt = [{'y': torch.tensor([[10., 10.]])}]
    pred = [{
        'y': torch.tensor([[10., 10.], [30., 30.]]),
        'scores': torch.tensor([0.9, 0.9]),
    }]
    gt[0]["tree_coverage_mask"] = torch.zeros((64, 64), dtype=torch.uint8)

    score = metric.compute(pred, gt)[metric.agg_metric_field]
    assert score == pytest.approx(0.5)


def test_maskaware_keypoint_precision_keeps_duplicate_predictions_as_false_positives():
    metric = MaskAwareKeypointPrecision(geometry_name="y",
                                        distance_threshold=0.02,
                                        image_size=448,
                                        tree_fraction_threshold=0.5)
    gt = [{'y': torch.tensor([[10., 10.]])}]
    pred = [{
        'y': torch.tensor([[10., 10.], [10., 10.]]),
        'scores': torch.tensor([0.9, 0.8]),
    }]
    gt[0]["tree_coverage_mask"] = torch.ones((64, 64), dtype=torch.uint8)

    score = metric.compute(pred, gt)[metric.agg_metric_field]
    assert score == pytest.approx(0.5)


def test_maskaware_keypoint_precision_falls_back_without_tree_mask():
    metric = MaskAwareKeypointPrecision(geometry_name="y",
                                        distance_threshold=0.02,
                                        image_size=448,
                                        tree_fraction_threshold=0.5)
    gt = [{'y': torch.tensor([[10., 10.]])}]
    pred = [{
        'y': torch.tensor([[10., 10.], [30., 30.]]),
        'scores': torch.tensor([0.9, 0.9]),
    }]

    score = metric.compute(pred, gt)[metric.agg_metric_field]
    assert score == pytest.approx(0.5)

def test_counting_error_points_gated_by_complete_flag():
    metric = CountingError(score_threshold=0.1, geometry_name="y")
    gt_points = torch.tensor([[10., 10.], [12., 12.], [80., 80.]])
    pred_points = torch.tensor([[10., 10.], [12., 12.], [80., 80.], [70., 70.]])
    scores = torch.tensor([0.9, 0.9, 0.9, 0.9])

    # complete=False -> excluded from aggregate (NaN), so result is NaN.
    gt_excluded = [{"y": gt_points, "complete": False}]
    pred = [{"y": pred_points, "scores": scores}]
    assert np.isnan(metric.compute(pred, gt_excluded)[metric.agg_metric_field])

    # complete=True -> |3 - 4| = 1.
    gt_included = [{"y": gt_points, "complete": True}]
    score = metric.compute(pred, gt_included)[metric.agg_metric_field]
    assert score == pytest.approx(1.0)


def test_TreePoints_download_url(dataset):
    ds = TreePointsDataset(download=False, root_dir=dataset, version="0.0")
    for version in ds._versions_dict.keys():
        print(version)
        # Confirm url can be downloaded
        url = ds._versions_dict[version]['download_url']
        # If the url is not accessible, skip the test
        if url == "":
            continue
        response = requests.head(url, allow_redirects=True)
        assert response.status_code == 200, f"URL {url} is not accessible"
