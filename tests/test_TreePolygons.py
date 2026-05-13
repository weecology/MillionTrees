from milliontrees.datasets.TreePolygons import TreePolygonsDataset
from milliontrees.datasets.polygon_stream_eval import TreePolygonsStreamingEvalState
from milliontrees.common.data_loaders import get_train_loader, get_eval_loader
from milliontrees.common.metrics.all_metrics import MaskAwareMaskPrecision

import math
import warnings

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

    # Use the first test image's own GT masks as predictions (guaranteed IoU = 1).
    _, _, ref_tgt = test_dataset[0]
    ref_masks = torch.as_tensor(ref_tgt["y"]).clone()
    ref_boxes = torch.as_tensor(ref_tgt["bboxes"]).clone()
    ref_labels = torch.as_tensor(ref_tgt["labels"]).clone()

    for metadata, x, y_true in test_loader:
        batch = [{
            'y': ref_masks,
            'bboxes': ref_boxes,
            'labels': ref_labels,
            'scores': torch.tensor([0.54] * len(ref_labels)),
        }]
        all_y_pred.extend(batch)
        all_y_true.extend(y_true)

    # Evaluate
    eval_results, eval_string = ds.eval(y_pred=all_y_pred,y_true=all_y_true, metadata=test_dataset.metadata_array)
    
    # One test image: prediction matches the lone GT mask → perfect accuracy.
    assert eval_results["accuracy"]["mask_acc_avg"] == pytest.approx(1.0)
    assert len(eval_results) 
    assert "accuracy" in eval_results.keys()
    assert "recall" in eval_results.keys()
    assert "maskaware_precision" in eval_results.keys()
    assert "merge_commission" in eval_results.keys()


def test_TreePolygons_eval_stream_matches_legacy(dataset):
    """Streaming eval must match legacy ``dataset.eval`` on the same predictions."""
    ds = TreePolygonsDataset(download=False, root_dir=dataset, version="0.0")
    test_dataset = ds.get_subset("test")
    test_loader = get_eval_loader("standard", test_dataset, batch_size=2)

    _, _, ref_tgt = test_dataset[0]
    ref_masks = torch.as_tensor(ref_tgt["y"]).clone()
    ref_boxes = torch.as_tensor(ref_tgt["bboxes"]).clone()
    ref_labels = torch.as_tensor(ref_tgt["labels"]).clone()

    all_y_pred = []
    all_y_true = []
    state = TreePolygonsStreamingEvalState(ds)

    for metadata, x, y_true in test_loader:
        batch = [{
            "y": ref_masks,
            "bboxes": ref_boxes,
            "labels": ref_labels,
            "scores": torch.tensor([0.54] * len(ref_labels)),
        }]
        all_y_pred.extend(batch)
        all_y_true.extend(y_true)
        state.update(batch, y_true, metadata)

    legacy_results, _ = ds.eval(
        y_pred=all_y_pred,
        y_true=all_y_true,
        metadata=test_dataset.metadata_array,
    )
    stream_results, _ = state.finalize()

    for metric_name in ("accuracy", "recall", "maskaware_precision", "merge_commission", "mAP"):
        lk = legacy_results[metric_name]
        sk = stream_results[metric_name]
        for key, lv in lk.items():
            if key == "eval_visualization_paths":
                continue
            sv = sk[key]
            fv = float(lv)
            fs = float(sv)
            if math.isnan(fv):
                assert math.isnan(fs), f"{metric_name}.{key} legacy=nan stream={sv}"
            else:
                assert fs == pytest.approx(fv, rel=1e-5, abs=1e-5), (
                    f"{metric_name}.{key} legacy={lv} stream={sv}"
                )
    lr_dom = float(legacy_results["detection_acc_avg_dom"])
    sr_dom = float(stream_results["detection_acc_avg_dom"])
    if math.isnan(lr_dom):
        assert math.isnan(sr_dom)
    else:
        assert sr_dom == pytest.approx(lr_dom, rel=1e-5, abs=1e-5)


def test_TreePolygons_map_allows_dense_predictions(dataset):
    ds = TreePolygonsDataset(download=False, root_dir=dataset, version="0.0")
    metric = ds.metrics["mAP"]
    assert metric.max_detection_thresholds == [1, 10, 1000]

    state = TreePolygonsStreamingEvalState(ds)
    assert state._map_global.max_detection_thresholds == [1, 10, 1000]
    assert all(m.max_detection_thresholds == [1, 10, 1000]
               for m in state._map_per_group)

    masks = torch.zeros((101, 2, 2), dtype=torch.bool)
    masks[:, 0, 0] = True
    y_pred = [{
        "y": masks,
        "scores": torch.linspace(1.0, 0.1, 101),
    }]
    y_true = [{"y": masks[:1]}]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metric.compute(y_pred, y_true, return_dict=False)

    assert not any("Encountered more than 100 detections" in str(w.message)
                   for w in caught)


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

def test_TreePolygons_empty_masks(dataset):
    """Test that images with no polygons are handled correctly."""
    ds = TreePolygonsDataset(download=False, root_dir=dataset, version="0.0")
    
    # Find an image that might have no polygons, or create a mock scenario
    # For now, we'll test that the dataset can handle empty masks gracefully
    # by checking the structure when masks might be empty after filtering
    
    # Test that empty masks create correct tensor shapes
    test_dataset = ds.get_subset("test")
    test_loader = get_eval_loader('standard', test_dataset, batch_size=1)
    
    # Iterate through and verify structure is correct even with potential empty cases
    for metadata, x, targets in test_loader:
        masks = targets[0]["y"]
        boxes = targets[0]["bboxes"]
        labels = targets[0]["labels"]
        
        # Masks should be a tensor, even if empty
        assert isinstance(masks, torch.Tensor)
        assert masks.dim() == 3  # [N, H, W] where N can be 0
        assert masks.shape[1] == 448  # Height
        assert masks.shape[2] == 448  # Width
        
        # Boxes should be a tensor or numpy array, even if empty
        # (albumentations may return numpy arrays)
        assert isinstance(boxes, (torch.Tensor, np.ndarray))
        if isinstance(boxes, np.ndarray):
            boxes = torch.from_numpy(boxes)
        assert boxes.dim() == 2  # [N, 4] where N can be 0
        if len(boxes) > 0:
            assert boxes.shape[1] == 4
        
        # Labels should match number of masks
        assert isinstance(labels, torch.Tensor)
        assert len(labels) == len(masks)
        
        # Only check first batch
        break

def test_TreePolygons_empty_masks_eval(dataset):
    """Test evaluation with empty predictions and empty ground truth."""
    ds = TreePolygonsDataset(download=False, root_dir=dataset, version="0.0")
    test_dataset = ds.get_subset("test")
    
    # Test case 1: Empty predictions, non-empty ground truth
    y_pred_empty = [{
        'y': torch.zeros((0, 448, 448), dtype=torch.uint8),
        'labels': torch.zeros((0,), dtype=torch.int64),
        'scores': torch.zeros((0,), dtype=torch.float32),
    }]
    
    # Get one real ground truth
    test_loader = get_eval_loader('standard', test_dataset, batch_size=1)
    for metadata, x, y_true in test_loader:
        # Evaluate with empty predictions
        results, results_str = ds.eval(
            y_pred=y_pred_empty,
            y_true=y_true,
            metadata=metadata[:1]
        )
        
        # Should complete without error
        assert "accuracy" in results.keys()
        assert "recall" in results.keys()
        break
    
    # Test case 2: Non-empty predictions, empty ground truth
    polygon = from_wkt("POLYGON((10 15, 50 15, 50 55, 10 55, 10 15))")
    pred_mask = ds.create_polygon_mask(width=448, height=448, vertices=polygon)
    y_pred = [{
        'y': torch.tensor([pred_mask], dtype=torch.uint8),
        'labels': torch.tensor([0], dtype=torch.int64),
        'scores': torch.tensor([0.8], dtype=torch.float32),
    }]
    
    y_true_empty = [{
        'y': torch.zeros((0, 448, 448), dtype=torch.uint8),
        'bboxes': torch.zeros((0, 4), dtype=torch.float32),
        'labels': torch.zeros((0,), dtype=torch.int64),
    }]
    
    # Should handle empty ground truth gracefully
    results, results_str = ds.eval(
        y_pred=y_pred,
        y_true=y_true_empty,
        metadata=torch.tensor([[0, 0]], dtype=torch.int64)
    )
    assert "accuracy" in results.keys()
    assert "recall" in results.keys()

def test_TreePolygons_box_to_mask_conversion(dataset):
    """Test that bounding boxes can be converted to masks for evaluation."""
    from milliontrees.common.metrics.all_metrics import MaskAccuracy
    
    ds = TreePolygonsDataset(download=False, root_dir=dataset, version="0.0")
    metric = MaskAccuracy(geometry_name='y', iou_threshold=0.5, score_threshold=0.1)
    
    # Create ground truth masks
    polygon = from_wkt("POLYGON((10 15, 50 15, 50 55, 10 55, 10 15))")
    gt_mask = ds.create_polygon_mask(width=448, height=448, vertices=polygon)
    gt_masks = torch.tensor([gt_mask], dtype=torch.uint8)
    
    # Create predictions as bounding boxes (simulating DeepForest output)
    pred_boxes = torch.tensor([[10, 15, 50, 55]], dtype=torch.float32)
    
    # Test that _mask_iou can handle boxes
    iou = metric._mask_iou(gt_masks, pred_boxes)
    
    # Should return IoU matrix
    assert iou.shape == (1, 1)  # [N_gt, N_pred]
    assert iou.dtype == torch.float32
    # IoU should be > 0 since boxes overlap with mask
    assert iou[0, 0] > 0.0
    
    # Test with empty boxes
    empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
    # Empty boxes should return zero IoU matrix
    iou_empty = metric._mask_iou(gt_masks, empty_boxes)
    assert iou_empty.shape == (1, 0)
    
    # Test with empty masks and boxes
    empty_masks = torch.zeros((0, 448, 448), dtype=torch.uint8)
    iou_both_empty = metric._mask_iou(empty_masks, empty_boxes)
    assert iou_both_empty.shape == (0, 0)


def test_maskaware_mask_precision_ignores_unmatched_tree_pixels():
    metric = MaskAwareMaskPrecision(geometry_name="y",
                                    iou_threshold=0.5,
                                    tree_fraction_threshold=0.5)
    gt_mask = torch.zeros((64, 64), dtype=torch.uint8)
    gt_mask[10:20, 10:20] = 1
    unmatched_mask = torch.zeros((64, 64), dtype=torch.uint8)
    unmatched_mask[30:40, 30:40] = 1
    gt = [{'y': torch.stack([gt_mask]), 'tree_coverage_mask': unmatched_mask.clone()}]
    pred = [{
        'y': torch.stack([gt_mask, unmatched_mask]),
        'scores': torch.tensor([0.9, 0.9]),
    }]

    score = metric.compute(pred, gt)[metric.agg_metric_field]
    assert score == pytest.approx(1.0)


def test_maskaware_mask_precision_counts_background_unmatched():
    metric = MaskAwareMaskPrecision(geometry_name="y",
                                    iou_threshold=0.5,
                                    tree_fraction_threshold=0.5)
    gt_mask = torch.zeros((64, 64), dtype=torch.uint8)
    gt_mask[10:20, 10:20] = 1
    unmatched_mask = torch.zeros((64, 64), dtype=torch.uint8)
    unmatched_mask[30:40, 30:40] = 1
    gt = [{'y': torch.stack([gt_mask]), 'tree_coverage_mask': torch.zeros((64, 64), dtype=torch.uint8)}]
    pred = [{
        'y': torch.stack([gt_mask, unmatched_mask]),
        'scores': torch.tensor([0.9, 0.9]),
    }]

    score = metric.compute(pred, gt)[metric.agg_metric_field]
    assert score == pytest.approx(0.5)


def test_maskaware_mask_precision_keeps_duplicate_predictions_as_false_positives():
    metric = MaskAwareMaskPrecision(geometry_name="y",
                                    iou_threshold=0.5,
                                    tree_fraction_threshold=0.5)
    gt_mask = torch.zeros((64, 64), dtype=torch.uint8)
    gt_mask[10:20, 10:20] = 1
    gt = [{'y': torch.stack([gt_mask]), 'tree_coverage_mask': torch.ones((64, 64), dtype=torch.uint8)}]
    pred = [{
        'y': torch.stack([gt_mask, gt_mask]),
        'scores': torch.tensor([0.9, 0.8]),
    }]

    score = metric.compute(pred, gt)[metric.agg_metric_field]
    assert score == pytest.approx(0.5)


def test_maskaware_mask_precision_falls_back_without_tree_mask():
    metric = MaskAwareMaskPrecision(geometry_name="y",
                                    iou_threshold=0.5,
                                    tree_fraction_threshold=0.5)
    gt_mask = torch.zeros((64, 64), dtype=torch.uint8)
    gt_mask[10:20, 10:20] = 1
    unmatched_mask = torch.zeros((64, 64), dtype=torch.uint8)
    unmatched_mask[30:40, 30:40] = 1
    gt = [{'y': torch.stack([gt_mask])}]
    pred = [{
        'y': torch.stack([gt_mask, unmatched_mask]),
        'scores': torch.tensor([0.9, 0.9]),
    }]

    score = metric.compute(pred, gt)[metric.agg_metric_field]
    assert score == pytest.approx(0.5)