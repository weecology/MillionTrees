"""Adapter example for external segmentation models (e.g., DeepTrees).

This script demonstrates how to convert external model outputs to the
MillionTrees evaluation format for TreePolygons:

    {
        "y": Tensor[N, H, W] uint8,
        "labels": Tensor[N] int64,
        "scores": Tensor[N] float32
    }

Use --mock to run a complete end-to-end evaluation without any external model.
To integrate a real model, implement `run_external_model_batch`.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader


def _to_mask_tensor(masks: Any) -> torch.Tensor:
    """Normalize masks to Tensor[N, H, W] uint8."""
    if isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks)
    elif isinstance(masks, list):
        if len(masks) == 0:
            return torch.zeros((0, 1, 1), dtype=torch.uint8)
        masks = torch.stack([
            torch.from_numpy(m) if isinstance(m, np.ndarray) else m for m in masks
        ])
    elif not isinstance(masks, torch.Tensor):
        raise TypeError("masks must be a list/ndarray/tensor")

    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with 3 dims [N,H,W], got {masks.shape}")

    return (masks > 0).to(torch.uint8)


def _resize_masks_if_needed(masks: torch.Tensor, target_h: int,
                            target_w: int) -> torch.Tensor:
    """Resize masks to image shape when model output resolution differs."""
    if masks.shape[-2:] == (target_h, target_w):
        return masks
    masks = masks.to(torch.float32).unsqueeze(1)
    masks = F.interpolate(masks, size=(target_h, target_w), mode="nearest")
    return masks.squeeze(1).to(torch.uint8)


def adapt_segmentation_prediction(raw_pred: dict[str, Any], image_h: int,
                                  image_w: int) -> dict[str, torch.Tensor]:
    """Convert one external segmentation prediction to MillionTrees format."""
    masks = _to_mask_tensor(raw_pred.get("masks", []))
    if masks.numel() == 0:
        return {
            "y": torch.zeros((0, image_h, image_w), dtype=torch.uint8),
            "labels": torch.zeros((0, ), dtype=torch.int64),
            "scores": torch.zeros((0, ), dtype=torch.float32),
        }

    masks = _resize_masks_if_needed(masks, image_h, image_w)
    n = masks.shape[0]

    scores = raw_pred.get("scores")
    if scores is None:
        scores_tensor = torch.ones((n, ), dtype=torch.float32)
    else:
        scores_tensor = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
        if scores_tensor.numel() != n:
            raise ValueError(
                f"scores length ({scores_tensor.numel()}) != number of masks ({n})")

    labels = raw_pred.get("labels")
    if labels is None:
        labels_tensor = torch.zeros((n, ), dtype=torch.int64)
    else:
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64).reshape(-1)
        if labels_tensor.numel() != n:
            raise ValueError(
                f"labels length ({labels_tensor.numel()}) != number of masks ({n})")

    return {
        "y": masks.to(torch.uint8),
        "labels": labels_tensor,
        "scores": scores_tensor,
    }


def run_external_model_batch(images: torch.Tensor) -> list[dict[str, Any]]:
    """Run your external model and return one dict per image.

    Expected output for each image:
      - masks: list/array/tensor with shape [N, H, W] or [H, W] for one instance
      - scores: optional list/array length N
      - labels: optional list/array length N

    Replace this stub with your model call, for example with DeepTrees:
    https://deeptrees.de/
    """
    raise NotImplementedError(
        "Implement run_external_model_batch(images) for your external model.")


def run_mock_model_batch(targets: list[dict[str, torch.Tensor]]
                         ) -> list[dict[str, Any]]:
    """Deterministic mock model for adapter smoke tests."""
    mock_preds: list[dict[str, Any]] = []
    for target in targets:
        gt_masks = target["y"]
        if isinstance(gt_masks, np.ndarray):
            gt_masks = torch.from_numpy(gt_masks)
        gt_masks = gt_masks.to(torch.uint8)

        if gt_masks.ndim == 2:
            gt_masks = gt_masks.unsqueeze(0)
        n = gt_masks.shape[0]

        if n == 0:
            mock_preds.append({"masks": torch.zeros((0, 1, 1), dtype=torch.uint8)})
            continue

        keep_n = max(1, n // 2)
        pred_masks = gt_masks[:keep_n]
        scores = torch.linspace(0.9, 0.6, keep_n).tolist()
        labels = torch.zeros((keep_n, ), dtype=torch.int64).tolist()
        mock_preds.append({
            "masks": pred_masks,
            "scores": scores,
            "labels": labels
        })
    return mock_preds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adapter example for external segmentation model outputs.")
    parser.add_argument("--root-dir", type=str, default="onboarding_data")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--split-scheme",
                        type=str,
                        default="random",
                        choices=["random", "zeroshot", "crossgeometry"])
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run with mock predictions instead of an external model.",
    )
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    dataset = get_dataset(
        "TreePolygons",
        root_dir=args.root_dir,
        mini=args.mini,
        download=args.download,
        split_scheme=args.split_scheme,
    )
    test_subset = dataset.get_subset("test")
    test_loader = get_eval_loader("standard",
                                  test_subset,
                                  batch_size=args.batch_size)

    all_y_pred: list[dict[str, torch.Tensor]] = []
    all_y_true: list[dict[str, torch.Tensor]] = []

    for batch_idx, (_, images, targets) in enumerate(test_loader):
        if args.mock:
            raw_preds = run_mock_model_batch(targets)
        else:
            raw_preds = run_external_model_batch(images)

        if len(raw_preds) != len(targets):
            raise ValueError(
                f"Model returned {len(raw_preds)} predictions for {len(targets)} images.")

        for raw_pred, target in zip(raw_preds, targets):
            _, image_h, image_w = images.shape[1:]
            mt_pred = adapt_segmentation_prediction(raw_pred, image_h, image_w)
            all_y_pred.append(mt_pred)
            all_y_true.append(target)

        if args.verbose:
            print(f"Processed batch {batch_idx + 1}/{len(test_loader)}")

        if args.max_batches is not None and (batch_idx + 1) >= args.max_batches:
            break

    used_metadata = test_subset.metadata_array[:len(all_y_true)]
    results, results_str = dataset.eval(all_y_pred, all_y_true, metadata=used_metadata)
    print(results_str)

    if args.verbose:
        print("Metric keys:", list(results.keys()))


if __name__ == "__main__":
    main()
