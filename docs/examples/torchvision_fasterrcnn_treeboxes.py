"""
Torchvision Faster R-CNN example for MillionTrees TreeBoxes.

This script:
- Loads the TreeBoxes dataset test split
- Runs an off-the-shelf Faster R-CNN (COCO-pretrained) model
- Formats predictions to MillionTrees eval format
- Computes and prints evaluation metrics

Usage:
    python docs/examples/torchvision_fasterrcnn_treeboxes.py \
        --root_dir data \
        --version 0.2 \
        --batch_size 8 \
        --device auto

Notes:
- No training is performed; this is a pure zero-shot baseline.
- Model labels are not used by the evaluation; only boxes and scores are required.
"""

import argparse
import os
from typing import List, Dict, Any

import torch
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Torchvision Faster R-CNN on MillionTrees TreeBoxes")
    parser.add_argument("--root_dir", type=str, default="data", help="Dataset root directory")
    parser.add_argument("--version", type=str, default="0.2", help="Dataset version (e.g., 0.2)")
    parser.add_argument("--batch_size", type=int, default=8, help="Eval batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--mini", action="store_true", help="Use mini datasets for fast dev")
    parser.add_argument("--download", action="store_true", help="Download dataset if missing")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run inference on",
    )
    parser.add_argument("--score_threshold", type=float, default=0.05, help="Score threshold for predictions")
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    score_threshold: float,
) -> List[Dict[str, Any]]:
    # Torchvision detection models expect a list[Tensor(C,H,W)]
    image_list = [img.to(device) for img in images]
    outputs = model(image_list)

    formatted: List[Dict[str, Any]] = []
    for out in outputs:
        boxes = out.get("boxes", torch.empty((0, 4), device=device))
        scores = out.get("scores", torch.empty((0,), device=device))
        labels = out.get("labels", torch.empty((0,), device=device))

        # Apply score threshold
        keep = scores >= score_threshold
        boxes = boxes[keep].detach().to("cpu").to(torch.float32)
        scores = scores[keep].detach().to("cpu").to(torch.float32)
        labels = labels[keep].detach().to("cpu")

        y_pred = {
            "y": boxes,
            "labels": labels,
            "scores": scores,
        }
        formatted.append(y_pred)
    return formatted


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    # Load dataset and test split
    dataset = get_dataset("TreeBoxes",
                          version=args.version,
                          root_dir=args.root_dir,
                          download=args.download,
                          mini=args.mini)
    test_dataset = dataset.get_subset("test")
    test_loader = get_eval_loader(
        "standard", test_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Model: COCO-pretrained Faster R-CNN
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT").to(device).eval()

    all_y_pred: List[Dict[str, Any]] = []
    all_y_true: List[Dict[str, Any]] = []

    for batch in test_loader:
        metadata, images, targets = batch
        batch_preds = run_inference(model, images, device, score_threshold=args.score_threshold)

        # Accumulate per-image predictions and ground truth in MillionTrees format
        for pred, target in zip(batch_preds, targets):
            all_y_pred.append(pred)
            all_y_true.append(target)

    # Evaluate. Use the test subset metadata array (aligned with dataset order)
    results, results_str = dataset.eval(all_y_pred, all_y_true, metadata=test_dataset.metadata_array)

    print("\nEvaluation Results (dict):")
    print(results)
    print("\nFormatted Results:")
    print(results_str)


if __name__ == "__main__":
    main()


