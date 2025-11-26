"""
Ultralytics YOLO example for MillionTrees TreeBoxes.

This script:
- Loads the TreeBoxes test split
- Runs YOLO (COCO-pretrained) inference
- Converts predictions to MillionTrees format
- Computes evaluation metrics

Usage:
  python docs/examples/yolo_treeboxes.py \
    --root_dir data \
    --version 0.2 \
    --batch_size 8 \
    --device auto \
    --weights yolov8n.pt

Note: Requires `pip install ultralytics`.
"""

import argparse
from typing import List, Dict, Any

import torch

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultralytics YOLO on MillionTrees TreeBoxes")
    parser.add_argument("--root_dir", type=str, default="data", help="Dataset root directory")
    parser.add_argument("--version", type=str, default="0.2", help="Dataset version (e.g., 0.2)")
    parser.add_argument("--batch_size", type=int, default=8, help="Eval batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--mini", action="store_true", help="Use mini datasets for fast dev")
    parser.add_argument("--download", action="store_true", help="Download dataset if missing")
    parser.add_argument("--split_scheme",
                        type=str,
                        default="random",
                        choices=["random", "zeroshot", "crossgeometry"],
                        help="Dataset split scheme")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for YOLO (auto/cpu/cuda)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics YOLO weights (e.g., yolov8n.pt)",
    )
    parser.add_argument("--score_threshold", type=float, default=0.05, help="Score threshold")
    return parser.parse_args()


def select_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def to_uint8_hwc(images: torch.Tensor) -> List[Any]:
    # Convert BCHW float [0,1] -> HWC uint8 per image
    imgs: List[Any] = []
    for img in images:
        hwc = (img.permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        imgs.append(hwc)
    return imgs


def main() -> None:
    args = parse_args()

    # Lazy import to avoid hard dependency at package level
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Ultralytics is required. Install with `pip install ultralytics`."
        ) from exc

    # Dataset and loader
    dataset = get_dataset("TreeBoxes",
                          version=args.version,
                          root_dir=args.root_dir,
                          download=args.download,
                          mini=args.mini,
                          split_scheme=args.split_scheme)
    test_dataset = dataset.get_subset("test")
    test_loader = get_eval_loader(
        "standard", test_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # YOLO model
    device = select_device(args.device)
    model = YOLO(args.weights)

    all_y_pred: List[Dict[str, Any]] = []
    all_y_true: List[Dict[str, Any]] = []

    for batch in test_loader:
        metadata, images, targets = batch
        # Prepare inputs
        imgs_hwc_uint8 = to_uint8_hwc(images)

        # Predict
        results = model.predict(
            imgs_hwc_uint8,
            device=device,
            verbose=False,
            conf=args.score_threshold,
        )

        # Convert predictions to MillionTrees format
        for res, target in zip(results, targets):
            if res.boxes is None or len(res.boxes) == 0:  # no detections
                y_pred = {
                    "y": torch.empty((0, 4), dtype=torch.float32),
                    "labels": torch.empty((0,), dtype=torch.int64),
                    "scores": torch.empty((0,), dtype=torch.float32),
                }
            else:
                boxes_xyxy = res.boxes.xyxy.detach().to("cpu").to(torch.float32)
                scores = res.boxes.conf.detach().to("cpu").to(torch.float32)
                labels = res.boxes.cls.detach().to("cpu").to(torch.int64)
                y_pred = {"y": boxes_xyxy, "labels": labels, "scores": scores}

            all_y_pred.append(y_pred)
            all_y_true.append(target)

    # Evaluate
    results, results_str = dataset.eval(all_y_pred, all_y_true, metadata=test_dataset.metadata_array)
    print("\nEvaluation Results (dict):")
    print(results)
    print("\nFormatted Results:")
    print(results_str)


if __name__ == "__main__":
    main()


