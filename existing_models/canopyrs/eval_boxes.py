"""Evaluate the CanopyRS DINO Swin-L detector on MillionTrees TreeBoxes.

Uses the multi-resolution, multi-dataset detector released with the SelvaBox paper
(https://huggingface.co/CanopyRS/dino-swin-l-384-multi-NQOS). The model is loaded
through CanopyRS' own detrex/Detectron2 wrapper; we feed it the MillionTrees test
images batch by batch and convert its boxes to the MillionTrees evaluation format.

CanopyRS (with detrex + Detectron2) must be installed in the environment; see
README.md for setup.
"""

import argparse
import json
import os
from typing import List

import numpy as np
import torch

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader
from milliontrees.common.eval_sweep import (add_sweep_args, maybe_run_sweep,
                                            maybe_subsample)

DETECTOR_CONFIG = "detectors/dino_swinL_multi_NQOS.yaml"
MODEL_NAME = "CanopyRS-DINO-SwinL"


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_detector(config_name: str):
    """Instantiate the CanopyRS detrex detector wrapper from a bundled config."""
    from canopyrs.engine.config_parsers import DetectorConfig
    from canopyrs.engine.config_parsers.base import get_config_path
    from canopyrs.engine.models.detector.detectron2_infer import Detectron2DetectorWrapper

    config = DetectorConfig.from_yaml(get_config_path(config_name))
    config.batch_size = 1
    return Detectron2DetectorWrapper(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="CanopyRS DINO on TreeBoxes.")
    parser.add_argument("--root-dir", type=str,
                        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--split-scheme", type=str, default="within-distribution",
                        choices=["within-distribution", "out-of-distribution", "crossgeometry"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--viz-dir", type=str, default=None,
                        help="Directory for per-source prediction overlay PNGs")
    add_sweep_args(parser)
    args = parser.parse_args()

    device = select_device(args.device)
    detector = load_detector(DETECTOR_CONFIG)

    dataset = get_dataset("TreeBoxes", root_dir=args.root_dir, download=args.download,
                          mini=args.mini, split_scheme=args.split_scheme)
    test_subset = maybe_subsample(dataset, dataset.get_subset("test"), args)
    test_loader = get_eval_loader("standard", test_subset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    print(f"Batches: {len(test_loader)}")

    all_y_pred, all_y_true = [], []
    for b_idx, (metadata, images, targets) in enumerate(test_loader):
        image_list: List[torch.Tensor] = [img.to(device) for img in images]
        preds = detector.forward(image_list)

        for pred, target in zip(preds, targets):
            boxes = pred["boxes"].float().cpu()
            scores = pred["scores"].float().cpu()
            keep = scores >= args.score_threshold
            boxes, scores = boxes[keep], scores[keep]
            all_y_pred.append({
                "y": boxes,
                "labels": torch.zeros(boxes.shape[0], dtype=torch.int64),
                "scores": scores,
            })
            all_y_true.append(target)

        if args.max_batches is not None and (b_idx + 1) >= args.max_batches:
            break

    if maybe_run_sweep(args, dataset, test_subset, all_y_pred, all_y_true,
                       model="CanopyRS-DINO-SwinL", task="TreeBoxes"):
        return

    results, results_str = dataset.eval(
        all_y_pred, all_y_true, test_subset.metadata_array[:len(all_y_true)],
        viz_dir=args.viz_dir,
    )
    print(results_str)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f"results_boxes_{args.split_scheme}.txt"), "w") as f:
            f.write(results_str)
        flat = {k: float(v) if hasattr(v, "item") else v
                for k, v in results.items() if isinstance(v, (int, float, torch.Tensor))}
        with open(os.path.join(args.output_dir, f"results_boxes_{args.split_scheme}.json"), "w") as f:
            json.dump({"model": MODEL_NAME, "task": "TreeBoxes",
                       "split": args.split_scheme, "metrics": flat}, f, indent=2)


if __name__ == "__main__":
    main()
