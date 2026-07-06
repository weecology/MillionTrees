"""Evaluate a Detectree2 model_garden checkpoint on MillionTrees TreePolygons.

Detectree2 (https://github.com/PatBall1/detectree2) is a Detectron2 Mask R-CNN
trained for tree-crown delineation. We bypass its geospatial tiling pipeline and
run the underlying Detectron2 ``DefaultPredictor`` directly on MillionTrees image
tiles, then convert instance masks to the MillionTrees evaluation format:

    {"y": Tensor[N, H, W] uint8, "labels": Tensor[N] int64, "scores": Tensor[N] float32}

Download a checkpoint first (see README.md), e.g. ``250312_flexi.pth``.
"""

import argparse
import json
import os

import numpy as np
import torch

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader
from milliontrees.common.eval_sweep import (add_sweep_args, maybe_run_sweep,
                                            maybe_subsample)


def select_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def build_predictor(model_path: str, base_model: str, device: str,
                    score_threshold: float):
    """Build a Detectron2 DefaultPredictor from a Detectree2 checkpoint."""
    from detectree2.models.train import setup_cfg
    from detectron2.engine import DefaultPredictor

    cfg = setup_cfg(base_model=base_model, update_model=model_path)
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    return DefaultPredictor(cfg)


def image_to_bgr_uint8(image: torch.Tensor) -> np.ndarray:
    """MillionTrees image tensor (C,H,W float[0,1] RGB) -> HWC uint8 BGR.

    Detectree2 checkpoints use ``cfg.INPUT.FORMAT == "BGR"`` and are trained on
    cv2-read tiles, so we hand the predictor a BGR uint8 array.
    """
    rgb = (image.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def predict_one(predictor, image: torch.Tensor) -> dict:
    outputs = predictor(image_to_bgr_uint8(image))
    inst = outputs["instances"].to("cpu")
    if len(inst) == 0:
        h, w = image.shape[1:]
        return {"y": torch.zeros((0, h, w), dtype=torch.uint8),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,), dtype=torch.float32)}
    masks = inst.pred_masks.to(torch.uint8)
    scores = inst.scores.to(torch.float32)
    return {"y": masks, "labels": torch.zeros(len(inst), dtype=torch.int64),
            "scores": scores}


def main() -> None:
    parser = argparse.ArgumentParser(description="Detectree2 on TreePolygons (instance masks).")
    parser.add_argument("--root-dir", type=str,
                        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    parser.add_argument("--model-path", type=str,
                        default=os.environ.get("DETECTREE2_MODEL", "250312_flexi.pth"),
                        help="Path to a Detectree2 model_garden .pth checkpoint.")
    parser.add_argument("--base-model", type=str,
                        default="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--split-scheme", type=str, default="within-distribution",
                        choices=["within-distribution", "out-of-distribution", "crossgeometry"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--viz-dir", type=str, default=None,
                        help="Directory for per-source prediction overlay PNGs")
    add_sweep_args(parser)
    args = parser.parse_args()

    # Eval viz is on by default: write per-source overlays to <output-dir>/viz.
    if args.viz_dir is None and args.output_dir:
        args.viz_dir = os.path.join(args.output_dir, "viz")

    device = select_device(args.device)
    predictor = build_predictor(args.model_path, args.base_model, device,
                                args.score_threshold)

    dataset = get_dataset("TreePolygons", root_dir=args.root_dir, download=args.download,
                          mini=args.mini, split_scheme=args.split_scheme,
                          image_size=args.image_size)
    test_subset = maybe_subsample(dataset, dataset.get_subset(args.eval_split), args)
    test_loader = get_eval_loader("standard", test_subset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    print(f"Batches: {len(test_loader)}")

    all_y_pred, all_y_true = [], []
    for b_idx, (metadata, images, targets) in enumerate(test_loader):
        for image, target in zip(images, targets):
            all_y_pred.append(predict_one(predictor, image))
            all_y_true.append(target)
        if args.max_batches is not None and (b_idx + 1) >= args.max_batches:
            break

    if maybe_run_sweep(args, dataset, test_subset, all_y_pred, all_y_true,
                       model="Detectree2", task="TreePolygons"):
        return

    results, results_str = dataset.eval(
        all_y_pred, all_y_true, metadata=test_subset.metadata_array[:len(all_y_true)],
        viz_dir=args.viz_dir,
    )
    print(results_str)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f"results_polygons_{args.split_scheme}.txt"), "w") as f:
            f.write(results_str)
        flat = {k: float(v) if hasattr(v, "item") else v
                for k, v in results.items() if isinstance(v, (int, float, torch.Tensor))}
        with open(os.path.join(args.output_dir, f"results_polygons_{args.split_scheme}.json"), "w") as f:
            json.dump({"model": "Detectree2", "task": "TreePolygons",
                       "split": args.split_scheme, "metrics": flat}, f, indent=2)


if __name__ == "__main__":
    main()
