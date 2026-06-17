"""Evaluate the CanopyRS SelvaMask pipeline on MillionTrees TreePolygons.

Reproduces the SelvaMask instance-segmentation baseline: the SelvaMask fine-tuned
DINO Swin-L detector
(https://huggingface.co/CanopyRS/dino-swin-l-384-multi-NQOS-selvamask-FT) proposes
crown boxes, and a (SelvaMask fine-tuned) SAM 3 segmenter turns each box into an
instance mask. Detector boxes and SAM 3 segmentation both run through CanopyRS' own
model wrappers; we drive them per image on the MillionTrees test set and convert the
masks to the MillionTrees evaluation format.

CanopyRS (detrex + Detectron2) and gated access to ``facebook/sam3`` are required;
see README.md.
"""

import argparse
import json
import os
from typing import List

import numpy as np
import torch
from PIL import Image

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader
from milliontrees.common.eval_sweep import (add_sweep_args, maybe_run_sweep,
                                            maybe_subsample)
from milliontrees.datasets.polygon_stream_eval import (
    TreePolygonsStreamingEvalState, merge_viz_samples)

DETECTOR_CONFIG = "detectors/dino_swinL_multi_NQOS_selvamask_FT.yaml"
SEGMENTER_CONFIG = "segmenters/sam3_multi_selvamask_FT.yaml"
MODEL_NAME = "CanopyRS-DINO-SAM3-SelvaMask"


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_detector(config_name: str):
    from canopyrs.engine.config_parsers import DetectorConfig
    from canopyrs.engine.config_parsers.base import get_config_path
    from canopyrs.engine.models.detector.detectron2_infer import Detectron2DetectorWrapper

    config = DetectorConfig.from_yaml(get_config_path(config_name))
    config.batch_size = 1
    return Detectron2DetectorWrapper(config)


def load_segmenter(config_name: str):
    from canopyrs.engine.config_parsers import SegmenterConfig
    from canopyrs.engine.config_parsers.base import get_config_path
    from canopyrs.engine.models.segmenter.sam3 import Sam3PredictorWrapper

    config = SegmenterConfig.from_yaml(get_config_path(config_name))
    return Sam3PredictorWrapper(config)


def image_to_pil(image: torch.Tensor) -> Image.Image:
    array = (image.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def segment_boxes(segmenter, pil_image: Image.Image, boxes: np.ndarray):
    """Return ((N, H, W) uint8 masks, (N,) validity mask) for the given xyxy boxes.

    Masks are produced one per input box (batched by ``box_batch_size``); degenerate
    boxes get an all-zero mask and are flagged ``False`` in the validity mask.
    """
    width, height = pil_image.size
    boxes = boxes.astype(np.float32).copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

    masks = np.zeros((boxes.shape[0], height, width), dtype=np.uint8)
    valid_boxes = boxes[valid]
    if len(valid_boxes) == 0:
        return masks, valid

    batch_size = segmenter.config.box_batch_size or len(valid_boxes)
    predicted = []
    for start in range(0, len(valid_boxes), batch_size):
        batch_masks, _ = segmenter._predict_batch(pil_image, valid_boxes[start:start + batch_size])
        predicted.append(batch_masks)
    masks[valid] = np.concatenate(predicted, axis=0)
    return masks, valid


def main() -> None:
    parser = argparse.ArgumentParser(description="CanopyRS DINO + SAM3 on TreePolygons.")
    parser.add_argument("--root-dir", type=str,
                        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--split-scheme", type=str, default="within-distribution",
                        choices=["within-distribution", "out-of-distribution", "crossgeometry"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--viz-dir", type=str, default=None,
                        help="Directory for per-source prediction overlay PNGs")
    parser.add_argument("--viz-n-per-source", type=int, default=10,
                        help="Number of overlay PNGs to write per source.")
    add_sweep_args(parser)
    args = parser.parse_args()

    device = select_device(args.device)
    detector = load_detector(DETECTOR_CONFIG)
    segmenter = load_segmenter(SEGMENTER_CONFIG)

    dataset = get_dataset("TreePolygons", root_dir=args.root_dir, download=args.download,
                          mini=args.mini, split_scheme=args.split_scheme,
                          image_size=args.image_size)
    test_subset = maybe_subsample(dataset, dataset.get_subset("test"), args)
    test_loader = get_eval_loader("standard", test_subset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    print(f"Batches: {len(test_loader)}")

    # Sweeping thresholds needs every prediction held in memory; a plain full eval
    # streams metrics batch-by-batch so peak RAM is one batch of masks, not the whole
    # test set (~270 GiB of dense 1024x1024 masks otherwise).
    sweep_mode = getattr(args, "sweep", False)
    stream_eval = None if sweep_mode else TreePolygonsStreamingEvalState(dataset)
    all_y_pred, all_y_true = [], []  # only populated in sweep mode

    # Capped per-source subset kept for visualization in streaming mode.
    viz_cap: dict = {}
    viz_y_pred: List[dict] = []
    viz_y_true: List[dict] = []
    viz_rows: List[torch.Tensor] = []

    for b_idx, (metadata, images, targets) in enumerate(test_loader):
        image_list: List[torch.Tensor] = [img.to(device) for img in images]
        preds = detector.forward(image_list)

        batch_y_pred: List[dict] = []
        batch_y_true: List[dict] = []
        for img, pred, target in zip(images, preds, targets):
            boxes = pred["boxes"].float().cpu()
            scores = pred["scores"].float().cpu()
            keep = scores >= args.score_threshold
            boxes, scores = boxes[keep], scores[keep]

            height, width = img.shape[-2], img.shape[-1]
            if boxes.shape[0] == 0:
                batch_y_pred.append({
                    "y": torch.zeros((0, height, width), dtype=torch.uint8),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                })
                batch_y_true.append(target)
                continue

            pil_image = image_to_pil(img)
            masks, valid = segment_boxes(segmenter, pil_image, boxes.numpy())
            masks = torch.from_numpy(masks).to(torch.uint8)
            scores = scores * torch.from_numpy(valid.astype(np.float32))
            batch_y_pred.append({
                "y": masks,
                "labels": torch.zeros(masks.shape[0], dtype=torch.int64),
                "scores": scores,
            })
            batch_y_true.append(target)

        batch_meta = metadata[:len(batch_y_pred)]
        if sweep_mode:
            all_y_pred.extend(batch_y_pred)
            all_y_true.extend(batch_y_true)
        else:
            stream_eval.update(batch_y_pred, batch_y_true, batch_meta)
            if args.viz_dir:
                merge_viz_samples(viz_cap, batch_meta, batch_y_pred, batch_y_true,
                                  viz_y_pred=viz_y_pred, viz_y_true=viz_y_true,
                                  viz_rows=viz_rows, n_per_source=args.viz_n_per_source)

        if args.max_batches is not None and (b_idx + 1) >= args.max_batches:
            break

    if sweep_mode:
        if maybe_run_sweep(args, dataset, test_subset, all_y_pred, all_y_true,
                           model=MODEL_NAME, task="TreePolygons"):
            return
        results, results_str = dataset.eval(
            all_y_pred, all_y_true,
            metadata=test_subset.metadata_array[:len(all_y_true)],
            viz_dir=args.viz_dir,
        )
    else:
        results, results_str = stream_eval.finalize(
            viz_dir=args.viz_dir,
            viz_y_pred=viz_y_pred or None,
            viz_y_true=viz_y_true or None,
            viz_metadata=torch.stack(viz_rows) if viz_rows else None,
            viz_n_per_source=args.viz_n_per_source,
        )
    print(results_str)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f"results_polygons_{args.split_scheme}.txt"), "w") as f:
            f.write(results_str)
        flat = {k: float(v) if hasattr(v, "item") else v
                for k, v in results.items() if isinstance(v, (int, float, torch.Tensor))}
        with open(os.path.join(args.output_dir, f"results_polygons_{args.split_scheme}.json"), "w") as f:
            json.dump({"model": MODEL_NAME, "task": "TreePolygons",
                       "split": args.split_scheme, "metrics": flat}, f, indent=2)


if __name__ == "__main__":
    main()
