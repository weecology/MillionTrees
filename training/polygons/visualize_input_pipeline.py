"""Render what actually reaches the Mask R-CNN: real train/val batches, post-augmentation.

Built to debug https://www.comet.com/bw4sz/milliontrees-polygons/... (experiments
3c2328fc.../37fdbc9e...), where map falls after an early peak and val_loss rises for
the rest of training on both the box_pretrained and coco arms of the Table 6 OOD
ckptselect job. That pattern already matches the known Mask R-CNN overfitting behavior
(see notes/ and CLAUDE.md's --keep-all-checkpoints mitigation), but it had never been
checked against the literal tensors DeepForest feeds the model. This script builds the
exact same ``deepforest`` config as ``train.py`` (same CSVs, same augmentations) and
dumps N real batches from ``train_dataloader()`` / ``val_dataloader()`` as image+mask+box
overlays plus tensor stats (shape/dtype/value range), so a misaligned mask, a bad
normalization, or a train/val scale mismatch would be visible directly.

Usage:
    uv run python training/polygons/visualize_input_pipeline.py \\
        --train-csv training/weak_supervision/outputs/table6_ckptselect_v024/coco/out-of-distribution/deepforest_train.csv \\
        --val-csv training/weak_supervision/outputs/table6_ckptselect_v024/coco/out-of-distribution/deepforest_val.csv \\
        --images-dir /orange/ewhite/web/public/MillionTrees/TreePolygons_supervised_v0.24/images \\
        --train-aug annotationsafecrop \\
        --n-batches 3 --batch-size 4
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

from deepforest.main import deepforest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from training.polygons.train import DEFAULT_CONFIG, build_config

COLORS = [
    (255, 99, 71), (30, 144, 255), (50, 205, 50), (255, 215, 0),
    (218, 112, 214), (0, 206, 209), (255, 140, 0), (199, 21, 133),
]


def render_sample(image, targets, out_path):
    """image: (C,H,W) float tensor in [0,1]. targets: boxes/labels/panoptic_masks/unique_ids."""
    img = (image.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
    overlay = img.copy()

    panoptic = targets["panoptic_masks"].numpy()
    boxes = targets["boxes"].numpy()
    n = len(boxes)

    for i in range(n):
        color = COLORS[i % len(COLORS)]
        instance_id = i + 1
        mask = panoptic == instance_id
        overlay[mask] = (
            0.5 * np.array(color) + 0.5 * overlay[mask]
        ).astype(np.uint8)
        x1, y1, x2, y2 = boxes[i].astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)

    caption = f"H{img.shape[0]}xW{img.shape[1]} n_instances={n}"
    cv2.putText(overlay, caption, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, overlay)


def dump_loader(loader, split_name, out_dir, n_batches):
    os.makedirs(out_dir, exist_ok=True)
    n_done = 0
    for batch_idx, (images, targets, image_names) in enumerate(loader):
        for i, (image, target, name) in enumerate(zip(images, targets, image_names)):
            arr = image.numpy()
            n_inst = len(target["boxes"])
            print(
                f"[{split_name}] batch {batch_idx} sample {i}: name={name} "
                f"shape={tuple(image.shape)} dtype={image.dtype} "
                f"value_range=[{arr.min():.4f}, {arr.max():.4f}] "
                f"n_instances={n_inst} "
                f"panoptic_unique_ids={target['unique_ids'].tolist()[:10]}"
            )
            out_path = os.path.join(
                out_dir, f"{split_name}_b{batch_idx}_s{i}_{os.path.basename(name)}.png"
            )
            render_sample(image, target, out_path)
        n_done += 1
        if n_done >= n_batches:
            break


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--config", default=None,
                     help="Defaults to the vendored deepforest_polygon.yaml (same as train.py).")
    ap.add_argument("--train-aug", default="annotationsafecrop",
                     choices=["crop", "resize", "nativecrop", "annotationsafecrop"])
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--warmup-epochs", type=int, default=0)
    ap.add_argument("--warmup-start-factor", type=float, default=0.001)
    ap.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--val-aug-match-train", action=argparse.BooleanOptionalAction, default=False,
                     help="Mirror train.py's flag of the same name; see its --help.")
    ap.add_argument("--n-batches", type=int, default=3,
                     help="Batches to dump per split.")
    ap.add_argument("--output-dir", default="notes/polygon_input_pipeline_viz")
    args = ap.parse_args()

    build_args = argparse.Namespace(
        config=args.config or DEFAULT_CONFIG,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        gpus=args.gpus,
        lr=args.lr,
        max_epochs=1,
        warmup_epochs=args.warmup_epochs,
        warmup_start_factor=args.warmup_start_factor,
        augment=args.augment,
        train_aug=args.train_aug,
        image_size=args.image_size,
        val_aug_match_train=args.val_aug_match_train,
    )

    cfg = build_config(
        build_args, args.train_csv, args.val_csv, args.images_dir,
        log_root="/tmp/polygon_input_pipeline_viz_logs",
    )
    print(f"train.augmentations: {cfg.train.augmentations}")
    print(f"validation.augmentations: {cfg.validation.augmentations}")

    model = deepforest(config=cfg)

    train_loader = model.train_dataloader()
    val_loader = model.val_dataloader()

    dump_loader(train_loader, "train", args.output_dir, args.n_batches)
    if val_loader:
        dump_loader(val_loader, "val", args.output_dir, args.n_batches)
    else:
        print("No val_dataloader (empty val CSV).")

    print(f"\nWrote overlays to {args.output_dir}")


if __name__ == "__main__":
    main()
