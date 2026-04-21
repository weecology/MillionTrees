"""Evaluate pretrained DeepForest on MillionTrees TreePolygons (box predictions, zero-shot)."""

import argparse
import json
import os
import warnings
from typing import List

import numpy as np
import torch

from deepforest import main as df_main
from deepforest.utilities import format_geometry

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader


def boxes_to_masks(boxes_df, height: int, width: int) -> np.ndarray:
    """Rasterize box predictions into (N, H, W) binary masks."""
    masks = []
    for _, row in boxes_df.iterrows():
        mask = np.zeros((height, width), dtype=bool)
        x1 = max(0, int(row["xmin"]))
        y1 = max(0, int(row["ymin"]))
        x2 = min(width, int(np.ceil(row["xmax"])))
        y2 = min(height, int(np.ceil(row["ymax"])))
        mask[y1:y2, x1:x2] = True
        masks.append(mask)
    return np.stack(masks) if masks else np.zeros((0, height, width), dtype=bool)


def predict_batch(
    images: torch.Tensor,
    model,
    metadata: torch.Tensor,
    targets: List[dict],
    batch_index: int,
) -> List[dict]:
    warnings.filterwarnings("ignore")
    _, _, img_h, img_w = images.shape
    predictions = model.predict_step(images, batch_index)
    y_preds: List[dict] = []
    for pred in predictions:
        if pred is None or len(pred["boxes"]) == 0:
            y_preds.append({
                "y": torch.zeros((0, img_h, img_w), dtype=torch.bool),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,), dtype=torch.float32),
            })
        else:
            df = format_geometry(pred)
            masks = boxes_to_masks(df, img_h, img_w)
            y_preds.append({
                "y": torch.from_numpy(masks),
                "labels": torch.tensor(df.label.values.astype(np.int64)),
                "scores": torch.tensor(df.score.values.astype("float32")),
            })
    return y_preds


def main():
    parser = argparse.ArgumentParser(description="Pretrained DeepForest on TreePolygons (box proxy, zero-shot).")
    parser.add_argument("--root-dir", type=str,
                        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--split-scheme", type=str, default="random",
                        choices=["random", "zeroshot", "crossgeometry"])
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--viz-dir", type=str, default=None,
                        help="Directory for per-source prediction overlay PNGs")
    args = parser.parse_args()

    model = df_main.deepforest()
    model.load_model("weecology/deepforest-tree")
    model.eval()

    dataset = get_dataset("TreePolygons", root_dir=args.root_dir, download=args.download,
                          mini=args.mini, split_scheme=args.split_scheme,
                          image_size=args.image_size)
    test_subset = dataset.get_subset("test")
    test_loader = get_eval_loader("standard", test_subset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    print(f"Batches: {len(test_loader)}")

    all_y_pred, all_y_true = [], []
    for b_idx, (metadata, images, targets) in enumerate(test_loader):
        preds = predict_batch(images, model, metadata, targets, b_idx)
        for y_pred, target in zip(preds, targets):
            all_y_pred.append(y_pred)
            all_y_true.append(target)
        if args.max_batches is not None and (b_idx + 1) >= args.max_batches:
            break

    results, results_str = dataset.eval(
        all_y_pred, all_y_true, test_subset.metadata_array[:len(all_y_true)],
        viz_dir=args.viz_dir,
    )
    print(results_str)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        txt_path = os.path.join(args.output_dir, f"results_polygons_{args.split_scheme}.txt")
        with open(txt_path, "w") as f:
            f.write(results_str)
        json_path = os.path.join(args.output_dir, f"results_polygons_{args.split_scheme}.json")
        flat = {k: float(v) if hasattr(v, "item") else v
                for k, v in results.items() if isinstance(v, (int, float, torch.Tensor))}
        with open(json_path, "w") as f:
            json.dump({"model": "DeepForest-pretrained", "task": "TreePolygons",
                       "split": args.split_scheme, "metrics": flat}, f, indent=2)


if __name__ == "__main__":
    main()
