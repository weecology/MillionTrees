"""Evaluate pretrained DeepForest on MillionTrees TreeBoxes (zero-shot)."""

import argparse
import json
import os
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from deepforest import main as df_main
from deepforest.utilities import format_geometry

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader


def predict_batch(
    images: torch.Tensor,
    model,
    metadata: torch.Tensor,
    targets: List[dict],
    dataset,
    batch_index: int,
) -> List[dict]:
    warnings.filterwarnings("ignore")
    predictions = model.predict_step(images, batch_index)
    y_preds: List[dict] = []
    for image_metadata, pred in zip(metadata, predictions):
        if pred is None or len(pred["boxes"]) == 0:
            y_preds.append({
                "y": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,), dtype=torch.float32),
            })
        else:
            df = format_geometry(pred)
            y_preds.append({
                "y": torch.tensor(df[["xmin", "ymin", "xmax", "ymax"]].values.astype("float32")),
                "labels": torch.tensor(df.label.values.astype(np.int64)),
                "scores": torch.tensor(df.score.values.astype("float32")),
            })
    return y_preds


def main():
    parser = argparse.ArgumentParser(description="Pretrained DeepForest on TreeBoxes.")
    parser.add_argument("--root-dir", type=str,
                        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--split-scheme", type=str, default="random",
                        choices=["random", "zeroshot", "crossgeometry"])
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--viz-dir", type=str, default=None,
                        help="Directory for per-source prediction overlay PNGs "
                             "(default: <output-dir>/viz, else ./eval_viz; pass '' to disable)")
    args = parser.parse_args()

    # Visualization on by default: 10 overlays per source (dataset.eval viz_n_per_source=10).
    if args.viz_dir is None:
        args.viz_dir = os.path.join(args.output_dir, "viz") if args.output_dir else "eval_viz"
    elif args.viz_dir == "":
        args.viz_dir = None

    model = df_main.deepforest()
    model.load_model("weecology/deepforest-tree")
    model.eval()

    dataset = get_dataset("TreeBoxes", root_dir=args.root_dir, download=args.download,
                          mini=args.mini, split_scheme=args.split_scheme)
    test_subset = dataset.get_subset("test")
    test_loader = get_eval_loader("standard", test_subset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    print(f"Batches: {len(test_loader)}")

    all_y_pred, all_y_true = [], []
    for b_idx, (metadata, images, targets) in enumerate(test_loader):
        preds = predict_batch(images, model, metadata, targets, dataset, b_idx)
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
        txt_path = os.path.join(args.output_dir, f"results_boxes_{args.split_scheme}.txt")
        with open(txt_path, "w") as f:
            f.write(results_str)
        json_path = os.path.join(args.output_dir, f"results_boxes_{args.split_scheme}.json")
        flat = {k: float(v) if hasattr(v, "item") else v
                for k, v in results.items() if isinstance(v, (int, float, torch.Tensor))}
        with open(json_path, "w") as f:
            json.dump({"model": "DeepForest-pretrained", "task": "TreeBoxes",
                       "split": args.split_scheme, "metrics": flat}, f, indent=2)


if __name__ == "__main__":
    main()
