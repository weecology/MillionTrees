"""Evaluate pretrained TreeFormer on MillionTrees TreePoints.

Inference-only baseline (no training on MillionTrees). For out-of-distribution or within-distribution
splits, pass ``--split-scheme`` to select which held-out test sources to score.
"""

import argparse
import json
import os
import warnings

import torch

from deepforest import main as df_main

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader
from milliontrees.common.eval_sweep import (add_sweep_args, maybe_run_sweep,
                                            maybe_subsample)


def predict_batch(model, images):
    """Run TreeFormer inference; return MillionTrees-format prediction dicts."""
    device = next(model.parameters()).device
    if not isinstance(images, torch.Tensor):
        images = torch.tensor(images)
    images = images.to(device)
    model.eval()
    with torch.no_grad():
        preds = model.predict_step(images, 0)

    batch_y_pred = []
    for pred in preds:
        points = pred.get("points", torch.zeros((0, 2)))
        if len(points) == 0:
            batch_y_pred.append({
                "y": torch.zeros((0, 2), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,), dtype=torch.float32),
            })
        else:
            scores = pred.get("scores", torch.ones(len(points), dtype=torch.float32))
            labels = pred.get("labels", torch.zeros(len(points), dtype=torch.int64))
            batch_y_pred.append({
                "y": points.detach().float().cpu(),
                "labels": labels.detach().cpu().long(),
                "scores": scores.detach().float().cpu(),
            })
    return batch_y_pred


def main():
    parser = argparse.ArgumentParser(description="Pretrained TreeFormer on TreePoints.")
    parser.add_argument("--root-dir", type=str,
                        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--small", action="store_true",
                        help="Use the SmallTreePoints release for fast iteration.")
    parser.add_argument("--image-size", type=int, default=448,
                        help="Square resize for images/points at eval. Also sets "
                             "the KeypointAccuracy pixel threshold (= normalized "
                             "distance x image_size).")
    parser.add_argument("--eval-score-threshold", type=float, default=None,
                        help="Override the dataset eval_score_threshold (the hard "
                             "filter on predicted-point scores). Default (None) "
                             "keeps the dataset default of 0.4. Ignored under --sweep.")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--split-scheme", type=str, default="within-distribution",
                        choices=["within-distribution", "out-of-distribution", "crossgeometry"])
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--viz-dir", type=str, default=None,
                        help="Directory for per-source prediction overlay PNGs "
                             "(default: <output-dir>/viz, else ./eval_viz; pass '' to disable)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="weecology/deepforest-tree-point",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.1,
        help="Relative peak threshold in [0, 1] for density_to_points. Lower "
             "values yield more (lower-confidence) points, trading precision "
             "for recall. Standard default 0.1 (note: the binding score filter "
             "is the dataset eval_score_threshold, not this).",
    )
    parser.add_argument(
        "--score-integration-radius",
        type=int,
        default=2,
        help="min_distance (in density-map pixels, ~4x in image px) for "
             "peak_local_max. Lower values allow closer-spaced detections, "
             "raising recall in dense canopy. Standard default 2 (tuned from "
             "the deepforest default of 5).",
    )
    add_sweep_args(parser)
    args = parser.parse_args()

    # Visualization on by default: 10 overlays per source (dataset.eval viz_n_per_source=10).
    if args.viz_dir is None:
        args.viz_dir = os.path.join(args.output_dir, "viz") if args.output_dir else "eval_viz"
    elif args.viz_dir == "":
        args.viz_dir = None

    warnings.filterwarnings("ignore")

    model = df_main.deepforest(config_args={"architecture": "treeformer"})
    model.load_model(args.checkpoint)
    # NOTE: postprocess_density reads the *submodule* attributes
    # (model.model.score_thresh / .score_integration_radius), which are frozen
    # at construction from config. Overriding model.config after load_model is a
    # no-op, so set them on the submodule directly.
    if args.score_thresh is not None:
        model.model.score_thresh = args.score_thresh
    if args.score_integration_radius is not None:
        model.model.score_integration_radius = args.score_integration_radius
    print(f"score_thresh: {model.model.score_thresh} "
          f"score_integration_radius: {model.model.score_integration_radius}")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    dataset_kwargs = dict(
        root_dir=args.root_dir,
        download=args.download,
        mini=args.mini,
        small=args.small,
        image_size=args.image_size,
        split_scheme=args.split_scheme,
    )
    if args.eval_score_threshold is not None:
        dataset_kwargs["eval_score_threshold"] = args.eval_score_threshold
    dataset = get_dataset("TreePoints", **dataset_kwargs)
    test_subset = maybe_subsample(dataset, dataset.get_subset(args.eval_split), args)
    test_loader = get_eval_loader(
        "standard", test_subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Batches: {len(test_loader)}")

    all_y_pred, all_y_true = [], []
    for b_idx, (_, images, targets) in enumerate(test_loader):
        preds = predict_batch(model, images)
        all_y_pred.extend(preds)
        all_y_true.extend(targets)
        if args.max_batches is not None and (b_idx + 1) >= args.max_batches:
            break

    if maybe_run_sweep(args, dataset, test_subset, all_y_pred, all_y_true,
                       model="TreeFormer-pretrained", task="TreePoints"):
        return

    results, results_str = dataset.eval(
        all_y_pred,
        all_y_true,
        test_subset.metadata_array[:len(all_y_true)],
        viz_dir=args.viz_dir,
    )
    print(results_str)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        txt_path = os.path.join(args.output_dir, f"results_points_{args.split_scheme}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(results_str)
        json_path = os.path.join(args.output_dir, f"results_points_{args.split_scheme}.json")
        flat = {
            k: float(v) if hasattr(v, "item") else v
            for k, v in results.items()
            if isinstance(v, (int, float, torch.Tensor))
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": "TreeFormer-pretrained",
                "task": "TreePoints",
                "split": args.split_scheme,
                "checkpoint": args.checkpoint,
                "metrics": flat,
            }, f, indent=2)


if __name__ == "__main__":
    main()
