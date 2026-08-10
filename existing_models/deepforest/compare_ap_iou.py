"""Compare TreeBoxes AP at IoU 0.5 (AP50) vs IoU 0.4 (AP40) for pretrained DeepForest.

Motivation: the leaderboard reports AP50 (IoU 0.5), while the custom recall/precision
metrics match at IoU 0.4. Before restricting AP50 to only the (completely-annotated)
validation datasets, we want to know how sensitive the AP number is to the IoU match
threshold. This runs DeepForest once, then scores the *same* predictions with a
DetectionMAP at iou_thresholds=[0.5] and at [0.4], overall and per source.

Usage (GPU node):
    uv run python compare_ap_iou.py --split-scheme out-of-distribution \
        --eval-split validation --score-threshold 0.1 --output-dir outputs/ap_iou_compare
"""

import argparse
import json
import os
import warnings

from deepforest import main as df_main

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader
from milliontrees.common.metrics.all_metrics import DetectionMAP

# Reuse the exact prediction path from the leaderboard eval.
from eval_boxes import predict_batch


def main():
    parser = argparse.ArgumentParser(
        description="AP50 (IoU 0.5) vs AP40 (IoU 0.4) for pretrained DeepForest on TreeBoxes.")
    parser.add_argument("--root-dir", type=str,
                        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--split-scheme", type=str, default="out-of-distribution",
                        choices=["within-distribution", "out-of-distribution", "crossgeometry"])
    parser.add_argument("--eval-split", type=str, default="validation")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.1,
                        help="DeepForest internal score_thresh (matches leaderboard eval default 0.1).")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    model = df_main.deepforest()
    model.load_model("weecology/deepforest-tree")
    if args.score_threshold is not None:
        model.config["score_thresh"] = args.score_threshold
        if getattr(model, "model", None) is not None:
            model.model.score_thresh = args.score_threshold
    model.eval()

    dataset = get_dataset("TreeBoxes", root_dir=args.root_dir, download=args.download,
                          mini=args.mini, split_scheme=args.split_scheme)
    test_subset = dataset.get_subset(args.eval_split)
    test_loader = get_eval_loader("standard", test_subset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    print(f"Split: {args.split_scheme} / {args.eval_split}  Batches: {len(test_loader)}")

    all_y_pred, all_y_true = [], []
    for b_idx, (metadata, images, targets) in enumerate(test_loader):
        preds = predict_batch(images, model, metadata, targets, dataset, b_idx)
        for y_pred, target in zip(preds, targets):
            all_y_pred.append(y_pred)
            all_y_true.append(target)
        if args.max_batches is not None and (b_idx + 1) >= args.max_batches:
            break

    metadata_array = test_subset.metadata_array[:len(all_y_true)]

    # Two AP metrics, identical except the IoU match threshold.
    ap50 = DetectionMAP(geometry_name=dataset.geometry_name,
                        score_threshold=dataset.eval_score_threshold,
                        iou_type="bbox", iou_thresholds=[0.5], name="AP50")
    ap40 = DetectionMAP(geometry_name=dataset.geometry_name,
                        score_threshold=dataset.eval_score_threshold,
                        iou_type="bbox", iou_thresholds=[0.4], name="AP40")

    warnings.filterwarnings("ignore")

    g = dataset._eval_grouper.metadata_to_group(metadata_array)
    n_groups = dataset._eval_grouper.n_groups

    # Overall (macro-average over sources) via compute(); per-source via compute_group_wise().
    overall = {
        "AP50": ap50.compute(all_y_pred, all_y_true, return_dict=False).item(),
        "AP40": ap40.compute(all_y_pred, all_y_true, return_dict=False).item(),
    }
    gm50 = ap50.compute_group_wise(all_y_pred, all_y_true, g, n_groups, return_dict=True)
    gm40 = ap40.compute_group_wise(all_y_pred, all_y_true, g, n_groups, return_dict=True)

    print("\n=== AP50 (IoU 0.5) vs AP40 (IoU 0.4) — DeepForest pretrained ===")
    print(f"{'source':<28}{'AP50':>10}{'AP40':>10}{'Δ(40-50)':>12}")
    print("-" * 60)
    per_source = []
    for grp in range(n_groups):
        count = int(gm50[ap50.group_count_field(grp)])
        if count == 0:
            continue
        name = dataset._eval_grouper.group_field_str(grp)
        v50 = gm50[ap50.group_metric_field(grp)]
        v40 = gm40[ap40.group_metric_field(grp)]
        per_source.append({"source": name, "count": count, "AP50": v50, "AP40": v40})
        print(f"{name:<28}{v50:>10.3f}{v40:>10.3f}{(v40 - v50):>12.3f}  (n={count})")

    print("-" * 60)
    print(f"{'MACRO AVG (over sources)':<28}{overall['AP50']:>10.3f}"
          f"{overall['AP40']:>10.3f}{(overall['AP40'] - overall['AP50']):>12.3f}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out = {
            "model": "DeepForest-pretrained", "task": "TreeBoxes",
            "split_scheme": args.split_scheme, "eval_split": args.eval_split,
            "overall": {k: float(v) for k, v in overall.items()},
            "per_source": per_source,
        }
        path = os.path.join(args.output_dir, f"ap_iou_compare_{args.split_scheme}_{args.eval_split}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
