"""Sweep the eval score threshold on a ``--save-predictions`` dump, offline.

The leaderboard applies a fixed ``eval_score_threshold`` (dataset default, currently 0.5)
to decide which predictions count toward recall / mask-aware precision / F1. A single
inference pass already carries the full score distribution down to whatever gate the
model was run at (e.g. Detectron2's ``SCORE_THRESH_TEST=0.15``), so the recall/precision
tradeoff across thresholds can be re-scored from one dump instead of one GPU run per
threshold.

Reuses the mask-aware matching/aggregation helpers from ``sweep_maskaware_precision.py``
(same metric, same ``tree_fraction_threshold`` cutoff held fixed at the leaderboard value)
and just varies the score threshold that ``analyze_image`` filters predictions on.

Usage:
    python scripts/sweep_score_threshold.py \
        --predictions outputs/threshold_sweep_preds/detectron2_polygons_out-of-distribution_test.pkl \
        --data-dir /orange/ewhite/web/public/MillionTrees/TreePolygons_v0.24 \
        --out-dir notes
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from milliontrees.common.prediction_dump import load_predictions
from scripts.sweep_maskaware_precision import analyze_image, aggregate

LEADERBOARD_CUTOFF = 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, help="Prediction dump (.pkl).")
    ap.add_argument("--data-dir", required=True,
                     help="Packaged dataset dir holding masks/ for the dump's split.")
    ap.add_argument("--thresholds", nargs="+", type=float,
                     default=[0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--cutoff", type=float, default=LEADERBOARD_CUTOFF,
                     help="Mask-aware tree_fraction_threshold to hold fixed (leaderboard=0.5).")
    ap.add_argument("--match-iou", type=float, default=0.4)
    ap.add_argument("--out-dir", default="notes")
    args = ap.parse_args()

    dump = load_predictions(args.predictions)
    meta = dump["meta"]
    iou_type = meta["iou_type"]
    model = meta.get("model") or Path(args.predictions).stem
    masks_dir = Path(args.data_dir) / "masks"

    print(f"=== {model} | {meta.get('task')} | {meta.get('split_scheme')} "
          f"| {meta.get('eval_split')} | {len(dump['images'])} images ===")

    rows = []
    for t in args.thresholds:
        images = [analyze_image(rec, iou_type, t, masks_dir, meta.get("image_size"),
                                  args.match_iou)
                  for rec in dump["images"]]
        for row in aggregate(images, args.cutoff):
            rows.append(dict(score_threshold=t, **row))

    df = pd.DataFrame(rows)
    macro = df[df["group"] == "overall (macro)"].sort_values("score_threshold")
    print(f"\n{'threshold':>9}  {'recall':>7}  {'precision':>9}  {'f1':>6}  "
          f"{'n_pred':>7}")
    for _, r in macro.iterrows():
        print(f"{r['score_threshold']:>9.2f}  {r['recall']:>7.3f}  "
              f"{r['precision']:>9.3f}  {r['f1']:>6.3f}  {r['n_pred']:>7d}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{Path(args.predictions).stem}_score_threshold_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")
    return df


if __name__ == "__main__":
    main()
