"""Select a TreePolygons checkpoint on the held-out validation split, then score it on test.

The polygon Mask R-CNN's ``val_loss`` bottoms out by ~epoch 9 and then rises while
detection quality keeps improving, so ``ModelCheckpoint(monitor="val_loss")`` hands
the final eval an under-trained network (see notes/polygon_maskrcnn_overfits and
CLAUDE.md). This pass re-scores *every* checkpoint kept by ``--keep-all-checkpoints``
on the Allen et al. 2025 validation split (identical rows in every split-scheme, never
in train/test), picks the highest validation F1, and runs the canonical test eval with
that checkpoint so ``results_<split>.txt`` is written exactly as train.py would.

Usage:
    python training/polygons/select_checkpoint.py \
        --checkpoints-dir <out>/logs/checkpoints \
        --split-scheme out-of-distribution \
        --init-mode coco \
        --image-size 448 \
        --output-dir <out>
"""

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

EVAL = str(Path(__file__).with_name("eval.py"))


def _parse_avg(text, metric):
    """Pull 'Average <metric>: <float>' out of a results_*.txt body."""
    m = re.search(rf"Average {re.escape(metric)}:\s*([-\d.]+)", text)
    return float(m.group(1)) if m else 0.0


def _run_eval(checkpoint, *, split_scheme, eval_split, image_size, init_mode, output_dir,
              eval_inference="tiled"):
    """Invoke eval.py for one checkpoint/split and return its parsed metrics dict."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable, EVAL,
        "--checkpoint", checkpoint,
        "--split-scheme", split_scheme,
        "--eval-split", eval_split,
        "--image-size", str(image_size),
        "--init-mode", init_mode,
        "--eval-inference", eval_inference,
        "--output-dir", output_dir,
        "--viz-dir", "",
    ]
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    tag = split_scheme if eval_split == "test" else f"{split_scheme}_{eval_split}"
    with open(os.path.join(output_dir, f"results_{tag}.txt")) as f:
        text = f.read()
    r = _parse_avg(text, "recall")
    p = _parse_avg(text, "maskaware_precision")
    return {
        "recall": r,
        "maskaware_precision": p,
        "f1": 2 * r * p / (r + p) if (r + p) > 0 else 0.0,
        "ap40": _parse_avg(text, "AP40"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints-dir", required=True)
    ap.add_argument("--split-scheme", default="out-of-distribution")
    ap.add_argument("--init-mode", default="unknown")
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--select-metric", default="f1", choices=["f1", "recall", "ap40"])
    ap.add_argument("--eval-inference", default="tiled", choices=["tiled", "resize"],
                    help="Inference mode for every eval.py call (validation scoring "
                         "and the final test eval). Must match --train-aug: resize "
                         "for whole-image resize training, tiled otherwise.")
    args = ap.parse_args()

    ckpts = sorted(glob.glob(os.path.join(args.checkpoints_dir, "polygons-epoch=*.ckpt")))
    if not ckpts:
        sys.exit(f"No polygons-epoch=*.ckpt under {args.checkpoints_dir}")
    print(f"Scoring {len(ckpts)} checkpoints on the validation split:")
    for c in ckpts:
        print(f"  {os.path.basename(c)}")

    rows = []
    for ckpt in ckpts:
        name = os.path.basename(ckpt)[:-5]
        m = _run_eval(
            ckpt,
            split_scheme=args.split_scheme, eval_split="validation",
            image_size=args.image_size, init_mode=args.init_mode,
            eval_inference=args.eval_inference,
            output_dir=os.path.join(args.output_dir, "select", name),
        )
        rows.append({
            "checkpoint": name,
            "path": ckpt,
            "val_recall": m["recall"],
            "val_maskaware_precision": m["maskaware_precision"],
            "val_f1": m["f1"],
            "val_ap40": m["ap40"],
        })

    key = {"f1": "val_f1", "recall": "val_recall", "ap40": "val_ap40"}[args.select_metric]
    rows.sort(key=lambda r: r[key], reverse=True)
    best = rows[0]

    csv_path = os.path.join(args.output_dir, "checkpoint_selection_validation.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nValidation scores ({args.select_metric} selection) -> {csv_path}")
    for r in rows:
        mark = "  <-- selected" if r is best else ""
        print(f"  {r['checkpoint']:<34} val_f1={r['val_f1']:.3f} "
              f"val_recall={r['val_recall']:.3f} val_ap40={r['val_ap40']:.3f}{mark}")

    print(f"\n=== Test eval with selected checkpoint: {best['checkpoint']} ===")
    _run_eval(
        best["path"],
        split_scheme=args.split_scheme, eval_split="test",
        image_size=args.image_size, init_mode=args.init_mode,
        eval_inference=args.eval_inference,
        output_dir=args.output_dir,
    )
    # Leave a breadcrumb next to the canonical results.
    with open(os.path.join(args.output_dir, "selected_checkpoint.json"), "w") as f:
        json.dump({"selected": best, "select_metric": args.select_metric,
                   "candidates": rows}, f, indent=1)
    print(f"Canonical test results written to "
          f"{args.output_dir}/results_{args.split_scheme}.txt")


if __name__ == "__main__":
    main()
