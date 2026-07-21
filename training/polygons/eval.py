"""Evaluate a trained DeepForest Mask R-CNN checkpoint on MillionTrees TreePolygons."""

import argparse
import csv
import json
import os
import re

from pathlib import Path

import torch

from deepforest import utilities as df_utilities
from deepforest.main import deepforest

_POLYGON_CONFIG = str(Path(__file__).with_name("deepforest_polygon.yaml"))

from training.polygons.train import (
    ensure_predict_trainer,
    evaluate,
    tiled_predict_for_eval,
)
from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader
from milliontrees.datasets.polygon_stream_eval import TreePolygonsStreamingEvalState


def _set_model_score_thresh(model, value):
    """Lower the underlying Mask R-CNN roi_heads score gate, if exposed."""
    roi_heads = getattr(getattr(model, "model", None), "roi_heads", None)
    if roi_heads is not None and hasattr(roi_heads, "score_thresh"):
        roi_heads.score_thresh = value
        return True
    return False


class _ThresholdView:
    """Lightweight dataset proxy with its own metrics at a given score threshold.

    Shares the underlying dataset's grouper and metadata but swaps in a fresh,
    threshold-specific ``metrics`` dict so several thresholds can be scored from a
    single inference pass.
    """

    def __init__(self, base, score_threshold):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "metrics", base.build_metrics(score_threshold))
        object.__setattr__(self, "eval_score_threshold", score_threshold)

    def __getattr__(self, name):
        return getattr(self._base, name)


def _parse_avg(results_str, name):
    m = re.search(rf"Average {re.escape(name)}:\s*([\-\d.]+)", results_str)
    return float(m.group(1)) if m else float("nan")


def run_threshold_sweep(model, dataset, eval_subset, thresholds, *,
                        batch_size, device, compute_map=True):
    """Single inference pass scored at several eval score thresholds.

    Returns a list of dicts (one per threshold) with the headline metrics.

    ``compute_map`` gates the (expensive) AP50 accumulation; pass ``False`` on
    the incompletely-annotated test split where AP50 is not meaningful.
    """
    loader = get_eval_loader("standard", eval_subset, batch_size=batch_size)
    states = {t: TreePolygonsStreamingEvalState(_ThresholdView(dataset, t),
                                                compute_map=compute_map)
              for t in thresholds}

    model.eval()
    ensure_predict_trainer(model)
    for batch in loader:
        metadata, images, targets = batch
        preds = tiled_predict_for_eval(model, dataset, metadata)
        for state in states.values():
            state.update(preds, targets, metadata)

    rows = []
    for t in thresholds:
        _, results_str = states[t].finalize(viz_dir=None)
        recall = _parse_avg(results_str, "recall")
        precision = _parse_avg(results_str, "maskaware_precision")
        f1 = (2 * recall * precision / (recall + precision)
              if recall + precision > 0 else 0.0)
        rows.append({
            "eval_score_threshold": t,
            "recall": recall,
            "maskaware_precision": precision,
            "f1": f1,
            "ap50": _parse_avg(results_str, "AP50"),
            "accuracy": _parse_avg(results_str, "accuracy"),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Eval a trained TreePolygons checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .ckpt file")
    parser.add_argument("--root-dir", type=str,
                        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    parser.add_argument("--split-scheme", type=str, default="within-distribution",
                        choices=["within-distribution", "out-of-distribution", "crossgeometry"])
    parser.add_argument("--eval-split", type=str, default="test",
                        choices=["train", "validation", "test"],
                        help="Which subset to evaluate. 'validation' is the held-out "
                             "Allen et al. 2025 TLS set (same rows in every split-scheme).")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--init-mode", type=str, default="unknown")
    parser.add_argument("--data-scope", type=str, default="unknown")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-unsupervised", action="store_true")
    parser.add_argument("--viz-dir", type=str, default=None,
                        help="Directory for per-source prediction overlay PNGs "
                             "(default: <output-dir>/viz, else ./eval_viz; pass '' to disable)")
    parser.add_argument("--score-threshold", type=float, default=None,
                        help="Override the Mask R-CNN roi_heads box score threshold "
                             "(standardized to 0.1 for the leaderboard).")
    parser.add_argument("--eval-score-threshold", type=float, default=None,
                        help="Override the MillionTrees evaluator score threshold "
                             "(default 0.5). Predictions scoring below this are "
                             "dropped from both metrics and viz overlays.")
    parser.add_argument("--sweep-eval-thresholds", type=str, default=None,
                        help="Comma-separated evaluator score thresholds to sweep "
                             "in a single inference pass, e.g. '0.1,0.2,0.3,0.4,0.5'. "
                             "Writes <output-dir>/eval_threshold_sweep.csv and skips "
                             "the normal single-threshold eval.")
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="stream",
        choices=["stream", "legacy"],
        help="stream = low-memory per-batch metrics; legacy = accumulate then dataset.eval()",
    )
    args = parser.parse_args()

    # Visualization on by default: 10 overlays per source (dataset.eval viz_n_per_source=10).
    if args.viz_dir is None:
        args.viz_dir = os.path.join(args.output_dir, "viz") if args.output_dir else "eval_viz"
    elif args.viz_dir == "":
        args.viz_dir = None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sweep_thresholds = None
    if args.sweep_eval_thresholds:
        sweep_thresholds = [float(t) for t in args.sweep_eval_thresholds.split(",") if t.strip()]

    print(f"Loading checkpoint: {args.checkpoint}")
    _poly_cfg = df_utilities.load_config(config_name=_POLYGON_CONFIG, overrides={})
    model = deepforest(config=_poly_cfg)
    _ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(_ckpt["state_dict"])
    # For a sweep we need the full score distribution down to the lowest swept
    # threshold, so drop the model's roi_heads gate (unless explicitly overridden).
    if args.score_threshold is not None:
        if _set_model_score_thresh(model, args.score_threshold):
            print(f"roi_heads.score_thresh: {args.score_threshold}")
    elif sweep_thresholds is not None:
        gate = min(0.01, min(sweep_thresholds))
        if _set_model_score_thresh(model, gate):
            print(f"roi_heads.score_thresh (sweep): {gate}")
    model.eval()
    model = model.to(device)

    dataset_kwargs = dict(
        root_dir=args.root_dir,
        mini=args.mini,
        split_scheme=args.split_scheme,
        image_size=args.image_size,
        include_unsupervised=args.include_unsupervised,
    )
    if args.eval_score_threshold is not None:
        dataset_kwargs["eval_score_threshold"] = args.eval_score_threshold
    dataset = get_dataset("TreePolygons", **dataset_kwargs)
    eval_subset = dataset.get_subset(args.eval_split)

    if sweep_thresholds is not None:
        rows = run_threshold_sweep(
            model, dataset, eval_subset, sweep_thresholds,
            batch_size=args.batch_size, device=device,
            compute_map=(args.eval_split != "test"),
        )
        header = ["eval_score_threshold", "recall", "maskaware_precision", "f1", "ap50", "accuracy"]
        print("\nEval score-threshold sweep ({} split, {}):".format(
            args.split_scheme, args.eval_split))
        print("  thresh  recall  precision   f1     ap50   acc")
        best = max(rows, key=lambda r: r["f1"])
        for r in rows:
            mark = "  <-- best F1" if r is best else ""
            print(f"  {r['eval_score_threshold']:.3f}   {r['recall']:.3f}   "
                  f"{r['maskaware_precision']:.3f}     {r['f1']:.3f}  "
                  f"{r['ap50']:.3f}  {r['accuracy']:.3f}{mark}")
        print(f"BEST: eval_score_threshold={best['eval_score_threshold']:.3f} "
              f"F1={best['f1']:.3f} recall={best['recall']:.3f} ap50={best['ap50']:.3f}")
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            csv_path = os.path.join(args.output_dir, "eval_threshold_sweep.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerows(rows)
            print(f"Sweep CSV saved to {csv_path}")
        return

    results, results_str = evaluate(
        model,
        dataset,
        eval_subset,
        batch_size=args.batch_size,
        device=device,
        viz_dir=args.viz_dir,
        eval_mode=args.eval_mode,
        split=args.eval_split,
    )
    print(results_str)

    # Keep the canonical test results filename for backward compat; suffix any
    # non-test eval (e.g. validation) so it never clobbers leaderboard outputs.
    tag = args.split_scheme if args.eval_split == "test" else f"{args.split_scheme}_{args.eval_split}"
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"results_{tag}.txt")
        with open(out_path, "w") as f:
            f.write(results_str)
        json_path = os.path.join(args.output_dir, f"results_{tag}.json")
        flat = {k: float(v) if hasattr(v, "item") else v
                for k, v in results.items() if isinstance(v, (int, float, torch.Tensor))}
        with open(json_path, "w") as f:
            json.dump(
                {
                    "model": "trained-polygons",
                    "task": "TreePolygons",
                    "split": args.split_scheme,
                    "eval_split": args.eval_split,
                    "metrics": flat,
                    "run_metadata": {
                        "init_mode": args.init_mode,
                        "data_scope": args.data_scope,
                        "seed": args.seed,
                        "include_unsupervised": args.include_unsupervised,
                        "checkpoint": args.checkpoint,
                        "eval_mode": args.eval_mode,
                    },
                },
                f,
                indent=2,
            )
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
