"""Score-threshold sweep utilities for MillionTrees evals.

Two helpers used by the per-model eval scripts to answer "what score_threshold
maximizes recall/precision for this model?":

- ``subsample_per_source`` builds a small, source-stratified test subset (e.g. 3
  images per source) so the sweep is quick.
- ``run_threshold_sweep`` re-scores a single set of in-memory predictions at many
  ``eval_score_threshold`` values by mutating the dataset's metric objects in place
  (they read ``self.score_threshold`` live), so the model only runs inference once.

IMPORTANT: the model must emit its *full* score range for this to be meaningful.
Most models apply their own internal confidence floor (detectree2's
``SCORE_THRESH_TEST``, the ``--score-threshold`` keep filter, SAM3's post-process
threshold, ...). Drive that emission threshold to ~0 when sweeping, otherwise the
low-threshold rows are clipped by the model, not the evaluator.
"""

import csv
import os

import numpy as np

DEFAULT_THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                      0.9)


def subsample_per_source(dataset, split, per_source, seed=0):
    """Return a MillionTreesSubset with at most ``per_source`` images per source_id."""
    from milliontrees.datasets.milliontrees_dataset import MillionTreesSubset

    split_mask = dataset.split_array == dataset.split_dict[split]
    split_idx = np.where(split_mask)[0]

    source_field_idx = dataset.metadata_fields.index("source_id")
    sources = dataset.metadata_array[split_idx,
                                     source_field_idx].numpy().astype(int)

    rng = np.random.default_rng(seed)
    chosen = []
    for src in np.unique(sources):
        src_idx = split_idx[sources == src]
        if len(src_idx) > per_source:
            src_idx = rng.choice(src_idx, size=per_source, replace=False)
        chosen.extend(src_idx.tolist())
    chosen = np.sort(np.array(chosen, dtype=int))
    return MillionTreesSubset(dataset, chosen, None, dataset.geometry_name)


def _recall_key(dataset):
    for k in ("recall", "KeypointAccuracy"):
        if k in dataset.metrics:
            return k
    return None


def _recall_avg(results, dataset):
    """Average recall, handling the points special-case key.

    TreePoints stores its recall (keypoint_acc) average at ``keypoint_acc_avg_dom``;
    TreeBoxes/TreePolygons store it under the ``recall`` metric's ``agg_metric_field``.
    """
    if "keypoint_acc_avg_dom" in results:
        try:
            return float(results["keypoint_acc_avg_dom"])
        except (TypeError, ValueError):
            return float("nan")
    return _metric_avg(results, dataset, _recall_key(dataset))


def _metric_avg(results, dataset, key):
    """Pull the aggregate average for a metric out of the results dict."""
    if key is None or key not in results or key not in dataset.metrics:
        return float("nan")
    metric = dataset.metrics[key]
    field = getattr(metric, "agg_metric_field", None)
    try:
        return float(results[key][field])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _set_threshold(dataset, t):
    # TreePoints.eval re-instantiates KeypointAccuracy from eval_score_threshold, and
    # other places may read it directly, so set the dataset-level value too.
    if hasattr(dataset, "eval_score_threshold"):
        dataset.eval_score_threshold = t
    for metric in dataset.metrics.values():
        if hasattr(metric, "score_threshold"):
            metric.score_threshold = t
        inner = getattr(metric, "_mask_accuracy", None)
        if inner is not None and hasattr(inner, "score_threshold"):
            inner.score_threshold = t


def run_threshold_sweep(dataset,
                        y_pred,
                        y_true,
                        metadata,
                        *,
                        model,
                        task,
                        split,
                        out_csv,
                        thresholds=DEFAULT_THRESHOLDS):
    """Re-score predictions across ``thresholds``; print and append rows to ``out_csv``.

    Returns the list of row dicts. The best row (max F1) is printed at the end.
    """
    rows = []
    for t in thresholds:
        _set_threshold(dataset, t)
        results, _ = dataset.eval(y_pred, y_true, metadata, viz_dir=None)
        recall = _recall_avg(results, dataset)
        precision = _metric_avg(results, dataset, "maskaware_precision")
        ap50 = _metric_avg(results, dataset, "AP50")
        denom = recall + precision
        f1 = (2 * recall * precision /
              denom) if denom and not np.isnan(denom) else 0.0
        row = {
            "model": model,
            "task": task,
            "split": split,
            "threshold": t,
            "recall": recall,
            "precision": precision,
            "ap50": ap50,
            "f1": f1
        }
        rows.append(row)
        print(
            f"[{model}/{task}/{split}] t={t:.2f}  "
            f"recall={recall:.3f}  precision={precision:.3f}  "
            f"ap50={ap50:.3f}  f1={f1:.3f}",
            flush=True)

    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        write_header = not os.path.exists(out_csv)
        with open(out_csv, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    valid = [r for r in rows if not np.isnan(r["f1"]) and r["f1"] > 0]
    if valid:
        best = max(valid, key=lambda r: r["f1"])
        print(
            f"BEST {model}/{task}/{split}: t={best['threshold']:.2f}  "
            f"f1={best['f1']:.3f}  recall={best['recall']:.3f}  "
            f"precision={best['precision']:.3f}",
            flush=True)
    return rows


# ---------------------------------------------------------------------------
# CLI glue shared by the per-model eval scripts.
# ---------------------------------------------------------------------------


def add_sweep_args(parser):
    """Add --per-source / --sweep to an eval script's argument parser."""
    parser.add_argument(
        "--per-source",
        type=int,
        default=0,
        help="If >0, subsample this many test images per source "
        "(source-stratified) for a quick run.")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep eval score_threshold over a grid and append rows "
        "to <output-dir>/threshold_sweep.csv (drive the model's "
        "own --score-threshold to ~0 so the full range is seen).")
    parser.add_argument(
        "--sweep-thresholds",
        type=str,
        default=None,
        help="Comma-separated eval score_thresholds to sweep "
        "(e.g. '0.025,0.05,0.075,0.1'). Defaults to the built-in grid.")


def maybe_subsample(dataset, test_subset, args):
    """Return a per-source-subsampled test subset when --per-source is set."""
    if getattr(args, "per_source", 0):
        return subsample_per_source(dataset, "test", args.per_source)
    return test_subset


def maybe_run_sweep(args, dataset, test_subset, y_pred, y_true, *, model, task):
    """If --sweep, run the threshold sweep and return True (caller should stop)."""
    if not getattr(args, "sweep", False):
        return False
    out_csv = os.path.join(
        getattr(args, "output_dir", None) or ".", "threshold_sweep.csv")
    thresholds = DEFAULT_THRESHOLDS
    sweep_thresholds = getattr(args, "sweep_thresholds", None)
    if sweep_thresholds:
        thresholds = tuple(
            float(t) for t in sweep_thresholds.split(",") if t.strip())
    run_threshold_sweep(dataset,
                        y_pred,
                        y_true,
                        test_subset.metadata_array[:len(y_true)],
                        model=model,
                        task=task,
                        split=args.split_scheme,
                        out_csv=out_csv,
                        thresholds=thresholds)
    return True
