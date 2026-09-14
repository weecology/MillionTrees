"""Re-score dumped validation predictions: AP at several IoUs, all tiles vs complete tiles.

Answers the manuscript blocker "is AP50 low because the reference data is incompletely
annotated, and does aligning AP's IoU with F1's IoU close the gap?" without re-running any
model. Reads a ``--save-predictions`` dump (see ``milliontrees.common.prediction_dump``)
and reports, per source and overall:

  recall@0.4, plain precision@0.4, F1  -- the F1 ingredients, but WITHOUT the
      tree-coverage-mask forgiveness the leaderboard's mask-aware precision applies
  AP40, AP50, AP60                     -- identical predictions, only the matching IoU
      differs. AP40 is the leaderboard AP (it uses the IoU recall/precision match at);
      AP60 is the stricter localisation complement reported beside it.

for three tile regimes:

  ``all``       every image in the split (reproduces the reported numbers)
  ``complete``  only tiles whose annotation footprint reaches the tile edge, i.e. tiles
                that are annotated wall-to-wall (``--max-margin``)
  ``cropped``   every tile, but restricted to its annotation footprint: ground truth and
                predictions outside the annotated region are dropped, so predictions are
                never charged as false positives for finding trees nobody labelled

Usage:
    python scripts/rescore_validation_ap.py \
        --predictions existing_models/canopyrs/outputs/validation/preds_boxes.pkl \
        --out-csv notes/validation_ap_iou_completeness.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from milliontrees.common.metrics.matching import greedy_iou_match, n_matched_gt
from milliontrees.common.prediction_dump import decode_masks, load_predictions

REGIMES = ("all", "complete", "cropped")

# AP is reported at each of these IoUs. 0.4 is the leaderboard AP (matches the IoU
# recall / mask-aware precision use); 0.6 is the stricter localisation complement.
AP_IOUS = (0.4, 0.5, 0.6)


def ap_key(iou):
    """Column name for AP at ``iou`` -- 0.4 -> 'AP40'."""
    return f"AP{int(round(iou * 100))}"


# --------------------------------------------------------------------------- #
# Geometry helpers -- everything is in the evaluation frame (the resized image)
# --------------------------------------------------------------------------- #
def boxes_from_masks(masks):
    """Tight [N, 4] boxes around each mask in an [N, H, W] stack."""
    if len(masks) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = np.zeros((len(masks), 4), dtype=np.float32)
    for i, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            continue
        out[i] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    return out


def footprint(gt_boxes):
    """Axis-aligned annotated region: the union extent of the ground-truth geometry."""
    if len(gt_boxes) == 0:
        return None
    return np.array([gt_boxes[:, 0].min(), gt_boxes[:, 1].min(),
                     gt_boxes[:, 2].max(), gt_boxes[:, 3].max()], dtype=np.float64)


def centers(boxes):
    if len(boxes) == 0:
        return np.zeros((0, 2))
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)


def inside(boxes, region):
    """Boolean keep-mask: geometry whose centroid falls in ``region``."""
    if region is None or len(boxes) == 0:
        return np.ones(len(boxes), dtype=bool)
    c = centers(boxes)
    return ((c[:, 0] >= region[0]) & (c[:, 0] <= region[2]) &
            (c[:, 1] >= region[1]) & (c[:, 1] <= region[3]))


def image_size(rec, iou_type):
    if iou_type == "segm":
        shape = rec["true"]["masks"].get("shape") or rec["pred"]["masks"].get("shape")
        if shape and shape[0]:
            return float(shape[1]), float(shape[0])
        return None
    return None


# --------------------------------------------------------------------------- #
# Per-image record: decode once, reuse for every regime
# --------------------------------------------------------------------------- #
def build_records(dump, score_threshold):
    iou_type = dump["meta"]["iou_type"]
    records = []
    for rec in dump["images"]:
        pred, true = rec["pred"], rec["true"]
        scores = pred.get("scores")
        scores = np.zeros(0, dtype=np.float32) if scores is None else scores
        if iou_type == "segm":
            pred_masks = decode_masks(pred["masks"])
            gt_masks = decode_masks(true["masks"])
            pred_boxes = boxes_from_masks(pred_masks)
            gt_boxes = boxes_from_masks(gt_masks)
            H, W = (gt_masks.shape[1:] if len(gt_masks) else
                    (pred_masks.shape[1:] if len(pred_masks) else (0, 0)))
        else:
            pred_masks = gt_masks = None
            pred_boxes = pred["geometry"]
            gt_boxes = true["geometry"]
            H = W = float(dump["meta"].get("image_size") or 0)
        keep = scores > score_threshold if len(scores) else np.zeros(len(pred_boxes), bool)
        records.append(dict(
            filename=rec["filename"], source=rec["source"],
            pred_boxes=pred_boxes[keep] if len(pred_boxes) else pred_boxes,
            pred_masks=pred_masks[keep] if pred_masks is not None and len(pred_masks) else pred_masks,
            pred_scores=scores[keep] if len(scores) else scores,
            gt_boxes=gt_boxes, gt_masks=gt_masks, height=float(H), width=float(W),
        ))
    return records, iou_type


def annotate_completeness(records):
    """Attach the annotated-footprint geometry and its coverage of the tile."""
    for r in records:
        fp = footprint(r["gt_boxes"])
        r["footprint"] = fp
        if fp is None or not r["width"] or not r["height"]:
            r["max_margin"] = np.nan
            r["footprint_frac"] = np.nan
            continue
        W, H = r["width"], r["height"]
        r["max_margin"] = float(max(fp[0] / W, fp[1] / H, (W - fp[2]) / W, (H - fp[3]) / H))
        r["footprint_frac"] = float((fp[2] - fp[0]) * (fp[3] - fp[1]) / (W * H))
    return records


def select(records, regime, max_margin):
    """Return (subset, region_per_image) for one tile regime."""
    if regime == "all":
        return records, [None] * len(records)
    if regime == "complete":
        keep = [r for r in records if np.isfinite(r["max_margin"]) and r["max_margin"] <= max_margin]
        return keep, [None] * len(keep)
    if regime == "cropped":
        keep = [r for r in records if r["footprint"] is not None]
        return keep, [r["footprint"] for r in keep]
    raise ValueError(regime)


def restrict(record, region):
    """Drop ground truth and predictions whose centroid lies outside ``region``."""
    if region is None:
        return record
    pk = inside(record["pred_boxes"], region)
    gk = inside(record["gt_boxes"], region)
    out = dict(record)
    out["pred_boxes"] = record["pred_boxes"][pk]
    out["pred_scores"] = record["pred_scores"][pk]
    out["gt_boxes"] = record["gt_boxes"][gk]
    if record["pred_masks"] is not None:
        out["pred_masks"] = record["pred_masks"][pk]
        out["gt_masks"] = record["gt_masks"][gk]
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def mask_iou(a, b):
    """IoU between two [N, H, W] / [M, H, W] boolean mask stacks.

    Goes through pycocotools' RLE IoU: the dense form allocates N x H*W floats per
    image, which is minutes and gigabytes at 1024x1024 with a few hundred instances.
    """
    if len(a) == 0 or len(b) == 0:
        return torch.zeros((len(a), len(b)))
    from pycocotools import mask as mask_utils

    def encode(stack):
        return mask_utils.encode(np.asfortranarray(
            stack.astype(np.uint8).transpose(1, 2, 0)))

    iou = mask_utils.iou(encode(a), encode(b), [0] * len(b))
    return torch.from_numpy(np.asarray(iou, dtype=np.float32)).reshape(len(a), len(b))


def recall_precision(records, iou_type, iou_threshold):
    """Image-averaged recall and *plain* precision (no tree-mask forgiveness)."""
    from torchvision.ops import box_iou

    rec_vals, prec_vals = [], []
    for r in records:
        n_gt = len(r["gt_masks"] if iou_type == "segm" else r["gt_boxes"])
        n_pred = len(r["pred_masks"] if iou_type == "segm" else r["pred_boxes"])
        if n_gt == 0:
            continue
        if n_pred == 0:
            rec_vals.append(0.0)
            prec_vals.append(0.0)
            continue
        if iou_type == "segm":
            iou = mask_iou(r["gt_masks"], r["pred_masks"])
        else:
            iou = box_iou(torch.from_numpy(r["gt_boxes"]).float(),
                          torch.from_numpy(r["pred_boxes"]).float())
        gt_to_pred = greedy_iou_match(iou, iou_threshold)
        tp = float(n_matched_gt(gt_to_pred))
        rec_vals.append(tp / n_gt)
        prec_vals.append(tp / n_pred)
    return (float(np.mean(rec_vals)) if rec_vals else float("nan"),
            float(np.mean(prec_vals)) if prec_vals else float("nan"))


def average_precision(records, iou_type, iou_threshold, max_detections=None):
    """AP at a single IoU, matching ``DetectionMAP``'s torchmetrics configuration.

    ``max_detections`` overrides the per-image detection cap. Both TreeBoxes and
    TreePolygons now use [1, 10, 1000], so that is the default here for either
    iou_type and an offline re-score matches what the dataset reports.

    Before 2026-08-26 TreeBoxes left this at torchmetrics' default (top 100 per
    image), which silently truncated dense crops: ~20-25% of test ground truth was
    unreachable and an oracle predicting GT exactly scored AP40 0.80 rather than 1.0.
    Box AP numbers produced before that date are not comparable to current ones --
    pass --max-detections 100 to reproduce them.
    """
    from milliontrees.common.metrics.all_metrics import make_mean_average_precision

    max_dets = max_detections
    if max_dets is None:
        max_dets = [1, 10, 1000]
    kwargs = dict(iou_type=iou_type, iou_thresholds=[iou_threshold], class_metrics=False)
    if max_dets is not None:
        kwargs["max_detection_thresholds"] = list(max_dets)
    metric = make_mean_average_precision(**kwargs)

    preds, targets = [], []
    for r in records:
        if iou_type == "segm":
            preds.append({"masks": torch.from_numpy(r["pred_masks"]).bool(),
                          "scores": torch.from_numpy(r["pred_scores"]).float(),
                          "labels": torch.zeros(len(r["pred_masks"]), dtype=torch.long)})
            targets.append({"masks": torch.from_numpy(r["gt_masks"]).bool(),
                            "labels": torch.zeros(len(r["gt_masks"]), dtype=torch.long)})
        else:
            preds.append({"boxes": torch.from_numpy(r["pred_boxes"]).float(),
                          "scores": torch.from_numpy(r["pred_scores"]).float(),
                          "labels": torch.zeros(len(r["pred_boxes"]), dtype=torch.long)})
            targets.append({"boxes": torch.from_numpy(r["gt_boxes"]).float(),
                            "labels": torch.zeros(len(r["gt_boxes"]), dtype=torch.long)})
    if not preds:
        return float("nan")
    metric.update(preds, targets)
    result = metric.compute()
    key = ("map_50" if iou_threshold == 0.5 and iou_type != "segm" and max_dets is None
           else "map")
    value = float(result[key])
    if value < 0:  # torchmetrics returns -1 when the requested key is undefined
        value = float(result["map"])
    return value


def score_group(records, iou_type, match_iou, max_detections=None, ap_ious=AP_IOUS):
    recall, precision = recall_precision(records, iou_type, match_iou)
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
    stats = {
        "n_images": len(records),
        "n_gt": int(sum(len(r["gt_boxes"]) for r in records)),
        "n_pred": int(sum(len(r["pred_boxes"]) for r in records)),
        "recall": recall,
        "precision_plain": precision,
        "f1_plain": f1,
    }
    for iou in ap_ious:
        stats[ap_key(iou)] = average_precision(records, iou_type, iou, max_detections)
    return stats


def macro_average(per_source_stats):
    """Average each metric over sources, the convention MillionTreesDataset.eval() uses."""
    stats = list(per_source_stats)
    out = {"n_images": sum(s["n_images"] for s in stats),
           "n_gt": sum(s["n_gt"] for s in stats),
           "n_pred": sum(s["n_pred"] for s in stats)}
    ap_keys = [k for k in (stats[0] if stats else {}) if k.startswith("AP")]
    for key in ["recall", "precision_plain", "f1_plain", *ap_keys]:
        out[key] = float(np.mean([s[key] for s in stats])) if stats else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", nargs="+", required=True,
                    help="One or more prediction dumps (.pkl).")
    ap.add_argument("--score-threshold", type=float, default=None,
                    help="Eval score threshold; default = the dump's own value or 0.1.")
    ap.add_argument("--match-iou", type=float, default=0.4,
                    help="IoU used by recall/precision (the leaderboard F1 convention).")
    ap.add_argument("--max-margin", type=float, default=0.02,
                    help="'complete' regime: max fraction of the tile on any side that may "
                         "sit outside the annotated footprint.")
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=list(REGIMES))
    ap.add_argument("--ap-ious", nargs="+", type=float, default=list(AP_IOUS),
                    help="IoU thresholds AP is computed at (one column each). Default "
                         "0.4 (the leaderboard AP) 0.5 0.6 (strict complement).")
    ap.add_argument("--max-detections", type=int, default=None,
                    help="Override the per-image detection cap used by AP (torchmetrics "
                         "max_detection_thresholds). Defaults to 1000, matching both "
                         "TreeBoxes and TreePolygons. Pass 100 to reproduce box AP "
                         "numbers from before 2026-08-26.")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    rows = []
    for path in args.predictions:
        dump = load_predictions(path)
        meta = dump["meta"]
        thr = args.score_threshold
        if thr is None:
            thr = meta.get("score_threshold")
        if thr is None:
            thr = 0.1
        records, iou_type = build_records(dump, thr)
        annotate_completeness(records)
        model = meta.get("model") or os.path.basename(path)
        print(f"\n=== {model} | {meta.get('task')} | {meta.get('eval_split')} "
              f"| score>{thr} | {len(records)} images ===")

        for regime in args.regimes:
            subset, regions = select(records, regime, args.max_margin)
            subset = [restrict(r, reg) for r, reg in zip(subset, regions)]
            if not subset:
                print(f"  [{regime}] no images pass the filter")
                continue
            max_dets = ([1, 10, args.max_detections] if args.max_detections else None)
            sources = sorted({r["source"] for r in subset if r["source"]})
            per_source = {src: score_group([r for r in subset if r["source"] == src],
                                           iou_type, args.match_iou, max_dets,
                                           args.ap_ious)
                          for src in sources}
            # "overall (macro)" is the macro-average over sources, the convention
            # MillionTreesDataset.eval() reports, so these rows line up with the
            # results_*.txt files the leaderboard is built from.
            groups = {"overall (macro)": macro_average(per_source.values()),
                      "overall (pooled)": score_group(subset, iou_type, args.match_iou,
                                                      max_dets, args.ap_ious)}
            groups.update(per_source)
            for name, stats in groups.items():
                rows.append(dict(model=model, task=meta.get("task"),
                                 eval_split=meta.get("eval_split"), regime=regime,
                                 group=name, score_threshold=thr, **stats))
                aps = "  ".join(f"{ap_key(i)}={stats[ap_key(i)]:.3f}"
                                for i in args.ap_ious)
                print(f"  [{regime:8s}] {name:<22s} n={stats['n_images']:>3d} "
                      f"gt={stats['n_gt']:>5d} pred={stats['n_pred']:>5d}  "
                      f"R={stats['recall']:.3f} P={stats['precision_plain']:.3f} "
                      f"F1={stats['f1_plain']:.3f}  {aps}")

    df = pd.DataFrame(rows)
    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"\nWrote {args.out_csv}")
    return df


if __name__ == "__main__":
    main()
