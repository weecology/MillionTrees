"""Sweep the mask-aware precision cutoff (``tree_fraction_threshold``) on dumped predictions.

``MaskAwareDetectionPrecision`` / ``MaskAwareMaskPrecision`` drop an unmatched prediction
from the false-positive count when at least ``tree_fraction_threshold`` of its area falls on
tree pixels in ``tree_coverage_mask``. The leaderboard fixes that cutoff at **0.5**, which on
near-fully-canopied validation imagery forgives almost every false positive and inflates F1
relative to AP (see ``notes/validation_ap_completeness.md``).

This script re-scores a ``--save-predictions`` dump at many cutoffs without re-running a model:

  cutoff 0.0   every unmatched prediction is forgiven -> precision is 1 wherever anything matched
  cutoff 0.5   the leaderboard setting
  cutoff 1.0   only predictions lying *entirely* on canopy are forgiven
  cutoff >1    nothing is forgiven -> plain precision (reported as the ``none`` row)

Double detections (unmatched predictions that still exceed the IoU threshold against some
ground-truth object) are never forgiven at any cutoff, exactly as in the metric.

Outputs a tidy CSV, a two-panel figure per model (metrics vs cutoff; where the forgiven
predictions live), and optional per-image overlays showing which predictions the cutoff
forgives.

Usage:
    python scripts/sweep_maskaware_precision.py \
        --predictions outputs/validation_preds/canopyrs_boxes_validation.pkl \
        --data-dir /orange/ewhite/web/public/MillionTrees/TreeBoxes_v0.22 \
        --out-dir docs/public/maskaware_cutoff
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from milliontrees.common.metrics.matching import greedy_iou_match, n_matched_gt
from milliontrees.common.prediction_dump import decode_masks, load_predictions

# dataviz reference palette, categorical slots 1-3 (light mode).
C_PRECISION = "#2a78d6"   # blue
C_F1 = "#eb6834"          # orange
C_FORGIVEN = "#1baf7a"    # aqua
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"

# Sentinel cutoff meaning "no forgiveness at all" (plain precision).
NO_FORGIVENESS = 1.0001


# --------------------------------------------------------------------------- #
# Geometry / mask helpers
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


def mask_iou(a, b):
    """IoU between two boolean mask stacks, via pycocotools RLE (dense is GB-scale)."""
    if len(a) == 0 or len(b) == 0:
        return torch.zeros((len(a), len(b)))
    from pycocotools import mask as mask_utils

    def encode(stack):
        return mask_utils.encode(np.asfortranarray(
            stack.astype(np.uint8).transpose(1, 2, 0)))

    iou = mask_utils.iou(encode(a), encode(b), [0] * len(b))
    return torch.from_numpy(np.asarray(iou, dtype=np.float32)).reshape(len(a), len(b))


def load_tree_mask(masks_dir, filename, eval_size):
    """Load ``<masks_dir>/<stem>.png`` and resize to the evaluation frame.

    The dataset feeds the coverage mask through the same ``A.Resize`` as the image, so the
    metric sees it at ``image_size``; nearest-neighbour keeps it binary.
    """
    import cv2
    from PIL import Image

    path = Path(masks_dir) / f"{Path(filename).stem}.png"
    if not path.exists():
        raise FileNotFoundError(f"Missing tree coverage mask: {path}")
    mask = np.array(Image.open(path).convert("L"), dtype=np.uint8) > 0
    if eval_size and mask.shape != tuple(eval_size):
        mask = cv2.resize(mask.astype(np.uint8), (int(eval_size[1]), int(eval_size[0])),
                          interpolation=cv2.INTER_NEAREST) > 0
    return mask


def box_tree_fraction(box, tree_mask):
    """Fraction of a box's pixels that are canopy -- mirrors ``_tree_pixel_fraction``."""
    h, w = tree_mask.shape
    x1 = min(max(int(np.floor(box[0])), 0), w)
    y1 = min(max(int(np.floor(box[1])), 0), h)
    x2 = min(max(int(np.ceil(box[2])), 0), w)
    y2 = min(max(int(np.ceil(box[3])), 0), h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(tree_mask[y1:y2, x1:x2].mean())


def mask_tree_fraction(pred_mask, tree_mask):
    """Fraction of a predicted mask's pixels that are canopy -- ``_mask_tree_fraction``."""
    area = pred_mask.sum()
    if area == 0:
        return 0.0
    return float((pred_mask & tree_mask).sum() / area)


# --------------------------------------------------------------------------- #
# Per-image analysis: everything the sweep needs, computed once
# --------------------------------------------------------------------------- #
def analyze_image(rec, iou_type, score_threshold, masks_dir, eval_size, iou_threshold):
    """Match one image and record the canopy fraction of every unmatched prediction."""
    pred, true = rec["pred"], rec["true"]
    scores = pred.get("scores")
    scores = np.zeros(0, dtype=np.float32) if scores is None else np.asarray(scores)

    if iou_type == "segm":
        pred_masks = decode_masks(pred["masks"])
        gt_masks = decode_masks(true["masks"])
        keep = scores > score_threshold if len(scores) else np.zeros(len(pred_masks), bool)
        pred_masks, scores = pred_masks[keep], scores[keep]
        pred_boxes, gt_boxes = boxes_from_masks(pred_masks), boxes_from_masks(gt_masks)
        frame = gt_masks.shape[1:] if len(gt_masks) else pred_masks.shape[1:]
        iou = mask_iou(gt_masks, pred_masks)
    else:
        pred_boxes = np.asarray(pred["geometry"], dtype=np.float32)
        gt_boxes = np.asarray(true["geometry"], dtype=np.float32)
        keep = scores > score_threshold if len(scores) else np.zeros(len(pred_boxes), bool)
        pred_boxes, scores = pred_boxes[keep], scores[keep]
        pred_masks = gt_masks = None
        frame = (int(eval_size), int(eval_size)) if eval_size else None
        from torchvision.ops import box_iou as tv_box_iou
        iou = (tv_box_iou(torch.from_numpy(gt_boxes).float(),
                          torch.from_numpy(pred_boxes).float())
               if len(gt_boxes) and len(pred_boxes) else
               torch.zeros((len(gt_boxes), len(pred_boxes))))

    n_gt, n_pred = len(gt_boxes), len(pred_boxes)
    tree_mask = load_tree_mask(masks_dir, rec["filename"], frame) if n_pred else None

    out = dict(filename=rec["filename"], source=rec["source"], n_gt=n_gt, n_pred=n_pred,
               tp=0, unmatched=[], pred_boxes=pred_boxes, gt_boxes=gt_boxes,
               pred_masks=pred_masks, gt_masks=gt_masks, scores=scores,
               tree_cover=float(tree_mask.mean()) if tree_mask is not None else np.nan)
    if n_pred == 0 or n_gt == 0:
        # Matches the metric's degenerate branches; an image with no GT contributes
        # only through the "all predictions unmatched" path.
        if n_gt == 0 and n_pred:
            out["unmatched"] = [
                dict(idx=i,
                     frac=(box_tree_fraction(pred_boxes[i], tree_mask) if iou_type != "segm"
                           else mask_tree_fraction(pred_masks[i], tree_mask)),
                     double=False)
                for i in range(n_pred)]
        return out

    gt_to_pred = greedy_iou_match(iou, iou_threshold)
    out["tp"] = int(n_matched_gt(gt_to_pred))
    matched = gt_to_pred[gt_to_pred >= 0].unique()
    unmatched_mask = np.ones(n_pred, dtype=bool)
    if matched.numel():
        unmatched_mask[matched.long().numpy()] = False

    for i in np.nonzero(unmatched_mask)[0]:
        # A prediction that still clears the IoU bar against some GT is a double
        # detection: the metric never forgives it, at any cutoff.
        double = bool(iou[:, i].max().item() > iou_threshold) if iou.numel() else False
        frac = (box_tree_fraction(pred_boxes[i], tree_mask) if iou_type != "segm"
                else mask_tree_fraction(pred_masks[i], tree_mask))
        out["unmatched"].append(dict(idx=int(i), frac=float(frac), double=double))
    return out


def image_precision(img, cutoff):
    """Mask-aware precision for one image at one cutoff, mirroring the metric's branches."""
    if img["n_pred"] == 0:
        return 0.0 if img["n_gt"] > 0 else 1.0
    charged = sum(1 for u in img["unmatched"]
                  if u["double"] or u["frac"] < cutoff)
    if img["n_gt"] == 0:
        return 1.0 if charged == 0 else 0.0
    denom = img["tp"] + charged
    return 1.0 if denom == 0 else img["tp"] / denom


def image_recall(img):
    if img["n_gt"] == 0:
        return np.nan
    return img["tp"] / img["n_gt"]


# --------------------------------------------------------------------------- #
# Aggregation -- image mean within source, then macro over sources (dataset.eval())
# --------------------------------------------------------------------------- #
def aggregate(images, cutoff):
    by_source = {}
    for img in images:
        by_source.setdefault(img["source"], []).append(img)
    rows = []
    for source, imgs in sorted(by_source.items()):
        prec = float(np.mean([image_precision(i, cutoff) for i in imgs]))
        rec = [image_recall(i) for i in imgs]
        rec = float(np.nanmean(rec)) if np.any(np.isfinite(rec)) else np.nan
        unmatched = [u for i in imgs for u in i["unmatched"]]
        forgiven = sum(1 for u in unmatched if not u["double"] and u["frac"] >= cutoff)
        rows.append(dict(group=source, n_images=len(imgs),
                         n_gt=sum(i["n_gt"] for i in imgs),
                         n_pred=sum(i["n_pred"] for i in imgs),
                         n_unmatched=len(unmatched), n_forgiven=forgiven,
                         recall=rec, precision=prec,
                         f1=f1(rec, prec)))
    macro = dict(group="overall (macro)",
                 n_images=sum(r["n_images"] for r in rows),
                 n_gt=sum(r["n_gt"] for r in rows),
                 n_pred=sum(r["n_pred"] for r in rows),
                 n_unmatched=sum(r["n_unmatched"] for r in rows),
                 n_forgiven=sum(r["n_forgiven"] for r in rows),
                 recall=float(np.mean([r["recall"] for r in rows])),
                 precision=float(np.mean([r["precision"] for r in rows])))
    macro["f1"] = f1(macro["recall"], macro["precision"])
    return [macro] + rows


def f1(recall, precision):
    if not np.isfinite(recall) or not np.isfinite(precision) or (recall + precision) == 0:
        return float("nan")
    return 2 * recall * precision / (recall + precision)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def style_axes(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)


def sweep_figure(model, df, images, out_path, leaderboard_cutoff=0.5):
    """Two panels: what the cutoff does to the score, and where the forgiven FPs sit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    macro = df[df["group"] == "overall (macro)"].sort_values("cutoff")
    swept = macro[macro["cutoff"] <= 1.0]
    plain = macro[macro["cutoff"] > 1.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")
    style_axes(ax1)
    style_axes(ax2)

    x = swept["cutoff"].to_numpy()
    x_off = 1.16  # "forgiveness disabled" sits off the end of the sweep, on its own tick
    ax1.plot(x, swept["precision"], color=C_PRECISION, linewidth=2, label="Mask-aware precision")
    ax1.plot(x, swept["f1"], color=C_F1, linewidth=2, label="F1")
    ax1.plot(x, swept["recall"], color=INK_MUTED, linewidth=1.4, linestyle=(0, (4, 3)),
             label="Recall (cutoff-independent)")
    if len(plain):
        for col, color in (("precision", C_PRECISION), ("f1", C_F1)):
            y0, y1 = swept[col].iloc[-1], plain[col].iloc[0]
            ax1.plot([1.0, x_off], [y0, y1], color=color, linewidth=1.2,
                     linestyle=(0, (2, 2)), alpha=0.8)
            ax1.scatter([x_off], [y1], s=40, facecolor="white", edgecolor=color,
                        linewidth=2, zorder=5)
        ax1.annotate(f"forgiveness off:\nP={plain['precision'].iloc[0]:.3f}  "
                     f"F1={plain['f1'].iloc[0]:.3f}",
                     xy=(x_off, plain["f1"].iloc[0]), xytext=(-6, 14),
                     textcoords="offset points", fontsize=8.5, color=INK_MUTED, ha="right")

    ax1.axvline(leaderboard_cutoff, color=INK_MUTED, linewidth=1, alpha=0.5)
    lead = macro[np.isclose(macro["cutoff"], leaderboard_cutoff)]
    if len(lead):
        ax1.annotate(f"leaderboard cutoff 0.5\nP={lead['precision'].iloc[0]:.3f}  "
                     f"F1={lead['f1'].iloc[0]:.3f}",
                     xy=(leaderboard_cutoff, 0.99), xytext=(6, -2),
                     textcoords="offset points", fontsize=8.5, color=INK_MUTED,
                     ha="left", va="top")
    ax1.set_ylim(0, 1.02)
    ax1.set_xlim(-0.01, 1.24)
    ax1.set_xticks(list(np.linspace(0, 1, 6)) + [x_off])
    ax1.set_xticklabels([f"{t:.1f}" for t in np.linspace(0, 1, 6)] + ["off"])
    ax1.set_xlabel("tree_fraction_threshold (canopy fraction required to forgive a false positive)",
                   fontsize=9, color=INK_MUTED)
    ax1.set_ylabel("score", fontsize=9, color=INK_MUTED)
    ax1.set_title(f"{model} — precision and F1 vs the mask-aware cutoff",
                  fontsize=11, color=INK, loc="left", pad=10)
    ax1.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, loc="lower left")

    # Panel 2: the ECDF of unmatched-prediction canopy fraction *is* the sensitivity
    # curve -- the share forgiven at cutoff t is the fraction of mass at or above t.
    fracs = np.array([u["frac"] for i in images for u in i["unmatched"] if not u["double"]])
    n_double = sum(1 for i in images for u in i["unmatched"] if u["double"])
    grid = np.linspace(0, 1, 201)
    share = np.array([(fracs >= t).mean() for t in grid]) if len(fracs) else np.zeros_like(grid)
    ax2.fill_between(grid, 0, 100 * share, color=C_FORGIVEN, alpha=0.18, linewidth=0)
    ax2.plot(grid, 100 * share, color=C_FORGIVEN, linewidth=2)
    ax2.axvline(leaderboard_cutoff, color=INK_MUTED, linewidth=1, alpha=0.5)
    if len(fracs):
        at_half = 100 * (fracs >= leaderboard_cutoff).mean()
        at_one = 100 * (fracs >= 1.0).mean()
        ax2.scatter([leaderboard_cutoff], [at_half], s=40, color=C_FORGIVEN, zorder=5)
        ax2.annotate(f"{at_half:.0f}% forgiven at the\nleaderboard cutoff 0.5",
                     xy=(leaderboard_cutoff, at_half), xytext=(-10, -58),
                     textcoords="offset points", fontsize=9, color=INK_MUTED, ha="right")
        ax2.scatter([1.0], [at_one], s=40, color=C_FORGIVEN, zorder=5)
        ax2.annotate(f"{at_one:.0f}% sit on 100% canopy —\nno cutoff below 1.0 charges them",
                     xy=(1.0, at_one), xytext=(-8, -46), textcoords="offset points",
                     fontsize=9, color=INK_MUTED, ha="right")
    ax2.set_ylim(0, 102)
    ax2.set_xlim(-0.01, 1.01)
    ax2.set_xlabel("canopy fraction of the unmatched prediction", fontsize=9, color=INK_MUTED)
    ax2.set_ylabel("% of unmatched predictions forgiven", fontsize=9, color=INK_MUTED)
    ax2.set_title(f"Canopy cover of the {len(fracs)} forgivable false positives\n"
                  f"({n_double} double detections are never forgiven)",
                  fontsize=11, color=INK, loc="left", pad=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor="white")
    plt.close(fig)
    return out_path


def combined_figure(all_df, out_path, leaderboard_cutoff=0.5):
    """F1 vs cutoff for every model in one axes -- ranking stability at a glance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
               "#4a3aa7", "#e34948"]
    macro = all_df[all_df["group"] == "overall (macro)"]
    x_off = 1.16
    fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor="white")
    style_axes(ax)
    labels = []
    for i, (model, sub) in enumerate(macro.groupby("model", sort=False)):
        sub = sub.sort_values("cutoff")
        swept, plain = sub[sub["cutoff"] <= 1.0], sub[sub["cutoff"] > 1.0]
        color = palette[i % len(palette)]
        ax.plot(swept["cutoff"], swept["f1"], color=color, linewidth=2, label=model)
        y_end = swept["f1"].iloc[-1]
        if len(plain):
            y_off = plain["f1"].iloc[0]
            ax.plot([1.0, x_off], [y_end, y_off], color=color, linewidth=1.2,
                    linestyle=(0, (2, 2)), alpha=0.8)
            ax.scatter([x_off], [y_off], s=38, facecolor="white", edgecolor=color,
                       linewidth=2, zorder=5)
            labels.append((y_off, model))
    # Direct labels at the "off" end, nudged apart so neighbouring models stay readable.
    labels.sort()
    last = -1.0
    for y, model in labels:
        y = max(y, last + 0.045)
        last = y
        ax.annotate(model, xy=(x_off + 0.02, y), fontsize=8.5, color=INK_MUTED, va="center")
    ax.axvline(leaderboard_cutoff, color=INK_MUTED, linewidth=1, alpha=0.5)
    ax.annotate("leaderboard\ncutoff 0.5", xy=(leaderboard_cutoff, 0.99), xytext=(6, -2),
                textcoords="offset points", fontsize=8.5, color=INK_MUTED, va="top")
    ax.set_xlim(-0.01, 1.62)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(list(np.linspace(0, 1, 6)) + [x_off])
    ax.set_xticklabels([f"{t:.1f}" for t in np.linspace(0, 1, 6)] + ["off"])
    ax.set_xlabel("tree_fraction_threshold", fontsize=9, color=INK_MUTED)
    ax.set_ylabel("F1 (recall × mask-aware precision)", fontsize=9, color=INK_MUTED)
    ax.set_title("F1 barely moves across the cutoff, then drops when forgiveness is off",
                 fontsize=11, color=INK, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_MUTED, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor="white")
    plt.close(fig)
    return out_path


def draw_mask_outline(ax, mask, color, linewidth, linestyle):
    """Trace an instance mask's boundary so polygon runs are drawn as shapes, not boxes."""
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if len(c) < 3:
            continue
        pts = c.squeeze(1)
        ax.plot(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]),
                color=color, linewidth=linewidth, linestyle=linestyle)


def overlay_figure(img, images_dir, masks_dir, cutoffs, out_path, iou_type, eval_size):
    """One image: canopy mask, ground truth, and which false positives each cutoff forgives."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    import cv2
    from PIL import Image

    path = Path(images_dir) / img["filename"]
    arr = np.array(Image.open(path).convert("RGB"))
    n = len(img["pred_boxes"])
    frame = (img["gt_masks"].shape[1:] if iou_type == "segm" and img["gt_masks"] is not None
             else (int(eval_size), int(eval_size)))
    arr = cv2.resize(arr, (int(frame[1]), int(frame[0])), interpolation=cv2.INTER_AREA)
    tree_mask = load_tree_mask(masks_dir, img["filename"], frame)

    fig, axes = plt.subplots(1, len(cutoffs), figsize=(4.3 * len(cutoffs), 5.3),
                             facecolor="white")
    axes = np.atleast_1d(axes)
    unmatched_by_idx = {u["idx"]: u for u in img["unmatched"]}
    for ax, cutoff in zip(axes, cutoffs):
        ax.imshow(arr)
        # Wash out the non-canopy pixels: forgiveness is only available inside the
        # unwashed region, so the eye can see how much of the tile qualifies.
        wash = np.zeros((*tree_mask.shape, 4))
        wash[..., :3] = 1.0
        wash[..., 3] = np.where(tree_mask, 0.0, 0.55)
        ax.imshow(wash)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)
        # Polygon runs are drawn as mask outlines; box runs as rectangles.
        segm = iou_type == "segm"
        for j, b in enumerate(img["gt_boxes"]):
            if segm:
                draw_mask_outline(ax, img["gt_masks"][j], "white", 1.4, "-")
            else:
                ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                       fill=False, edgecolor="white", linewidth=1.6,
                                       alpha=0.9))
        charged = 0
        for i in range(n):
            b = img["pred_boxes"][i]
            u = unmatched_by_idx.get(i)
            if u is None:
                color, style, lw = C_PRECISION, "-", 1.6      # matched a ground-truth tree
            elif not u["double"] and u["frac"] >= cutoff:
                color, style, lw = C_FORGIVEN, (0, (3, 2)), 1.6  # forgiven: on canopy
            else:
                color, style, lw = C_F1, "-", 2.0             # charged as a false positive
                charged += 1
            if segm:
                draw_mask_outline(ax, img["pred_masks"][i], color, lw, style)
            else:
                ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                                       edgecolor=color, linewidth=lw, linestyle=style))
        prec = image_precision(img, cutoff)
        label = "no forgiveness" if cutoff > 1.0 else f"cutoff {cutoff:.2f}"
        ax.set_title(f"{label} — precision {prec:.2f}\n"
                     f"{img['tp']} matched · {charged} charged FP · "
                     f"{len(img['unmatched']) - charged} forgiven",
                     fontsize=10, color=INK, loc="left", pad=8)

    handles = [Patch(facecolor="#4a4a4a", edgecolor="white", linewidth=1.6,
                     label="ground truth"),
               Patch(facecolor="none", edgecolor=C_PRECISION, linewidth=1.6,
                     label="matched prediction"),
               Patch(facecolor="none", edgecolor=C_FORGIVEN, linewidth=1.6,
                     label="forgiven (on canopy)"),
               Patch(facecolor="none", edgecolor=C_F1, linewidth=1.6,
                     label="charged false positive")]
    fig.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK_MUTED,
               loc="lower left", bbox_to_anchor=(0.01, 0.005), ncol=4)
    fig.suptitle(f"{img['filename']} — {img['source']} "
                 f"({100 * img['tree_cover']:.0f}% canopy pixels; washed-out area is non-canopy)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.09, wspace=0.05)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor="white")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", nargs="+", required=True, help="Prediction dumps (.pkl).")
    ap.add_argument("--data-dir", required=True,
                    help="Packaged dataset dir holding masks/ and images/ for the dump's split.")
    ap.add_argument("--cutoffs", nargs="+", type=float, default=None,
                    help="Cutoffs to sweep (default 0.0..1.0 by 0.05, plus no-forgiveness).")
    ap.add_argument("--match-iou", type=float, default=0.4,
                    help="IoU used for matching (the metric's iou_threshold).")
    ap.add_argument("--score-threshold", type=float, default=None,
                    help="Override the dump's own eval score threshold.")
    ap.add_argument("--out-dir", default="docs/public/maskaware_cutoff")
    ap.add_argument("--viz-n", type=int, default=3,
                    help="Per-model overlay images to render (highest-canopy first).")
    ap.add_argument("--viz-cutoffs", nargs="+", type=float, default=[0.5, 0.9, NO_FORGIVENESS])
    args = ap.parse_args()

    cutoffs = args.cutoffs or list(np.round(np.arange(0, 1.0001, 0.05), 4)) + [NO_FORGIVENESS]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = Path(args.data_dir) / "masks"
    images_dir = Path(args.data_dir) / "images"

    all_rows = []
    for path in args.predictions:
        dump = load_predictions(path)
        meta = dump["meta"]
        iou_type = meta["iou_type"]
        thr = args.score_threshold
        if thr is None:
            thr = meta.get("score_threshold")
        if thr is None:
            thr = 0.1
        model = meta.get("model") or Path(path).stem
        tag = Path(path).stem
        print(f"\n=== {model} | {meta.get('task')} | {meta.get('eval_split')} | "
              f"score>{thr} | {len(dump['images'])} images ===")

        images = [analyze_image(rec, iou_type, thr, masks_dir, meta.get("image_size"),
                                args.match_iou)
                  for rec in dump["images"]]

        rows = []
        for cutoff in cutoffs:
            for row in aggregate(images, cutoff):
                rows.append(dict(model=model, task=meta.get("task"),
                                 eval_split=meta.get("eval_split"),
                                 score_threshold=thr, cutoff=cutoff, **row))
        df = pd.DataFrame(rows)
        all_rows.extend(rows)

        macro = df[df["group"] == "overall (macro)"].sort_values("cutoff")
        for _, r in macro.iterrows():
            label = "none " if r["cutoff"] > 1.0 else f"{r['cutoff']:.2f} "
            print(f"  cutoff {label} R={r['recall']:.3f} P={r['precision']:.3f} "
                  f"F1={r['f1']:.3f}  forgiven {r['n_forgiven']:>5d}/{r['n_unmatched']:<5d}")

        fig_path = sweep_figure(model, df, images, out_dir / f"{tag}_cutoff_sweep.png")
        print(f"  wrote {fig_path}")

        # Show a spread of canopy cover (most-, median-, least-canopied tile with
        # predictions), so the overlays cover both ends of the forgiveness regime.
        ranked = sorted([i for i in images if i["n_pred"] and i["n_gt"]
                         and np.isfinite(i["tree_cover"])],
                        key=lambda i: -i["tree_cover"])
        if ranked and args.viz_n:
            picks = [ranked[int(round(q * (len(ranked) - 1)))]
                     for q in np.linspace(0, 1, min(args.viz_n, len(ranked)))]
            for img in picks:
                p = overlay_figure(img, images_dir, masks_dir, args.viz_cutoffs,
                                   out_dir / f"{tag}_overlay_{Path(img['filename']).stem}.png",
                                   iou_type, meta.get("image_size"))
                print(f"  wrote {p}")

    all_df = pd.DataFrame(all_rows)
    csv_path = out_dir / "maskaware_cutoff_sweep.csv"
    all_df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")
    if all_df["model"].nunique() > 1:
        print(f"Wrote {combined_figure(all_df, out_dir / 'maskaware_cutoff_f1_all_models.png')}")
    return all_df


if __name__ == "__main__":
    main()
