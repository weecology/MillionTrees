"""Show which predictions AP40 counts and AP60 rejects.

AP40 and AP60 score the *same* predictions in the same order; only the IoU a match
requires changes. The gap between them is therefore made of concrete boxes -- the ones
that clear IoU 0.4 against a ground-truth crown but fall short of 0.6 -- and this script
renders them so the number has a picture attached.

Every prediction above the eval score threshold is bucketed by the repo's own matcher
(``greedy_iou_match``, the one recall/precision use), run twice per image:

  survives    matched at IoU 0.6  -- counted by both AP40 and AP60
  AP40-only   matched at 0.4 but not 0.6  -- the AP40 -> AP60 drop, drawn in orange
  miss        matched at neither -- a false positive at both IoUs

Outputs a contact sheet of the worst offenders, per-crop close-ups, a matched-IoU
histogram with the 0.4 / 0.6 lines drawn on it, and a per-source CSV of bucket counts.

Usage:
    python scripts/visualize_ap40_vs_ap60.py \
        --predictions outputs/validation_preds/deepforest_finetuned_boxes_validation_v0.23.pkl \
        --image-dir /orange/ewhite/web/public/MillionTrees/TreeBoxes_v0.23/images \
        --out-dir docs/public/ap40_vs_ap60
"""

import argparse
import os
import re
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from PIL import Image

from milliontrees.common.metrics.matching import greedy_iou_match
from milliontrees.common.prediction_dump import load_predictions

# Bucket colors. Orange is reserved for the AP40-only boxes because they are the
# subject of the figure; purple matches the repo's ground-truth convention.
C_GT = (128, 60, 200)       # purple  -- ground truth
C_SURVIVES = (40, 170, 90)  # green   -- matched at IoU 0.6
C_AP40_ONLY = (255, 140, 0) # orange  -- matched at 0.4 only
C_MISS = (200, 40, 40)      # red     -- matched at neither


def _slug(name):
    return re.sub(r"[^\w]+", "_", str(name)).strip("_").lower()[:40]


@lru_cache(maxsize=48)
def load_eval_frame_image(image_dir, filename, size):
    """The dump's coordinates live in the resized frame the model saw, so the image
    has to be resized the same way before anything is drawn on it.

    Cached: a paginated crop sheet revisits the same tile dozens of times, and these
    are multi-thousand-pixel GeoTIFFs.
    """
    path = os.path.join(image_dir, filename)
    img = Image.open(path).convert("RGB").resize((size, size),
                                                 Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def bucket_predictions(gt, pred, scores, thr, iou_lo, iou_hi):
    """Classify every kept prediction as survives / ap40_only / miss.

    Both thresholds go through ``greedy_iou_match`` so the buckets agree with how the
    metrics themselves match, not with a looser per-prediction best-IoU rule.
    """
    from torchvision.ops import box_iou

    keep = scores > thr
    pred, scores = pred[keep], scores[keep]
    n_pred, n_gt = len(pred), len(gt)
    out = dict(pred=pred, scores=scores, n_gt=n_gt,
               survives=np.zeros(n_pred, bool), ap40_only=np.zeros(n_pred, bool),
               matched_iou=np.full(n_pred, np.nan), gt_of=np.full(n_pred, -1, int))
    if n_pred == 0 or n_gt == 0:
        return out

    iou = box_iou(torch.from_numpy(gt).float(), torch.from_numpy(pred).float())
    lo = greedy_iou_match(iou, iou_lo)
    hi = greedy_iou_match(iou, iou_hi)

    for gi, pj in enumerate(lo.tolist()):
        if pj >= 0:
            out["matched_iou"][pj] = float(iou[gi, pj])
            out["gt_of"][pj] = gi
            out["ap40_only"][pj] = True
    for gi, pj in enumerate(hi.tolist()):
        if pj >= 0:
            out["survives"][pj] = True
            out["ap40_only"][pj] = False
            out["matched_iou"][pj] = float(iou[gi, pj])
            out["gt_of"][pj] = gi
    return out


def draw(ax, boxes, color, lw=1.6, ls="-"):
    from matplotlib.patches import Rectangle
    for b in boxes:
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                               fill=False, edgecolor=np.array(color) / 255,
                               linewidth=lw, linestyle=ls))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--out-dir", default="docs/public/ap40_vs_ap60")
    ap.add_argument("--score-threshold", type=float, default=None)
    ap.add_argument("--iou-lo", type=float, default=0.4)
    ap.add_argument("--iou-hi", type=float, default=0.6)
    ap.add_argument("--n-crops", type=int, default=12,
                    help="Close-ups of individual AP40-only boxes. Use --all-crops to "
                         "render every one instead.")
    ap.add_argument("--all-crops", action="store_true",
                    help="Render every AP40-only box, paginated over several sheets.")
    ap.add_argument("--crops-per-sheet", type=int, default=24,
                    help="Crops per close-up sheet when paginating.")
    ap.add_argument("--source", default=None,
                    help="Restrict to one source (substring match, e.g. 'Allen'). "
                         "Bucket counts and the histogram still cover everything.")
    ap.add_argument("--n-tiles", type=int, default=6,
                    help="Whole tiles on the contact sheet.")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out_dir, exist_ok=True)
    dump = load_predictions(args.predictions)
    meta = dump["meta"]
    thr = args.score_threshold
    if thr is None:
        thr = meta.get("score_threshold") or 0.1
    size = int(meta.get("image_size") or 448)

    recs, rows = [], []
    for rec in dump["images"]:
        b = bucket_predictions(rec["true"]["geometry"], rec["pred"]["geometry"],
                               rec["pred"]["scores"], thr, args.iou_lo, args.iou_hi)
        b.update(filename=rec["filename"], source=rec["source"],
                 gt=rec["true"]["geometry"])
        recs.append(b)
        rows.append(dict(filename=rec["filename"], source=rec["source"],
                         n_gt=b["n_gt"], n_pred=len(b["pred"]),
                         survives=int(b["survives"].sum()),
                         ap40_only=int(b["ap40_only"].sum()),
                         miss=int(len(b["pred"]) - b["survives"].sum() - b["ap40_only"].sum())))

    df = pd.DataFrame(rows)
    csv = os.path.join(args.out_dir, "bucket_counts.csv")
    df.to_csv(csv, index=False)

    tot = df[["survives", "ap40_only", "miss"]].sum()
    print(f"{meta.get('model')} | {meta.get('task')} | {meta.get('eval_split')} | score>{thr}")
    print(f"  predictions: {int(tot.sum())}  survives@{args.iou_hi}: {int(tot.survives)}  "
          f"AP40-only: {int(tot.ap40_only)}  miss: {int(tot.miss)}")
    for src, g in df.groupby("source"):
        s, a = int(g.survives.sum()), int(g.ap40_only.sum())
        print(f"  {src:20s} matched@0.4={s + a:5d}  of which lost at 0.6: {a:5d} "
              f"({100 * a / max(s + a, 1):.0f}%)")

    # ---- is the AP40-only box systematically the wrong SIZE, or just offset? ----
    # area ratio pred/gt separates the two failure modes: ~1.0 means a correctly-sized
    # box in the wrong place, <1 means the model under-draws the crown.
    def area(b):
        return max(b[2] - b[0], 0) * max(b[3] - b[1], 0)

    ratios = {"ap40_only": [], "survives": []}
    for r in recs:
        for key in ratios:
            for j in np.nonzero(r[key])[0]:
                g = r["gt"][r["gt_of"][j]]
                if area(g) > 0:
                    ratios[key].append(area(r["pred"][j]) / area(g))
    for key, v in ratios.items():
        v = np.asarray(v)
        if len(v):
            print(f"  area(pred)/area(gt) [{key:9s}] median {np.median(v):.2f}  "
                  f"under-drawn (<0.8): {100 * (v < 0.8).mean():.0f}%")

    # ---- histogram of matched IoU: where the AP40 matches actually sit ----
    ious = np.concatenate([r["matched_iou"][~np.isnan(r["matched_iou"])] for r in recs])
    fig, axh = plt.subplots(figsize=(7, 3.4), dpi=160)
    axh.hist(ious, bins=np.arange(0.4, 1.001, 0.02),
             color="#8899aa", edgecolor="white", linewidth=0.4)
    axh.axvline(args.iou_hi, color=np.array(C_MISS) / 255, lw=1.8)
    axh.text(args.iou_hi + 0.006, axh.get_ylim()[1] * 0.92,
             f"IoU {args.iou_hi}\nAP60 cutoff", fontsize=8,
             color=np.array(C_MISS) / 255, va="top")
    frac = float((ious < args.iou_hi).mean())
    axh.axvspan(args.iou_lo, args.iou_hi, color=np.array(C_AP40_ONLY) / 255, alpha=0.13)
    axh.set_xlabel("IoU of each AP40-matched prediction with its ground-truth crown")
    axh.set_ylabel("predictions")
    axh.set_title(f"{frac:.0%} of AP40's matches sit below IoU {args.iou_hi}", fontsize=10)
    for sp in ("top", "right"):
        axh.spines[sp].set_visible(False)
    fig.tight_layout()
    hist_path = os.path.join(args.out_dir, "matched_iou_histogram.png")
    fig.savefig(hist_path)
    plt.close(fig)

    # ---- contact sheet: tiles with the most AP40-only boxes ----
    order = sorted(range(len(recs)), key=lambda i: -recs[i]["ap40_only"].sum())
    picks = order[:args.n_tiles]
    ncol = 3
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 5.3 * nrow), dpi=150)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, picks):
        r = recs[i]
        ax.imshow(load_eval_frame_image(args.image_dir, r["filename"], size))
        draw(ax, r["gt"], C_GT, lw=1.0, ls="--")
        draw(ax, r["pred"][r["survives"]], C_SURVIVES, lw=1.2)
        draw(ax, r["pred"][r["ap40_only"]], C_AP40_ONLY, lw=2.0)
        ax.set_title(f"{r['filename'][:34]}\n{int(r['ap40_only'].sum())} boxes lost at IoU "
                     f"{args.iou_hi} of {int(r['survives'].sum() + r['ap40_only'].sum())} matched",
                     fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{meta.get('model')} — purple dashed = ground truth · green = survives IoU "
                 f"{args.iou_hi} · orange = counted by AP40, rejected by AP60", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.subplots_adjust(hspace=0.16)
    sheet_path = os.path.join(args.out_dir, "contact_sheet.png")
    fig.savefig(sheet_path)
    plt.close(fig)

    # ---- close-ups of individual AP40-only boxes ----
    # Ordered by IoU and paginated, so each sheet spans a narrow IoU band and the
    # progression from "obviously wrong" to "looks fine by eye" is readable across
    # sheets. Sampling the lowest-IoU ones only returns a page of near-identical
    # IoU 0.40 boxes sitting on the matcher's floor.
    cands = []
    for i, r in enumerate(recs):
        if args.source and args.source.lower() not in (r["source"] or "").lower():
            continue
        for j in np.nonzero(r["ap40_only"])[0]:
            cands.append((float(r["matched_iou"][j]), i, int(j)))
    cands.sort()

    if not args.all_crops and len(cands) > args.n_crops:
        idx = np.linspace(0, len(cands) - 1, args.n_crops).round().astype(int)
        cands = [cands[k] for k in idx]

    tag = "_" + _slug(args.source) if args.source else ""
    per = args.crops_per_sheet if args.all_crops else len(cands)
    per = max(per, 1)
    pages = [cands[k:k + per] for k in range(0, len(cands), per)] or [[]]

    crop_paths = []
    for pno, page in enumerate(pages, 1):
        if not page:
            continue
        ncol = 4 if len(page) <= 12 else 6
        nrow = int(np.ceil(len(page) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.4 * nrow), dpi=170)
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for ax, (iou_v, i, j) in zip(axes, page):
            r = recs[i]
            img = load_eval_frame_image(args.image_dir, r["filename"], size)
            pb, gb = r["pred"][j], r["gt"][r["gt_of"][j]]
            cx = (min(pb[0], gb[0]) + max(pb[2], gb[2])) / 2
            cy = (min(pb[1], gb[1]) + max(pb[3], gb[3])) / 2
            half = max(max(pb[2], gb[2]) - min(pb[0], gb[0]),
                       max(pb[3], gb[3]) - min(pb[1], gb[1])) * 0.85 + 8
            x0, x1 = int(max(0, cx - half)), int(min(size, cx + half))
            y0, y1 = int(max(0, cy - half)), int(min(size, cy + half))
            ax.imshow(img[y0:y1, x0:x1])
            draw(ax, [gb - np.array([x0, y0, x0, y0])], C_GT, lw=2.0, ls="--")
            draw(ax, [pb - np.array([x0, y0, x0, y0])], C_AP40_ONLY, lw=2.0)
            ratio = ((pb[2] - pb[0]) * (pb[3] - pb[1]) /
                     max((gb[2] - gb[0]) * (gb[3] - gb[1]), 1e-6))
            ax.set_title(f"IoU {iou_v:.2f} · area {ratio:.2f}× GT\n"
                         f"score {r['scores'][j]:.2f} · {r['filename'][:20]}", fontsize=7)
            ax.axis("off")
        band = f"IoU {page[0][0]:.2f}–{page[-1][0]:.2f}"
        src_label = args.source or "all sources"
        suptitle = (f"True positive for AP40, false positive for AP60 — {src_label}"
                    f"{f'  ·  sheet {pno}/{len(pages)}' if len(pages) > 1 else ''}  ·  {band}\n"
                    "purple dashed = ground-truth crown · orange = prediction")
        fig.suptitle(suptitle, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95 if nrow > 2 else 0.90))
        name = (f"ap40_only_closeups{tag}_p{pno:02d}.png" if len(pages) > 1
                else f"ap40_only_closeups{tag}.png")
        cp = os.path.join(args.out_dir, name)
        fig.savefig(cp)
        plt.close(fig)
        crop_paths.append(cp)

    crop_path = "\n  ".join(crop_paths)
    print(f"\n  {len(cands)} close-ups over {len(crop_paths)} sheet(s)"
          f"{' for ' + args.source if args.source else ''}")

    print(f"\nWrote:\n  {sheet_path}\n  {crop_path}\n  {hist_path}\n  {csv}")


if __name__ == "__main__":
    main()
