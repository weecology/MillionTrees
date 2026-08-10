"""Before/after figure: pretrained vs count-loss-fixed TreeFormer on TreePoints.

One row per source, two columns (pretrained | fine-tuned). Every prediction is
classified the way MaskAwareKeypointPrecision does, so the picture agrees with the
reported precision:

  ground truth   purple hollow circle
  true positive  green x   (matched to a GT point within the source's radius)
  false positive red x     (unmatched, counted against precision)
  ignored        grey x    (unmatched but on tree-covered pixels -- the mask-aware
                            exemption, not counted against precision)

The match radius is the dataset's own GSD-normalized per-source threshold
(real_world_m / (gsd * native_px), scaled by image_size), not a fixed pixel distance.

Run with the frozen treeformer venv:
  .venv-treeformer/bin/python scripts/make_countfix_precision_figure.py \
      --finetuned training/points/outputs/within-distribution_896_countfix/checkpoints/<ckpt> \
      --sources "Amirkolaee et al. 2023" "Ventura et al. 2022" "NEON_points"
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
# Keep SVG text as real <text> elements (not outlined paths) so labels stay
# editable in Illustrator/Inkscape. The imshow rasters are still embedded as
# base64 PNG inside the SVG, which is what we want -- vector text, raster imagery.
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepforest import main as df_main  # noqa: E402
from milliontrees import get_dataset  # noqa: E402
from milliontrees.common.metrics.all_metrics import greedy_distance_match  # noqa: E402
from training.points.train import predict_batch  # noqa: E402

GT_C, TP_C, FP_C, IG_C = "#7B2CBF", "#2A9D3F", "#E03131", "#9AA0A6"


def load_model(checkpoint, score_thresh, radius):
    """Load either a Lightning .ckpt or a HuggingFace-format TreeFormer.

    Mirrors training/points/eval.py: load_model() resolves its argument as an HF
    repo id, so a local .ckpt path must go through load_from_checkpoint instead.
    Multi-GPU checkpoints save devices=[0,1,2] in their config and would build a
    multi-device trainer here, so a single-device override is merged over it.
    """
    if checkpoint.endswith(".ckpt") and os.path.isfile(checkpoint):
        print(f"Loading Lightning checkpoint: {checkpoint}")
        model = df_main.deepforest.load_from_checkpoint(
            checkpoint,
            weights_only=False,
            config_args={"devices": 1, "accelerator": "auto", "workers": 0},
        )
    else:
        print(f"Loading TreeFormer weights: {checkpoint}")
        model = df_main.deepforest(config_args={
            "architecture": "treeformer",
            "model": {"name": checkpoint, "revision": "main"},
        })
        model.load_model(checkpoint)
    model.model.score_thresh = score_thresh
    model.model.score_integration_radius = radius
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def predict_points(model, x, score_threshold):
    pred = predict_batch(model, x.unsqueeze(0))[0]
    pts = pred.get("y", torch.zeros((0, 2))).detach().cpu().numpy().reshape(-1, 2)
    sc = pred.get("scores")
    if sc is not None and len(pts):
        sc = sc.detach().cpu().numpy().reshape(-1)
        pts = pts[sc > score_threshold]
    return pts


def classify(gt, pred, pixel_threshold, tree_mask=None):
    """Return (tp_idx, counted_fp_idx, ignored_fp_idx) into ``pred``."""
    if len(gt) == 0 or len(pred) == 0:
        return np.array([], int), np.arange(len(pred)), np.array([], int)
    d = torch.cdist(torch.as_tensor(gt).float(), torch.as_tensor(pred).float(), p=2)
    gt_to_pred = greedy_distance_match(d, pixel_threshold)
    matched = gt_to_pred[gt_to_pred >= 0]
    tp_idx = matched.unique().long().cpu().numpy() if matched.numel() else np.array([], int)
    unmatched = np.setdiff1d(np.arange(len(pred)), tp_idx)
    if tree_mask is None or len(unmatched) == 0:
        return tp_idx, unmatched, np.array([], int)
    mask = np.asarray(tree_mask)
    if mask.ndim == 3:
        mask = mask[0] if mask.shape[0] in (1, 3) else mask[..., 0]
    h, w = mask.shape[:2]
    xy = np.round(pred[unmatched]).astype(int)
    inb = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    on_tree = np.zeros(len(unmatched), bool)
    on_tree[inb] = mask[xy[inb, 1], xy[inb, 0]] > 0
    return tp_idx, unmatched[~on_tree], unmatched[on_tree]


def to_display(x):
    img = x.detach().cpu().permute(1, 2, 0).numpy()
    if img.max() <= 1.001:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def panel(ax, img, gt, pred, tp, fp, ig, header):
    ax.imshow(img)
    if len(gt):
        ax.scatter(gt[:, 0], gt[:, 1], s=70, facecolors="none", edgecolors=GT_C,
                   linewidths=1.5)
    for idx, c in ((tp, TP_C), (ig, IG_C), (fp, FP_C)):
        if len(idx):
            ax.scatter(pred[idx, 0], pred[idx, 1], s=26, c=c, marker="x", linewidths=1.7)
    prec = len(tp) / max(len(tp) + len(fp), 1)
    ax.set_title(f"{header}\nprecision {prec:.2f}  |  {len(pred)} predicted, "
                 f"{len(gt)} GT  |  {len(fp)} false pos.", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return prec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dir", default="/orange/ewhite/web/public/MillionTrees")
    ap.add_argument("--split-scheme", default="within-distribution")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--image-size", type=int, default=896)
    ap.add_argument("--score-thresh", type=float, default=0.1)
    ap.add_argument("--score-integration-radius", type=int, default=2)
    ap.add_argument("--pretrained", default="weecology/deepforest-tree-point")
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--min-gt", type=int, default=15)
    ap.add_argument("--max-gt", type=int, default=400)
    ap.add_argument("--scan-limit", type=int, default=1600)
    ap.add_argument("--candidates", type=int, default=5,
                    help="Images considered per source; the MEDIAN-improvement one is drawn.")
    ap.add_argument("--out", default="docs/public/countfix_precision_figure.png",
                    help="Output path; format follows the extension (.png or .svg).")
    ap.add_argument("--no-suptitle", action="store_true",
                    help="Omit the figure title (for supplementary figures with a caption).")
    ap.add_argument("--cache", default=None,
                    help="Panel cache path (default: <out>.panels.npz). Written on every "
                         "run so styling can be re-rendered without redoing inference.")
    ap.add_argument("--from-cache", action="store_true",
                    help="Render from the cache and skip dataset + model loading entirely.")
    ap.add_argument("--list-sources", action="store_true")
    args = ap.parse_args()

    cache_path = args.cache or (os.path.splitext(args.out)[0] + ".panels.npz")

    # Re-rendering styling (SVG vs PNG, title on/off) costs ~10 min of CPU inference
    # otherwise, so the panel data is cached and can be replayed without any model.
    if args.from_cache:
        if not os.path.isfile(cache_path):
            raise SystemExit(f"--from-cache given but no cache at {cache_path}")
        print("rendering from cache:", cache_path)
        _, panels = load_cache(cache_path)
        render(args, panels)
        return

    ds = get_dataset("TreePoints", root_dir=args.root_dir,
                     split_scheme=args.split_scheme, image_size=args.image_size)
    subset = ds.get_subset(args.eval_split)

    if args.list_sources:
        print("sources in", args.eval_split, ":")
        for sid in sorted(ds._source_id_to_code):
            print(f"  {sid:3d}  {ds._source_id_to_code[sid]}")
        return

    # Collect candidate images per source, then pick the MEDIAN-improvement one.
    # Taking the first match makes the panel a coin flip against the source-level
    # trend (a first-match Amirkolaee image showed precision 1.00 -> 0.96 while the
    # source aggregate moves 0.782 -> 0.915); taking the best would be cherry-picking.
    candidates = {s: [] for s in args.sources}
    for i in range(min(len(subset), args.scan_limit)):
        meta, x, targets = subset[i]
        name = ds._source_id_to_code[int(meta[1].item())]
        if name not in candidates or len(candidates[name]) >= args.candidates:
            continue
        gt = targets[ds.geometry_name]
        if args.min_gt <= len(gt) <= args.max_gt:
            candidates[name].append((to_display(x), x,
                                     np.asarray(gt).reshape(-1, 2),
                                     targets.get("tree_coverage_mask")))
        if all(len(v) >= args.candidates for v in candidates.values()):
            break
    missing = [s for s in args.sources if not candidates[s]]
    if missing:
        raise SystemExit(f"no image with {args.min_gt}-{args.max_gt} points found for: "
                         f"{missing} (scanned {args.scan_limit}); use --list-sources")

    models = [("Pretrained\n(no MillionTrees training)", load_model(
                   args.pretrained, args.score_thresh, args.score_integration_radius)),
              ("Fine-tuned\n(count loss enabled)", load_model(
                   args.finetuned, args.score_thresh, args.score_integration_radius))]

    def precision_of(m, x, gt, tmask, thr):
        pred = predict_points(m, x, ds.eval_score_threshold)
        tp, fp, _ = classify(gt, pred, thr, tmask)
        return len(tp) / max(len(tp) + len(fp), 1)

    chosen = {}
    for src in args.sources:
        thr = ds._source_thresholds.get(src, ds.distance_threshold) * args.image_size
        scored = []
        for cand in candidates[src]:
            _, x, gt, tmask = cand
            d = (precision_of(models[1][1], x, gt, tmask, thr)
                 - precision_of(models[0][1], x, gt, tmask, thr))
            scored.append((d, cand))
        scored.sort(key=lambda t: t[0])
        pick = scored[len(scored) // 2]
        chosen[src] = pick[1]
        print(f"{src}: candidate deltas "
              f"{[round(d, 3) for d, _ in scored]} -> chose median {pick[0]:+.3f}")

    panels = {}
    for r, src in enumerate(args.sources):
        img, x, gt, tmask = chosen[src]
        thr = ds._source_thresholds.get(src, ds.distance_threshold) * args.image_size
        for c, (label, m) in enumerate(models):
            pred = predict_points(m, x, ds.eval_score_threshold)
            tp, fp, ig = classify(gt, pred, thr, tmask)
            panels[(r, c)] = dict(img=img, gt=gt, pred=pred, tp=tp, fp=fp, ig=ig,
                                  header=f"{src}\n{label}")
    save_cache(cache_path, args.sources, [lab for lab, _ in models], panels)
    render(args, panels)


def save_cache(path, sources, labels, panels):
    flat = {"sources": np.array(sources, dtype=object),
            "labels": np.array(labels, dtype=object)}
    for (r, c), d in panels.items():
        for k, v in d.items():
            flat[f"{r}_{c}_{k}"] = v
    np.savez_compressed(path, **flat)
    print("cached panels ->", path)


def load_cache(path):
    z = np.load(path, allow_pickle=True)
    sources = list(z["sources"])
    labels = list(z["labels"])
    panels = {}
    for r in range(len(sources)):
        for c in range(len(labels)):
            panels[(r, c)] = {k: z[f"{r}_{c}_{k}"] for k in
                              ("img", "gt", "pred", "tp", "fp", "ig")}
            panels[(r, c)]["header"] = str(z[f"{r}_{c}_header"])
    return sources, panels


def render(args, panels):
    n = max(r for r, _ in panels) + 1
    fig, axes = plt.subplots(n, 2, figsize=(10.5, 5.4 * n), squeeze=False)
    for (r, c), d in sorted(panels.items()):
        p = panel(axes[r][c], d["img"], d["gt"], d["pred"], d["tp"], d["fp"],
                  d["ig"], d["header"])
        print(f"{d['header'].splitlines()[0]:28s} "
              f"{d['header'].splitlines()[1]:12s} precision={p:.3f} "
              f"pred={len(d['pred'])} tp={len(d['tp'])} fp={len(d['fp'])} "
              f"ignored={len(d['ig'])}")

    handles = [
        mlines.Line2D([], [], color=GT_C, marker="o", markerfacecolor="none",
                      linestyle="none", markersize=9, label="ground truth"),
        mlines.Line2D([], [], color=TP_C, marker="x", linestyle="none",
                      markersize=8, label="true positive"),
        mlines.Line2D([], [], color=FP_C, marker="x", linestyle="none",
                      markersize=8, label="false positive"),
        mlines.Line2D([], [], color=IG_C, marker="x", linestyle="none",
                      markersize=8, label="ignored (tree-covered)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    if args.no_suptitle:
        fig.tight_layout(rect=[0, 0.03, 1, 1])
    else:
        fig.suptitle("TreeFormer point predictions before and after enabling the count loss",
                     fontsize=13)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
