"""Three-regime figure: pretrained vs fine-tuned TreeFormer as the evaluation moves away
from the training distribution.

One row per evaluation regime, two columns (pretrained | fine-tuned):

  1. within-distribution test  -- same sources as training; fine-tuning helps
  2. out-of-distribution test  -- held-out aerial sources; the fine-tune over-predicts
  3. held-out TLS validation   -- different reference geometry; the fine-tune collapses

This is the picture behind the "TreeFormer fine-tunes well and generalizes poorly" section
of docs/leaderboard.md and the three-evaluation table in docs/leaderboard_validation.md.
Rows 2 and 3 both score the *out-of-distribution* fine-tuned checkpoint, which is the model
those leaderboard rows report; row 1 scores the within-distribution checkpoint.

Predictions are classified exactly the way MaskAwareKeypointPrecision does, so panel
precision agrees with the reported metric rather than being a generic dot plot:

  ground truth   purple hollow circle
  true positive  green x   (matched to a GT point within the source's radius)
  false positive red x     (unmatched, counted against precision)
  ignored        grey x    (unmatched but on tree-covered pixels -- the mask-aware
                            exemption, not counted against precision)

The match radius is the dataset's own GSD-normalized per-source threshold
(real_world_m / (gsd * native_px), scaled by image_size), not a fixed pixel distance.

A single 896-px tile is an illustration, not evidence: mask-aware precision in particular
swings hard tile to tile on closed-canopy sources, where unmatched predictions land on
tree-covered pixels and are exempted. Each row therefore carries its **source-level
aggregate** (recall / precision / counting MAE for both models), parsed straight out of the
eval result files listed in ROWS so the figure cannot drift from the leaderboard.

Image choice follows from that: of ``--candidates`` images from the row's source, the one
drawn is the one whose per-tile recall and precision sit closest to that published
aggregate for BOTH models, so no panel contradicts the numbers printed above it. Source
choice per row is a deliberate, documented argument -- see ROWS.

Run with the frozen treeformer venv (the shared .venv is on the polygon branch and
silently collapses TreeFormer point inference):

  .venv-treeformer/bin/python scripts/make_generalization_figure.py \
      --out docs/public/treeformer_generalization_figure.png

Re-style without re-running any inference:

  .venv-treeformer/bin/python scripts/make_generalization_figure.py --from-cache \
      --out docs/public/treeformer_generalization_figure.svg
"""
import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
# Keep SVG text as real <text> elements (not outlined paths) so labels stay editable in
# Illustrator/Inkscape. The imshow rasters are still embedded as base64 PNG inside the
# SVG, which is what we want -- vector text, raster imagery.
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from milliontrees import get_dataset  # noqa: E402

# Shared with the count-loss before/after figure: same model loading, same prediction
# filtering, same mask-aware classification, so the two figures cannot drift apart.
from make_countfix_precision_figure import (  # noqa: E402
    FP_C, GT_C, IG_C, TP_C, classify, load_model, predict_points, to_display,
)

CKPT_WITHIN = ("training/points/outputs/within-distribution_896_countfix/checkpoints/"
               "treeformer-epoch=08-val_loss=1.7299.ckpt")
CKPT_OOD = ("training/points/outputs/out-of-distribution_896_countfix/checkpoints/"
            "treeformer-epoch=17-val_loss=2.5534.ckpt")
# NOTE: both countfix dirs also hold a *cancelled* count-blind checkpoint from job 38977184
# (val_loss 0.3730 / 0.5028). Those are a different model. The countfix ones carry the
# larger val_loss (~1.7 / ~2.6) because density_l1_weight rose 0.01 -> 0.05 and the count
# term is live. Never glob these directories.

ROWS = [
    dict(
        key="within",
        title="Within-distribution test",
        subtitle="test images from sources the model trained on",
        split_scheme="within-distribution",
        eval_split="test",
        # Amirkolaee is the source where the aggregate within-distribution result is most
        # visible in a single image: fine-tuning raises keypoint accuracy 0.308 -> 0.609
        # AND precision 0.782 -> 0.915 (counting MAE 111.8 -> 43.4), i.e. it improves on
        # both axes the way the split aggregate does (F1 0.726 -> 0.782, nMAE 1.611 -> 0.214).
        source="Amirkolaee et al. 2023",
        checkpoint=CKPT_WITHIN,
        results_pretrained=("existing_models/treeformer/outputs/within-distribution_896_isolated/"
                            "results_points_within-distribution.txt"),
        results_finetuned=("training/points/outputs/within-distribution_896_countfix/"
                           "results_within-distribution.txt"),
    ),
    dict(
        key="ood",
        title="Out-of-distribution test",
        subtitle="held-out aerial sources the model never saw",
        split_scheme="out-of-distribution",
        eval_split="test",
        # OFO is 1387 of the 3260 out-of-distribution test images and is what drives the
        # aggregate precision drop (0.737 -> 0.662): on this source the fine-tune finds
        # nearly every tree (keypoint accuracy 0.976) while precision falls 0.440 -> 0.180,
        # which is over-prediction, not a detection failure.
        source="OFO field 2025",
        checkpoint=CKPT_OOD,
        results_pretrained=("existing_models/treeformer/outputs/out-of-distribution_896_isolated/"
                            "results_points_out-of-distribution.txt"),
        results_finetuned=("training/points/outputs/out-of-distribution_896_countfix/"
                           "results_out-of-distribution.txt"),
    ),
    dict(
        key="validation",
        title="Held-out TLS validation",
        subtitle="different reference geometry, never in any train or test split",
        split_scheme="out-of-distribution",
        eval_split="validation",
        # Allen carries the counting collapse: MAE 46.9 -> 136.5, nMAE 2.035, slope -0.309
        # (per-image predicted counts anti-correlated with truth). Frey degrades too
        # (keypoint 0.270 -> 0.180), so this is not an Allen tiling artifact.
        source="Allen et al. 2025",
        checkpoint=CKPT_OOD,
        results_pretrained=("existing_models/treeformer/outputs/validation_896_isolated/"
                            "results_points_out-of-distribution.txt"),
        results_finetuned=("training/points/outputs/out-of-distribution_896_countfix/"
                           "results_out-of-distribution_validation.txt"),
    ),
]

PRETRAINED_LABEL = "Pretrained\n(no MillionTrees training)"
FINETUNED_LABEL = "Fine-tuned\n(MillionTrees, count loss enabled)"


def parse_source_stats(path, source):
    """Pull one source's aggregate metrics out of an eval ``results_*.txt``.

    The eval writer emits three per-source line shapes, all keyed by the source name:

      source:8  [OFO field 2025]  [n = 1387]  [threshold = ...]:\tkeypoint_acc = 0.976
      source_id = OFO field 2025  [n = 1387]:\tmaskaware_keypoint_precision = 0.180
      [Amirkolaee et al. 2023]  [n = 167]:  MAE = 43.45  nMAE = 0.299  R2 = ...

    The MAE line exists only for completely-annotated sources, so ``mae`` may be None.
    Missing recall/precision is an error, not a None: it means the row points at a result
    file that never scored this source, and a figure built on that would be wrong.
    """
    with open(path) as fh:
        text = fh.read()
    s = re.escape(source)

    def grab(pattern, required=True):
        m = re.search(pattern, text)
        if m:
            return float(m.group(1))
        if required:
            raise SystemExit(f"{path}: no '{pattern}' for source {source!r}")
        return None

    return dict(
        recall=grab(rf"\[{s}\].*?keypoint_acc = ([-\d.]+)"),
        precision=grab(rf"source_id = {s}\s+\[n =.*?maskaware_keypoint_precision = ([-\d.]+)"),
        mae=grab(rf"\[{s}\]\s+\[n =\s*\d+\]:\s+MAE = ([-\d.]+)", required=False),
        n=int(re.search(rf"\[{s}\]\s+\[n =\s*(\d+)\]", text).group(1)),
    )


def aggregate_line(row):
    """One-line source-level summary, pretrained -> fine-tuned, for the row header."""
    pre = parse_source_stats(row["results_pretrained"], row["source"])
    fine = parse_source_stats(row["results_finetuned"], row["source"])
    parts = [f"recall {pre['recall']:.2f} → {fine['recall']:.2f}",
             f"precision {pre['precision']:.2f} → {fine['precision']:.2f}"]
    if pre["mae"] is not None and fine["mae"] is not None:
        parts.append(f"counting MAE {pre['mae']:.0f} → {fine['mae']:.0f}")
    return (f"{row['source']} ({pre['n']} images) — source aggregate: "
            + ",  ".join(parts))


def scores(gt, tp, fp):
    """Recall and mask-aware precision for one panel, as the metrics define them.

    greedy_distance_match is 1:1, so each true positive is one matched ground-truth point
    and recall is len(tp)/len(gt). ``fp`` excludes the mask-aware ignored predictions, so
    precision here is the same quantity the leaderboard reports.
    """
    recall = len(tp) / max(len(gt), 1)
    precision = len(tp) / max(len(tp) + len(fp), 1)
    return recall, precision


def collect_candidates(ds, subset, source, n, min_gt, max_gt, scan_limit):
    """Return up to ``n`` (display, tensor, gt, tree_mask) tuples for one source.

    Candidate positions come from ``metadata_array`` (source id is column 1), so only
    images of the requested source are decoded. Scanning the subset in order instead
    would load thousands of 896-px tiles to find a handful -- minutes per row.
    """
    code_to_id = {v: k for k, v in ds._source_id_to_code.items()}
    if source not in code_to_id:
        raise SystemExit(f"source {source!r} not in this dataset; use --list-sources")
    source_ids = subset.metadata_array[:, 1]
    positions = (source_ids == code_to_id[source]).nonzero().flatten().tolist()
    print(f"  {len(positions)} {source} images in this split")

    out = []
    for pos in positions[:scan_limit]:
        _, x, targets = subset[pos]
        gt = np.asarray(targets[ds.geometry_name]).reshape(-1, 2)
        if min_gt <= len(gt) <= max_gt:
            out.append((to_display(x), x, gt, targets.get("tree_coverage_mask")))
        if len(out) >= n:
            break
    return out


def build_row(row, args, pretrained):
    """Run both models on one regime and return its two panels."""
    ds = get_dataset("TreePoints", version=args.version, root_dir=args.root_dir,
                     split_scheme=row["split_scheme"], image_size=args.image_size)
    if ds.version != args.version:
        raise SystemExit(f"asked for TreePoints v{args.version}, got v{ds.version}")
    subset = ds.get_subset(row["eval_split"])
    print(f"\n=== {row['title']}: {row['split_scheme']} / {row['eval_split']} "
          f"({len(subset)} images), source {row['source']}")

    if args.list_sources:
        for sid in sorted(ds._source_id_to_code):
            print(f"  {sid:3d}  {ds._source_id_to_code[sid]}")
        return None

    candidates = collect_candidates(ds, subset, row["source"], args.candidates,
                                    args.min_gt, args.max_gt, args.scan_limit)
    if not candidates:
        raise SystemExit(f"no {row['source']} image with {args.min_gt}-{args.max_gt} "
                         f"points in {row['eval_split']} (scanned {args.scan_limit}); "
                         f"re-run with --list-sources")

    finetuned = load_model(row["checkpoint"], args.score_thresh,
                           args.score_integration_radius)
    thr = ds._source_thresholds.get(row["source"], ds.distance_threshold) * args.image_size

    def evaluate(model, x, gt, tmask):
        pred = predict_points(model, x, ds.eval_score_threshold)
        tp, fp, ig = classify(gt, pred, thr, tmask)
        return pred, tp, fp, ig

    # Most-representative pick: the candidate whose per-tile recall and precision sit
    # closest to this source's published aggregate, for BOTH models at once.
    #
    # The obvious alternatives are both worse. Taking the first match is a coin flip
    # against the source trend, and taking the median by F1 change still lets a tile
    # through whose precision contradicts the aggregate printed above it -- on
    # closed-canopy sources the mask-aware rule exempts nearly every unmatched prediction,
    # so single tiles read 1.00 against a source aggregate of 0.54. Selecting on agreement
    # with the aggregate is not cherry-picking: it cannot make the fine-tune look better or
    # worse than its reported numbers, it can only make the panel honest about them.
    agg = [parse_source_stats(row["results_pretrained"], row["source"]),
           parse_source_stats(row["results_finetuned"], row["source"])]
    scored = []
    for cand in candidates:
        _, x, gt, tmask = cand
        dist, tile = 0.0, []
        for model, a in zip((pretrained, finetuned), agg):
            _, tp, fp, _ = evaluate(model, x, gt, tmask)
            r, p = scores(gt, tp, fp)
            dist += abs(r - a["recall"]) + abs(p - a["precision"])
            tile.append((r, p))
        scored.append((dist, tile, cand))
    scored.sort(key=lambda t: t[0])
    dist, tile, chosen = scored[0]
    print(f"  candidate distance-to-aggregate {[round(d, 3) for d, _, _ in scored]}")
    print(f"  chose {dist:.3f}: pretrained R/P {tile[0][0]:.2f}/{tile[0][1]:.2f} vs "
          f"aggregate {agg[0]['recall']:.2f}/{agg[0]['precision']:.2f}; "
          f"fine-tuned {tile[1][0]:.2f}/{tile[1][1]:.2f} vs "
          f"{agg[1]['recall']:.2f}/{agg[1]['precision']:.2f}")

    img, x, gt, tmask = chosen
    panels = []
    for label, model in ((PRETRAINED_LABEL, pretrained), (FINETUNED_LABEL, finetuned)):
        pred, tp, fp, ig = evaluate(model, x, gt, tmask)
        r, p = scores(gt, tp, fp)
        panels.append(dict(img=img, gt=gt, pred=pred, tp=tp, fp=fp, ig=ig, label=label,
                           title=row["title"], subtitle=row["subtitle"],
                           source=row["source"], aggregate=aggregate_line(row)))
        print(f"  {label.splitlines()[0]:11s} recall={r:.2f} precision={p:.2f} "
              f"pred={len(pred)} gt={len(gt)} fp={len(fp)} ignored={len(ig)}")
    del finetuned
    return panels


def panel(ax, d):
    ax.imshow(d["img"])
    gt, pred = d["gt"], d["pred"]
    if len(gt):
        ax.scatter(gt[:, 0], gt[:, 1], s=70, facecolors="none", edgecolors=GT_C,
                   linewidths=1.5)
    for idx, c in ((d["tp"], TP_C), (d["ig"], IG_C), (d["fp"], FP_C)):
        if len(idx):
            ax.scatter(pred[idx, 0], pred[idx, 1], s=26, c=c, marker="x", linewidths=1.7)
    r, p = scores(gt, d["tp"], d["fp"])
    # Per-tile numbers go *below* the image: the space above each row carries the regime
    # header and the column labels, and stacking four text blocks there is unreadable.
    ax.set_xlabel(f"this tile:  recall {r:.2f} · precision {p:.2f} · "
                  f"{len(pred)} pred / {len(gt)} GT", fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])
    return r, p


def render(args, panels):
    n = len(panels)
    # The tiles are square and drawn with equal aspect, so the row height has to be set
    # from the column width or matplotlib pillarboxes each image inside a taller cell and
    # the rows drift apart. Columns sit close together (wspace) so the two models read as
    # one comparison and the per-tile caption fits under its own tile.
    col_w = args.fig_width / 2
    fig, axes = plt.subplots(n, 2, figsize=(args.fig_width, (col_w + args.row_pad) * n),
                             squeeze=False,
                             gridspec_kw=dict(wspace=0.04, hspace=args.row_pad / col_w))
    for r, row_panels in enumerate(panels):
        for c, d in enumerate(row_panels):
            panel(axes[r][c], d)
            if r == 0:
                axes[r][c].annotate(d["label"], xy=(0.5, 1.0), xytext=(0, 44),
                                    xycoords="axes fraction", textcoords="offset points",
                                    fontsize=12, fontweight="bold", ha="center",
                                    va="bottom", annotation_clip=False)
        # Regime header spanning the row, so the two columns read as one comparison
        # rather than as two independent images.
        head = axes[r][0]
        head.annotate(f"{r + 1}.  {d['title']} — {d['subtitle']}",
                      xy=(0.0, 1.0), xytext=(0, 20), xycoords="axes fraction",
                      textcoords="offset points", fontsize=11.5, fontweight="bold",
                      ha="left", va="bottom", annotation_clip=False)
        head.annotate(d["aggregate"], xy=(0.0, 1.0), xytext=(0, 6),
                      xycoords="axes fraction", textcoords="offset points",
                      fontsize=8.5, style="italic", color="0.35",
                      ha="left", va="bottom", annotation_clip=False)

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
    # Reserve the legend strip and the header band in figure fractions rather than calling
    # tight_layout, which re-fits the axes and undoes the aspect-matched row height above.
    legend_h = 0.42 / fig.get_figheight()
    header_h = (0.95 if args.no_suptitle else 1.35) / fig.get_figheight()
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, legend_h * 0.15))
    fig.subplots_adjust(left=0.035, right=0.965, bottom=legend_h * 1.5,
                        top=1 - header_h)
    if not args.no_suptitle:
        fig.suptitle("TreeFormer fine-tuning helps in proportion to how closely the "
                     "evaluation resembles its training sources", fontsize=13, y=0.995)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # No bbox_inches="tight": the reserved bands above are already the right margins, and
    # re-cropping to the artists would clip the row headers, which overhang the axes.
    fig.savefig(args.out, dpi=180)
    print("wrote", args.out)


def save_cache(path, panels):
    flat = {}
    for r, row_panels in enumerate(panels):
        for c, d in enumerate(row_panels):
            for k, v in d.items():
                flat[f"{r}_{c}_{k}"] = v
    flat["shape"] = np.array([len(panels), 2])
    np.savez_compressed(path, **flat)
    print("cached panels ->", path)


def load_cache(path):
    z = np.load(path, allow_pickle=True)
    nrow, ncol = z["shape"]
    panels = []
    for r in range(nrow):
        row_panels = []
        for c in range(ncol):
            d = {k: z[f"{r}_{c}_{k}"] for k in ("img", "gt", "pred", "tp", "fp", "ig")}
            for k in ("label", "title", "subtitle", "source", "aggregate"):
                d[k] = str(z[f"{r}_{c}_{k}"])
            row_panels.append(d)
        panels.append(row_panels)
    return panels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dir", default="/orange/ewhite/web/public/MillionTrees")
    ap.add_argument("--version", default="0.23",
                    help="Pinned dataset version. NOT the loader default: the default is "
                         "whatever _versions_dict currently ends with, which silently "
                         "changed from 0.23 to 0.22 and back mid-development and would "
                         "draw tiles from one packaging under aggregates measured on "
                         "another. The ROWS result files are v0.23 runs.")
    ap.add_argument("--image-size", type=int, default=896)
    ap.add_argument("--score-thresh", type=float, default=0.1)
    ap.add_argument("--score-integration-radius", type=int, default=2)
    ap.add_argument("--pretrained", default="weecology/deepforest-tree-point")
    ap.add_argument("--rows", nargs="+", default=[r["key"] for r in ROWS],
                    choices=[r["key"] for r in ROWS])
    ap.add_argument("--min-gt", type=int, default=15)
    ap.add_argument("--max-gt", type=int, default=400)
    ap.add_argument("--scan-limit", type=int, default=200,
                    help="Cap on that source's images examined per row.")
    ap.add_argument("--candidates", type=int, default=9,
                    help="Images considered per row; the one closest to the source "
                         "aggregate is drawn.")
    ap.add_argument("--out", default="docs/public/treeformer_generalization_figure.png")
    ap.add_argument("--no-suptitle", action="store_true")
    ap.add_argument("--fig-width", type=float, default=11.0)
    ap.add_argument("--row-pad", type=float, default=1.05,
                    help="Inches of header/caption space per row, on top of the square tile.")
    ap.add_argument("--cache", default=None,
                    help="Panel cache path (default: <out>.panels.npz).")
    ap.add_argument("--from-cache", action="store_true",
                    help="Render from the cache; skips dataset and model loading entirely.")
    ap.add_argument("--list-sources", action="store_true")
    args = ap.parse_args()

    cache_path = args.cache or (os.path.splitext(args.out)[0] + ".panels.npz")

    if args.from_cache:
        if not os.path.isfile(cache_path):
            raise SystemExit(f"--from-cache given but no cache at {cache_path}")
        print("rendering from cache:", cache_path)
        render(args, load_cache(cache_path))
        return

    pretrained = load_model(args.pretrained, args.score_thresh,
                            args.score_integration_radius)
    panels = []
    for row in ROWS:
        if row["key"] not in args.rows:
            continue
        built = build_row(row, args, pretrained)
        if built is not None:
            panels.append(built)
    if not panels:
        return
    save_cache(cache_path, panels)
    render(args, panels)


if __name__ == "__main__":
    main()
