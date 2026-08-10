"""Per-image annotation-completeness audit of the held-out validation split.

Motivation: the F1-vs-AP50 gap is hypothesised to come from incomplete annotation
(recall / mask-aware precision forgive predictions that land on unannotated canopy;
DetectionMAP does not). This script quantifies, for every validation image, how much
of the tile the annotations actually cover, so we can restrict AP to tiles that are
"pretty much complete" and re-score there.

Outputs a CSV of per-image stats plus GT-overlay PNGs (annotations + annotated
footprint + tree-coverage mask) for visual inspection.
"""

import argparse
import os

import numpy as np
import pandas as pd

TASK = {"boxes": "TreeBoxes", "polygons": "TreePolygons", "points": "TreePoints"}


def version_dir(geometry, version, supervised=True):
    """Packaged directory for a geometry/version.

    The loaders default to the ``_supervised_`` package (``include_unsupervised=False``),
    so audit that one -- it is what every eval run actually reads.
    """
    suffix = "_supervised" if supervised else ""
    return f"{TASK[geometry]}{suffix}_v{version}"


def load_validation(root_dir, geometry, version, split_scheme="out-of-distribution",
                    supervised=True):
    path = os.path.join(root_dir, version_dir(geometry, version, supervised),
                        f"{split_scheme}.csv")
    chunks = []
    for chunk in pd.read_csv(path, chunksize=500_000):
        chunks.append(chunk[chunk["split"] == "validation"])
    return pd.concat(chunks, ignore_index=True)


def geom_bounds(df, geometry):
    """Return per-row (xmin, ymin, xmax, ymax) for boxes or polygon WKT."""
    if geometry == "boxes":
        return df[["xmin", "ymin", "xmax", "ymax"]].to_numpy(float)
    from shapely import wkt
    bounds = np.array([wkt.loads(p).bounds for p in df["polygon"]], dtype=float)
    return bounds


def annotated_footprint_mask(bounds, shape):
    """Boolean [H, W] union of the annotation bounding boxes."""
    H, W = shape
    covered = np.zeros((H, W), dtype=bool)
    for x0, y0, x1, y1 in bounds:
        xa, xb = int(max(0, np.floor(x0))), int(min(W, np.ceil(x1)))
        ya, yb = int(max(0, np.floor(y0))), int(min(H, np.ceil(y1)))
        if xb > xa and yb > ya:
            covered[ya:yb, xa:xb] = True
    return covered


def annotated_canopy_fraction(bounds, tree_mask):
    """Fraction of canopy pixels that fall inside at least one annotation.

    Catches the failure a margin test misses: a tile whose annotations span edge to edge
    but leave a whole unlabelled block of forest inside. Predictions there are charged as
    false positives by AP even though the trees are real.
    """
    covered = annotated_footprint_mask(bounds, tree_mask.shape)
    total = int(tree_mask.sum())
    if total == 0:
        return float("nan")
    return float((covered & tree_mask).sum() / total)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root-dir", default=os.environ.get(
        "MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    ap.add_argument("--version", default="0.22")
    ap.add_argument("--unsupervised-package", action="store_true",
                    help="Audit the full package instead of the _supervised_ one the "
                         "loaders default to.")
    ap.add_argument("--geometry", default="boxes", choices=["boxes", "polygons", "points"])
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--viz-dir", default=None)
    ap.add_argument("--viz-max-side", type=int, default=1200)
    ap.add_argument("--viz-per-source", type=int, default=None,
                    help="Limit overlays per source (default: all images).")
    args = ap.parse_args()

    supervised = not args.unsupervised_package
    ds_dir = os.path.join(args.root_dir,
                          version_dir(args.geometry, args.version, supervised))
    img_dir = os.path.join(ds_dir, "images")
    mask_dir = os.path.join(ds_dir, "masks")

    df = load_validation(args.root_dir, args.geometry, args.version,
                         supervised=supervised)
    print(f"validation rows={len(df)} images={df.filename.nunique()} "
          f"sources={sorted(df.source.unique())}")

    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    rows = []
    viz_count = {}
    for fname, g in df.groupby("filename", sort=True):
        source = g["source"].iloc[0]
        img_path = os.path.join(img_dir, fname)
        with Image.open(img_path) as im:
            W, H = im.size
        b = geom_bounds(g, args.geometry)
        x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
        hull_frac = ((x1 - x0) * (y1 - y0)) / float(W * H)
        # Margin of unannotated image on each side, as a fraction of the tile.
        margins = np.array([x0 / W, y0 / H, (W - x1) / W, (H - y1) / H])

        # Tree-coverage mask: fraction of the tile the mask calls canopy. A tile whose
        # annotated footprint is small but whose canopy fraction is high is exactly the
        # case where mask-aware precision forgives FPs and AP does not. The mask comes
        # from an independent canopy segmenter (Restor TCD SegFormer), never from the
        # annotations, so "how much canopy is annotated" is not circular.
        mask_path = os.path.join(mask_dir, os.path.splitext(fname)[0] + ".png")
        tree_frac = np.nan
        canopy_annotated = np.nan
        if os.path.exists(mask_path):
            with Image.open(mask_path) as mk:
                tree = np.asarray(mk.convert("L")) > 0
            tree_frac = float(tree.mean())
            canopy_annotated = annotated_canopy_fraction(b, tree)

        rows.append(dict(filename=fname, source=source, complete=bool(g["complete"].iloc[0]),
                         n_annotations=len(g), width=W, height=H,
                         gt_bbox_frac=hull_frac, max_margin=float(margins.max()),
                         tree_mask_frac=tree_frac,
                         canopy_annotated_frac=canopy_annotated))

        if args.viz_dir:
            n = viz_count.get(source, 0)
            if args.viz_per_source is None or n < args.viz_per_source:
                viz_count[source] = n + 1
                render(img_path, mask_path, b, (W, H), args.viz_dir, source, fname,
                       args.viz_max_side)

    out = pd.DataFrame(rows).sort_values(["source", "filename"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(out.groupby("source")[["gt_bbox_frac", "max_margin", "tree_mask_frac",
                                 "n_annotations"]].describe().T.to_string())
    print(f"\nWrote {args.out_csv}")


def render(img_path, mask_path, bounds, size, viz_dir, source, fname, max_side):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    W, H = size
    scale = min(1.0, max_side / max(W, H))
    with Image.open(img_path) as im:
        im = im.convert("RGB").resize((max(1, int(W * scale)), max(1, int(H * scale))))
        arr = np.asarray(im)

    fig, ax = plt.subplots(figsize=(arr.shape[1] / 100, arr.shape[0] / 100), dpi=100)
    ax.imshow(arr)
    if os.path.exists(mask_path):
        with Image.open(mask_path) as mk:
            mk = mk.convert("L").resize((arr.shape[1], arr.shape[0]))
            m = np.asarray(mk) > 0
        # Red wash = canopy the segmenter found but nobody annotated: exactly the pixels
        # where a correct detection is scored as a false positive by AP.
        unlabelled = m & ~annotated_footprint_mask(bounds * scale, arr.shape[:2])
        overlay = np.zeros(arr.shape[:2] + (4,), dtype=float)
        overlay[unlabelled] = [1.0, 0.1, 0.1, 0.30]
        ax.imshow(overlay)
    for x0, y0, x1, y1 in bounds * scale:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor="#a020f0", linewidth=0.6))
    b = bounds * scale
    ax.add_patch(Rectangle((b[:, 0].min(), b[:, 1].min()),
                           b[:, 2].max() - b[:, 0].min(), b[:, 3].max() - b[:, 1].min(),
                           fill=False, edgecolor="#ffcc00", linewidth=2, linestyle="--"))
    ax.set_axis_off()
    ax.set_title(f"{fname}  n={len(bounds)}", fontsize=8)
    d = os.path.join(viz_dir, source.replace(" ", "_").replace(".", ""))
    os.makedirs(d, exist_ok=True)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(d, os.path.splitext(fname)[0] + ".png"),
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    main()
