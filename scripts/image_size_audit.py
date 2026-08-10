"""
Audit image sizes across all v0.11 datasets.
Produces:
  1. Count table by size bin (binned on the shorter dimension)
  2. Count table of sliver images (aspect ratio >= 3:1)
  3. Per-dataset summaries

Does NOT delete or modify any files.
"""

import os
import glob
from pathlib import Path
from collections import defaultdict

import pandas as pd
from PIL import Image

# ── Config ──────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/blue/ewhite/b.weinstein/src/MillionTrees/data")
VERSION = "0.11"

DATASETS = ["TreeBoxes", "TreePolygons", "TreePoints"]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Bins defined on the shorter dimension (min(width, height))
SIZE_BINS = [0, 32, 64, 128, 256, 512, 1024, 2048, float("inf")]
SIZE_LABELS = [
    "<32",
    "32–63",
    "64–127",
    "128–255",
    "256–511",
    "512–1023",
    "1024–2047",
    "≥2048",
]

# Aspect-ratio threshold: flag if max/min dimension ratio >= this value
SLIVER_THRESHOLD = 3.0


def bin_size(shorter_side):
    for i in range(len(SIZE_BINS) - 1):
        if SIZE_BINS[i] <= shorter_side < SIZE_BINS[i + 1]:
            return SIZE_LABELS[i]
    return SIZE_LABELS[-1]


def collect_images(dataset_name):
    img_dir = DATA_ROOT / f"{dataset_name}_v{VERSION}" / "images"
    if not img_dir.exists():
        print(f"  [skip] {img_dir} not found")
        return []
    paths = [
        p for p in img_dir.iterdir()
        if p.suffix.lower() in IMG_EXTS
    ]
    return paths


records = []

for ds in DATASETS:
    print(f"Scanning {ds} …", flush=True)
    paths = collect_images(ds)
    print(f"  Found {len(paths)} images")
    for p in paths:
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception as e:
            print(f"  [warn] cannot open {p.name}: {e}")
            continue

        shorter = min(w, h)
        longer = max(w, h)
        aspect = longer / shorter if shorter > 0 else float("inf")
        is_sliver = aspect >= SLIVER_THRESHOLD

        records.append(
            {
                "dataset": ds,
                "filename": p.name,
                "width": w,
                "height": h,
                "shorter_dim": shorter,
                "longer_dim": longer,
                "aspect_ratio": round(aspect, 2),
                "size_bin": bin_size(shorter),
                "is_sliver": is_sliver,
            }
        )

df = pd.DataFrame(records)

# ── 1. Size-bin table ────────────────────────────────────────────────────────
# Order bins correctly
bin_order = pd.Categorical(df["size_bin"], categories=SIZE_LABELS, ordered=True)
df["size_bin_ord"] = bin_order

pivot = (
    df.groupby(["dataset", "size_bin_ord"], observed=True)
    .size()
    .unstack("dataset", fill_value=0)
)
pivot["TOTAL"] = pivot.sum(axis=1)

print("\n\n=== Image count by size bin (shorter dimension) ===")
print(pivot.to_string())

# ── 2. Sliver table ──────────────────────────────────────────────────────────
sliver_df = df[df["is_sliver"]].copy()

sliver_pivot = (
    sliver_df.groupby(["dataset", "size_bin_ord"], observed=True)
    .size()
    .unstack("dataset", fill_value=0)
)
if not sliver_pivot.empty:
    sliver_pivot["TOTAL"] = sliver_pivot.sum(axis=1)

print(f"\n\n=== Sliver images (aspect ratio ≥ {SLIVER_THRESHOLD}:1) ===")
print(f"Total slivers: {len(sliver_df)} of {len(df)} ({100*len(sliver_df)/max(len(df),1):.1f}%)")
if not sliver_pivot.empty:
    print(sliver_pivot.to_string())

# ── 3. Per-dataset summary ───────────────────────────────────────────────────
print("\n\n=== Per-dataset summary ===")
summary = df.groupby("dataset").agg(
    total_images=("filename", "count"),
    min_shorter=("shorter_dim", "min"),
    median_shorter=("shorter_dim", "median"),
    max_shorter=("shorter_dim", "max"),
    sliver_count=("is_sliver", "sum"),
).copy()
summary["sliver_pct"] = (summary["sliver_count"] / summary["total_images"] * 100).round(1)
print(summary.to_string())

# ── 4. Worst slivers ─────────────────────────────────────────────────────────
print(f"\n\n=== Top 20 most extreme slivers ===")
if not sliver_df.empty:
    top = sliver_df.nlargest(20, "aspect_ratio")[
        ["dataset", "filename", "width", "height", "aspect_ratio"]
    ]
    print(top.to_string(index=False))
else:
    print("No slivers found.")

# ── 5. Very small images (<64px on shorter side) ─────────────────────────────
tiny = df[df["shorter_dim"] < 64]
print(f"\n\n=== Very small images (shorter dim < 64px): {len(tiny)} total ===")
if not tiny.empty:
    print(tiny[["dataset", "filename", "width", "height"]].to_string(index=False))

# ── Save full table ──────────────────────────────────────────────────────────
out_csv = Path(__file__).parent / "image_size_audit.csv"
df.drop(columns="size_bin_ord").to_csv(out_csv, index=False)
print(f"\n\nFull table saved to: {out_csv}")
