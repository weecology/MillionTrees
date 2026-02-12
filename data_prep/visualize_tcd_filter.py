"""
Visualize TCD filter: annotations (kept vs removed) overlaid on imagery and tree/no-tree predictions.

Generates one figure per image showing:
  - RGB image with points (green = kept, red = removed by TCD)
  - Same image with semi-transparent tree mask overlay
  - Optional: mask only

Run on a sample of images per city to inspect whether filtering is too aggressive.
"""
import argparse
import os
import sys
from pathlib import Path

# Project root so "data_prep" resolves when run as script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

# Reuse filter logic and model
from data_prep.filter_auto_arborist_tcd import (
    load_model,
    predict_tree_mask,
    resolve_image_path,
    tree_fraction_at_point,
)

DEFAULT_ANNOTATIONS_CSV = (
    "/orange/ewhite/DeepForest/AutoArborist/downloaded_imagery/AutoArborist_combined_annotations.csv"
)


def load_rgb(image_path: str) -> np.ndarray:
    """Load image as (H, W, 3) uint8 RGB."""
    with rasterio.open(image_path) as src:
        arr = src.read()
        if arr.shape[0] >= 4:
            arr = arr[:3, :, :]
    arr = np.transpose(arr, (1, 2, 0))
    if arr.max() > 1:
        arr = (arr.astype(np.float32) / 255.0 * 255).astype(np.uint8)
    return arr


def plot_one(
    image_path: str,
    rgb: np.ndarray,
    mask: np.ndarray,
    df: pd.DataFrame,
    radius: int,
    min_tree_fraction: float,
    city_name: str,
    out_path: str,
    point_radius: int = 8,
) -> None:
    """
    Create a figure: (1) RGB + points green/red, (2) RGB + tree overlay, (3) mask.
    """
    kept_x, kept_y = [], []
    removed_x, removed_y = [], []
    for _, row in df.iterrows():
        x, y = row["x"], row["y"]
        frac = tree_fraction_at_point(mask, x, y, radius)
        if frac >= min_tree_fraction:
            kept_x.append(x)
            kept_y.append(y)
        else:
            removed_x.append(x)
            removed_y.append(y)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: RGB with points (green = kept, red = removed)
    ax1 = axes[0]
    ax1.imshow(rgb)
    if kept_x:
        ax1.scatter(kept_x, kept_y, c="lime", s=point_radius**2, label="kept", alpha=0.9)
    if removed_x:
        ax1.scatter(
            removed_x, removed_y, c="red", s=point_radius**2, label="removed", alpha=0.9
        )
    ax1.set_title(f"{city_name}\n{Path(image_path).name}\nkept={len(kept_x)} removed={len(removed_x)}")
    ax1.legend(loc="upper right")
    ax1.axis("off")

    # Panel 2: RGB with tree mask overlay (green = tree)
    ax2 = axes[1]
    ax2.imshow(rgb)
    overlay = np.zeros((*mask.shape, 4))
    overlay[mask == 1, 1] = 0.5  # green
    overlay[mask == 1, 3] = 0.4
    ax2.imshow(overlay)
    ax2.set_title("Tree mask overlay (green = tree)")
    ax2.axis("off")

    # Panel 3: Mask only
    ax3 = axes[2]
    ax3.imshow(mask, cmap="gray", vmin=0, vmax=1)
    ax3.set_title("Tree / no-tree mask")
    ax3.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def run(
    annotations_csv: str,
    output_dir: str,
    root_dir: str | None,
    device: str,
    max_per_city: int,
    max_cities: int | None,
    radius: int,
    min_tree_fraction: float,
) -> None:
    df = pd.read_csv(annotations_csv)
    if "image_path" not in df.columns or "x" not in df.columns or "y" not in df.columns:
        raise ValueError("CSV must have image_path, x, y.")
    if "city" not in df.columns:
        df["city"] = "unknown"

    if root_dir is None and len(df) and not str(df["image_path"].iloc[0]).startswith("/"):
        root_dir = str(Path(annotations_csv).parent)

    os.makedirs(output_dir, exist_ok=True)
    processor, model = load_model(device)

    cities = df["city"].unique().tolist()
    if max_cities is not None:
        cities = cities[: max_cities]

    for city in cities:
        city_df = df[df["city"] == city]
        images = city_df["image_path"].unique().tolist()
        if max_per_city is not None:
            images = images[: max_per_city]
        for imp in images:
            full_path = resolve_image_path(imp, root_dir)
            if not os.path.exists(full_path):
                print(f"Skipping missing: {full_path}")
                continue
            sub = city_df[city_df["image_path"] == imp]
            print(f"  {city} / {Path(full_path).name} ({len(sub)} annotations)")
            rgb = load_rgb(full_path)
            mask = predict_tree_mask(processor, model, full_path, device)
            safe_name = Path(full_path).stem.replace(" ", "_")
            out_path = os.path.join(output_dir, f"{city}_{safe_name}.png")
            plot_one(
                full_path,
                rgb,
                mask,
                sub,
                radius=radius,
                min_tree_fraction=min_tree_fraction,
                city_name=city,
                out_path=out_path,
            )
    print(f"Saved figures to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize TCD filter: annotations and tree mask per image."
    )
    parser.add_argument(
        "--annotations-csv",
        default=DEFAULT_ANNOTATIONS_CSV,
        help="AutoArborist combined annotations CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="/orange/ewhite/DeepForest/AutoArborist/downloaded_imagery/tcd_visualization",
        help="Directory to save PNGs",
    )
    parser.add_argument("--root-dir", default=None, help="Root for relative image paths")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-per-city",
        type=int,
        default=2,
        help="Max images to plot per city (default 2)",
    )
    parser.add_argument(
        "--max-cities",
        type=int,
        default=None,
        help="Max cities to process (default all)",
    )
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--min-tree-fraction", type=float, default=0.5)
    args = parser.parse_args()

    run(
        annotations_csv=args.annotations_csv,
        output_dir=args.output_dir,
        root_dir=args.root_dir,
        device=args.device,
        max_per_city=args.max_per_city,
        max_cities=args.max_cities,
        radius=args.radius,
        min_tree_fraction=args.min_tree_fraction,
    )


if __name__ == "__main__":
    main()
