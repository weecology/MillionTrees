"""
Filter AutoArborist point annotations using Restor TCD SegFormer (tree/no-tree segmentation).

Removes annotations that sit on majority no-tree pixels according to
https://huggingface.co/restor/tcd-segformer-mit-b5, and writes a new CSV suitable
for package_datasets.py TreePoints.

Usage:
  Debug (1–2 tiles):  uv run python data_prep/filter_auto_arborist_tcd.py --debug
  Full run:           uv run python data_prep/filter_auto_arborist_tcd.py
  SLURM:              sbatch slurm/filter_auto_arborist_tcd.sbatch
  Visualize:          uv run python data_prep/visualize_tcd_filter.py --max-per-city 2

Then in package_datasets.py TreePoints list, add or replace with the filtered path:
  "/orange/ewhite/DeepForest/AutoArborist/downloaded_imagery/AutoArborist_combined_annotations_tcd_filtered.csv"
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

# Default paths (same as package_datasets.py and AutoArborist)
DEFAULT_ANNOTATIONS_CSV = (
    "/orange/ewhite/DeepForest/AutoArborist/downloaded_imagery/AutoArborist_combined_annotations.csv"
)
DEFAULT_OUTPUT_CSV = (
    "/orange/ewhite/DeepForest/AutoArborist/downloaded_imagery/AutoArborist_combined_annotations_tcd_filtered.csv"
)
TILE_SIZE = 1024  # Model card recommends 1024 px max for inference
TREE_CLASS_ID = 1  # Binary: 0=no tree, 1=tree


def load_model(device: str):
    """Load TCD SegFormer model and processor."""
    processor = AutoImageProcessor.from_pretrained("restor/tcd-segformer-mit-b5")
    model = SegformerForSemanticSegmentation.from_pretrained("restor/tcd-segformer-mit-b5")
    model.to(device)
    model.eval()
    return processor, model


def image_to_tiles(arr: np.ndarray, tile_size: int):
    """Yield (y_start, x_start, tile) for non-overlapping tiles. arr is (C, H, W)."""
    _, h, w = arr.shape
    for y0 in range(0, h, tile_size):
        for x0 in range(0, w, tile_size):
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            tile = arr[:, y0:y1, x0:x1]
            if tile.size == 0:
                continue
            yield y0, x0, tile


def _array_to_pil(arr_chw: np.ndarray) -> Image.Image:
    """(C, H, W) uint8 or float [0,1] -> PIL RGB."""
    if arr_chw.max() <= 1.0:
        arr_chw = (arr_chw * 255).astype(np.uint8)
    arr_hwc = np.transpose(arr_chw, (1, 2, 0))
    return Image.fromarray(arr_hwc)


def predict_tree_mask(processor, model, image_path: str, device: str) -> np.ndarray:
    """
    Run TCD SegFormer on an image and return binary tree mask (H, W), 1 = tree.
    Tiles at 1024x1024 if image is larger.
    """
    with rasterio.open(image_path) as src:
        arr = src.read()  # (C, H, W)
        if arr.shape[0] >= 4:
            arr = arr[:3, :, :]
        height, width = arr.shape[1], arr.shape[2]
    if arr.max() > 1:
        arr = arr.astype(np.float32) / 255.0

    if height <= TILE_SIZE and width <= TILE_SIZE:
        pil_image = _array_to_pil(arr)
        inputs = processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        logits = out.logits  # (1, num_classes, h, w)
        mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        tree_mask = (mask == TREE_CLASS_ID).astype(np.uint8)
        if tree_mask.shape != (height, width):
            tree_mask = cv2.resize(
                tree_mask, (width, height), interpolation=cv2.INTER_NEAREST
            )
        return tree_mask

    # Tile and stitch; resize each tile's mask to match tile size (model may output smaller)
    full_mask = np.zeros((height, width), dtype=np.uint8)
    for y0, x0, tile in image_to_tiles(arr, TILE_SIZE):
        tile_h, tile_w = tile.shape[1], tile.shape[2]
        tile_pil = _array_to_pil(tile)
        inputs = processor(images=tile_pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        logits = out.logits
        tile_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        tree_tile = (tile_mask == TREE_CLASS_ID).astype(np.uint8)
        if tree_tile.shape != (tile_h, tile_w):
            tree_tile = cv2.resize(
                tree_tile, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST
            )
        full_mask[y0 : y0 + tile_h, x0 : x0 + tile_w] = tree_tile
    return full_mask


def tree_fraction_at_point(mask: np.ndarray, x: float, y: float, radius: int) -> float:
    """
    Fraction of tree pixels in a square window centered at (x, y).
    Returns value in [0, 1]. Clips window to image bounds.
    """
    h, w = mask.shape
    cx, cy = int(round(x)), int(round(y))
    y_lo = max(0, cy - radius)
    y_hi = min(h, cy + radius + 1)
    x_lo = max(0, cx - radius)
    x_hi = min(w, cx + radius + 1)
    window = mask[y_lo:y_hi, x_lo:x_hi]
    if window.size == 0:
        return 0.0
    return float(np.mean(window == 1))


def filter_annotations_with_mask(
    df: pd.DataFrame, mask: np.ndarray, radius: int = 5, min_tree_fraction: float = 0.5
) -> pd.DataFrame:
    """
    Keep only rows where (x, y) has >= min_tree_fraction tree pixels in a (2*radius+1)^2 window.
    """
    keep = []
    for _, row in df.iterrows():
        x, y = row["x"], row["y"]
        frac = tree_fraction_at_point(mask, x, y, radius)
        keep.append(frac >= min_tree_fraction)
    return df.loc[keep].copy()


def resolve_image_path(row_path: str, root_dir: str | None) -> str:
    """Return absolute path; if root_dir given and row_path is basename, join."""
    if os.path.isabs(row_path) and os.path.exists(row_path):
        return row_path
    if root_dir and not os.path.isabs(row_path):
        candidate = os.path.join(root_dir, os.path.basename(row_path))
        if os.path.exists(candidate):
            return candidate
    return row_path


def run(
    annotations_csv: str,
    output_csv: str,
    root_dir: str | None,
    device: str,
    debug: bool,
    max_tiles: int,
    radius: int,
    min_tree_fraction: float,
):
    df = pd.read_csv(annotations_csv)
    if "image_path" not in df.columns or "x" not in df.columns or "y" not in df.columns:
        raise ValueError(
            "CSV must have columns image_path, x, y. "
            f"Found: {list(df.columns)}"
        )

    # Optional root for relative image_path
    if root_dir is None and not df["image_path"].iloc[0].startswith("/"):
        root_dir = str(Path(annotations_csv).parent)

    processor, model = load_model(device)
    unique_images = df["image_path"].unique().tolist()
    if debug and max_tiles is not None:
        unique_images = unique_images[: max_tiles]

    kept_dfs = []
    for imp in unique_images:
        full_path = resolve_image_path(imp, root_dir)
        if not os.path.exists(full_path):
            print(f"Skipping missing image: {full_path}")
            continue
        sub = df.loc[df["image_path"] == imp].copy()
        mask = predict_tree_mask(processor, model, full_path, device)
        filtered = filter_annotations_with_mask(
            sub, mask, radius=radius, min_tree_fraction=min_tree_fraction
        )
        kept_dfs.append(filtered)
        if debug:
            print(f"  {Path(full_path).name}: {len(sub)} -> {len(filtered)} kept")

    if not kept_dfs:
        raise RuntimeError("No images were processed; check paths and --root-dir.")
    out_df = pd.concat(kept_dfs, ignore_index=True)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Wrote {len(out_df)} annotations to {output_csv} (from {len(df)} total)")
    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description="Filter AutoArborist annotations by TCD tree/no-tree segmentation."
    )
    parser.add_argument(
        "--annotations-csv",
        default=DEFAULT_ANNOTATIONS_CSV,
        help="Path to AutoArborist combined annotations CSV",
    )
    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV,
        help="Path to write filtered CSV",
    )
    parser.add_argument(
        "--root-dir",
        default=None,
        help="Root dir for image_path if paths in CSV are relative",
    )
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run on one or two tiles only for testing",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=2,
        help="With --debug, max number of unique images to process (default 2)",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=5,
        help="Window half-size around each point for tree fraction (default 5 -> 11x11)",
    )
    parser.add_argument(
        "--min-tree-fraction",
        type=float,
        default=0.5,
        help="Keep annotation if tree fraction in window >= this (default 0.5)",
    )
    args = parser.parse_args()

    run(
        annotations_csv=args.annotations_csv,
        output_csv=args.output_csv,
        root_dir=args.root_dir,
        device=args.device,
        debug=args.debug,
        max_tiles=args.max_tiles,
        radius=args.radius,
        min_tree_fraction=args.min_tree_fraction,
    )


if __name__ == "__main__":
    main()
