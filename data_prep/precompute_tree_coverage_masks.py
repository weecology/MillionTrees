"""
Precompute tree coverage masks (tree/no-tree) for MillionTrees packaging.

This script runs the Restor TCD SegFormer model once per unique image and writes
binary PNG masks keyed by image basename. These masks can then be copied into
dataset packages under masks/.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from data_prep.filter_auto_arborist_tcd import (
    load_model,
    predict_tree_mask,
    resolve_image_path,
)


def _collect_unique_images(annotation_csvs: list[str], root_dir: str | None) -> list[str]:
    image_paths = []
    for csv_path in annotation_csvs:
        df = pd.read_csv(csv_path)
        col = "filename" if "filename" in df.columns else "image_path"
        if col not in df.columns:
            raise ValueError(f"{csv_path} must include 'filename' or 'image_path'")
        image_paths.extend(df[col].dropna().unique().tolist())

    unique = sorted(set(image_paths))
    return [resolve_image_path(path, root_dir) for path in unique]


def _save_binary_mask(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    img.save(output_path)


def run(
    annotation_csvs: list[str],
    output_dir: str,
    root_dir: str | None,
    stats_csv: str | None,
    device: str,
    overwrite: bool,
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    processor, model = load_model(device)

    images = _collect_unique_images(annotation_csvs, root_dir)
    rows = []
    for image_path in images:
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Missing image: {image_file}")
        mask_path = output_root / f"{image_file.stem}.png"
        if mask_path.exists() and not overwrite:
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
        else:
            mask = predict_tree_mask(processor, model, str(image_file), device)
            _save_binary_mask(mask, mask_path)

        rows.append(
            {
                "filename": image_file.name,
                "mask_path": str(mask_path),
                "tree_fraction": float(np.mean(mask > 0)),
            }
        )

    if stats_csv:
        pd.DataFrame(rows).to_csv(stats_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute tree coverage masks for MillionTrees packaging."
    )
    parser.add_argument(
        "--annotations-csv",
        nargs="+",
        required=True,
        help="One or more annotation CSV paths containing filename or image_path columns.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write mask PNG files.",
    )
    parser.add_argument(
        "--root-dir",
        default=None,
        help="Optional root for relative image paths in CSV files.",
    )
    parser.add_argument(
        "--stats-csv",
        default=None,
        help="Optional output CSV for per-image coverage statistics.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run(
        annotation_csvs=args.annotations_csv,
        output_dir=args.output_dir,
        root_dir=args.root_dir,
        stats_csv=args.stats_csv,
        device=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
