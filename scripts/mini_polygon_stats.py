"""Report original image sizes and annotation counts for the TreePolygons MINI dataset.

Does not use the full dataset class so we can handle CSV parsing. Reads the
extracted mini data dir (e.g. data/TreePolygons_v0.11 after downloading with mini=True).

Usage:
  uv run python scripts/mini_polygon_stats.py --data-dir data/TreePolygons_v0.11
"""

import argparse
import csv
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/TreePolygons_v0.11"))
    parser.add_argument(
        "--heavy-paths",
        action="store_true",
        help="Print full paths for heavy images only (large size or many annotations)",
    )
    parser.add_argument(
        "--heavy-mb",
        type=float,
        default=50.0,
        help="Threshold MB for 'heavy' by size (default 50)",
    )
    parser.add_argument(
        "--heavy-anns",
        type=int,
        default=500,
        help="Threshold annotation count for 'heavy' (default 500)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    csv_path = data_dir / "random.csv"
    images_dir = data_dir / "images"

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return
    if not images_dir.exists():
        print(f"Images dir not found: {images_dir}")
        return

    # Parse CSV with proper quoting (polygon column contains WKT with commas)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            filename_idx = header.index("filename")
        except ValueError:
            print("No 'filename' column in", header)
            return
        rows = list(reader)

    # Count annotations per filename
    from collections import Counter
    counts = Counter()
    for row in rows:
        if len(row) > filename_idx:
            counts[row[filename_idx]] += 1

    # Get image dimensions for each filename
    unique_filenames = sorted(counts.keys())
    print(f"Mini TreePolygons: {len(unique_filenames)} unique images, {sum(counts.values())} total annotations\n")
    print(f"{'Filename':<55} {'Shape (W,H)':<14} {'Annotations':<12} {'Approx. image MB':<16}")
    print("-" * 100)

    total_img_mb = 0.0
    heavy_paths = []
    for fn in unique_filenames:
        img_path = images_dir / fn
        if not img_path.exists():
            shape_str = "(missing)"
            img_mb = 0.0
        else:
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
                    shape_str = f"{w}x{h}"
                    # Approx uncompressed RGB float32 MB
                    img_mb = (w * h * 3 * 4) / (1024 * 1024)
                    total_img_mb += img_mb
            except Exception as e:
                shape_str = f"err: {e}"
                img_mb = 0.0
        n = counts[fn]
        if args.heavy_paths and (img_mb >= args.heavy_mb or n >= args.heavy_anns):
            heavy_paths.append((str(img_path.resolve()), img_mb, n))
        display_fn = fn if len(fn) <= 52 else "..." + fn[-49:]
        print(f"{display_fn:<55} {shape_str:<14} {n:<12} {img_mb:.2f}")

    print("-" * 100)
    print(f"Total ~{total_img_mb:.2f} MB (images as float32 RGB). One batch of 4 would be ~{4 * total_img_mb / len(unique_filenames):.2f} MB per batch (avg).")

    if args.heavy_paths and heavy_paths:
        print("\nHeavy images (full paths):")
        for path, img_mb, n in sorted(heavy_paths, key=lambda x: (-x[1], -x[2])):
            print(path)


if __name__ == "__main__":
    main()
