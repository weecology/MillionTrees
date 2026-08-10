"""Measure memory footprint of one TreePolygons sample: full-res image + masks.

Run from repo root with data available, e.g.:
  uv run python scripts/measure_polygon_sample_sizes.py --root-dir /orange/ewhite/web/public/MillionTrees

Use --synthetic to print size examples without loading the dataset.
"""

import argparse
import sys
from pathlib import Path

# Add src so milliontrees is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def nbytes_str(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GiB"
    if n >= 1024**2:
        return f"{n / 1024**2:.2f} MiB"
    if n >= 1024:
        return f"{n / 1024:.2f} KiB"
    return f"{n} B"


def main():
    parser = argparse.ArgumentParser(description="Measure TreePolygons sample sizes (full-res).")
    parser.add_argument("--root-dir", type=str, default="/orange/ewhite/web/public/MillionTrees")
    parser.add_argument("--split", type=str, default="random")
    parser.add_argument("--n-samples", type=int, default=3, help="Number of train samples to measure")
    parser.add_argument("--synthetic", action="store_true", help="Print size examples only (no dataset load)")
    parser.add_argument("--mini", action="store_true", help="Load mini dataset and report all images (size + annotation count)")
    args = parser.parse_args()

    if args.synthetic:
        print("Synthetic size examples (full-res vs resized to 448):\n")
        for H, W, num_trees in [(1000, 1000, 20), (2000, 2000, 50), (4000, 4000, 100)]:
            img_full = 3 * H * W * 4  # float32 RGB
            masks_full = num_trees * H * W * 1  # uint8
            img_448 = 3 * 448 * 448 * 4
            masks_448 = num_trees * 448 * 448 * 1
            print(f"  Image {H}x{W}, {num_trees} trees:")
            print(f"    Full-res:  image {nbytes_str(img_full)} + masks {nbytes_str(masks_full)} = {nbytes_str(img_full + masks_full)}")
            print(f"    Resized:   image {nbytes_str(img_448)} + masks {nbytes_str(masks_448)} = {nbytes_str(img_448 + masks_448)}")
            print(f"  Batch of 4 full-res: {nbytes_str(4 * (img_full + masks_full))}\n")
        return

    from milliontrees import get_dataset

    if args.mini:
        print("Loading TreePolygons MINI dataset (download=True)...")
        dataset = get_dataset(
            "TreePolygons",
            root_dir=args.root_dir,
            split_scheme=args.split,
            image_size=448,
            mini=True,
            download=True,
        )
        base = dataset
        # Iterate over all unique images (dataset is indexed by image)
        indices = list(range(len(base._input_array)))
        print(f"Mini dataset: {len(indices)} images\n")
        print(f"{'Filename':<50} {'Shape (H,W,C)':<18} {'Annotations':<12} {'Image size':<12} {'Masks size':<12}")
        print("-" * 104)
        total_img = 0
        total_masks = 0
        total_ann = 0
        for idx in indices:
            metadata, x, targets = base[idx]
            filename = base._input_array[idx]
            if len(filename) > 48:
                filename = "..." + filename[-45:]
            y = targets["y"]
            n_ann = y.shape[0] if y.size else 0
            img_nbytes = x.nbytes
            mask_nbytes = y.nbytes if y.size else 0
            total_img += img_nbytes
            total_masks += mask_nbytes
            total_ann += n_ann
            print(f"{filename:<50} {str(x.shape):<18} {n_ann:<12} {nbytes_str(img_nbytes):<12} {nbytes_str(mask_nbytes):<12}")
        print("-" * 104)
        print(f"Total: {len(indices)} images, {total_ann} annotations, images {nbytes_str(total_img)}, masks {nbytes_str(total_masks)}")
        return

    print("Loading TreePolygons (image_size only affects transform; we read raw sample)...")
    dataset = get_dataset(
        "TreePolygons",
        root_dir=args.root_dir,
        split_scheme=args.split,
        image_size=448,
    )
    train_subset = dataset.get_subset("train")
    base = train_subset.dataset
    indices = train_subset.indices

    total_full_res_image = 0
    total_full_res_masks = 0
    total_resized_image = 0
    total_resized_masks = 0

    for i in range(min(args.n_samples, len(indices))):
        idx = indices[i]
        # Get one raw sample from base dataset (no subset transform) = full-res only
        metadata, x, targets = base[idx]

        # Full-res sizes
        image_nbytes = x.nbytes
        y = targets["y"]
        if y.size == 0:
            mask_nbytes = 0
            num_masks = 0
        else:
            mask_nbytes = y.nbytes
            num_masks = y.shape[0]

        # Resized (448) sizes for comparison
        h448, w448 = 448, 448
        resized_image_nbytes = x.dtype.itemsize * 3 * h448 * w448
        resized_masks_nbytes = (num_masks * h448 * w448) if num_masks else 0

        total_full_res_image += image_nbytes
        total_full_res_masks += mask_nbytes
        total_resized_image += resized_image_nbytes
        total_resized_masks += resized_masks_nbytes

        print(f"\n--- Sample {i} (dataset index {idx}) ---")
        print(f"  Image: shape {x.shape}, dtype {x.dtype}, size {nbytes_str(image_nbytes)}")
        print(f"  Masks: count {num_masks}, shape {y.shape if y.size else (0,0,0)}, size {nbytes_str(mask_nbytes)}")
        print(f"  One sample full-res total: {nbytes_str(image_nbytes + mask_nbytes)}")
        print(f"  Same sample if resized to 448 first: image {nbytes_str(resized_image_nbytes)} + masks {nbytes_str(resized_masks_nbytes)} = {nbytes_str(resized_image_nbytes + resized_masks_nbytes)}")

    print("\n--- Summary ---")
    n = min(args.n_samples, len(indices))
    print(f"  Over {n} sample(s): full-res image total {nbytes_str(total_full_res_image)}, masks total {nbytes_str(total_full_res_masks)}")
    print(f"  Batch of 4 (full-res): ~{nbytes_str(4 * (total_full_res_image + total_full_res_masks) // n)}")
    print(f"  Batch of 4 (resized 448): ~{nbytes_str(4 * (total_resized_image + total_resized_masks) // n)}")


if __name__ == "__main__":
    main()
