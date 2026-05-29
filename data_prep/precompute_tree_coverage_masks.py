"""
Precompute tree coverage masks (tree/no-tree) for MillionTrees packaging.

This script runs the Restor TCD SegFormer model once per unique image and writes
binary PNG masks keyed by image basename. These masks can then be copied into
dataset packages under masks/.
"""
import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from data_prep.filter_auto_arborist_tcd import (
    load_model,
    predict_tree_mask,
)
try:
    from data_prep.packaging_utils import (
        build_unique_name_map,
        collect_image_source_pairs,
    )
except ImportError:  # when run as a script from inside data_prep/
    from packaging_utils import build_unique_name_map, collect_image_source_pairs


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
    reuse_legacy: bool = True,
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    pairs = collect_image_source_pairs(annotation_csvs, root_dir)
    name_map = build_unique_name_map(pairs)

    # Legacy masks were keyed by bare stem, which collides across sources. A stem
    # shared by >1 distinct image is unreliable and must be recomputed; a unique
    # stem's legacy mask is correct and can simply be renamed to the new key.
    stem_paths: dict[str, set] = defaultdict(set)
    for resolved, _source in pairs:
        stem_paths[Path(resolved).stem].add(resolved)
    collided_stems = {s for s, ps in stem_paths.items() if len(ps) > 1}
    print(
        f"{len(pairs)} (image, source) pairs; {len(collided_stems)} colliding "
        f"stem(s) will be recomputed, the rest reused from legacy masks."
    )

    processor = model = None  # lazy-load the model only if we actually compute
    rows = []
    skipped = []
    reused = computed = 0
    for resolved, source in pairs:
        image_file = Path(resolved)
        if not image_file.exists():
            print(f"WARNING: Missing image, skipping: {image_file}")
            skipped.append(str(image_file))
            continue
        mask_path = output_root / f"{Path(name_map[resolved]).stem}.png"
        legacy_path = output_root / f"{image_file.stem}.png"

        if mask_path.exists() and not overwrite:
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
        else:
            reusable = (
                reuse_legacy
                and image_file.stem not in collided_stems
                and legacy_path.exists()
                and legacy_path != mask_path
            )
            if reusable:
                # The legacy mask is the one TCD produced for this exact image
                # (unique stem) and was sized to it; copy it under the new key.
                shutil.copy(legacy_path, mask_path)
                mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
                reused += 1
            else:
                if model is None:
                    processor, model = load_model(device)
                try:
                    mask = predict_tree_mask(processor, model, str(image_file), device)
                    _save_binary_mask(mask, mask_path)
                    computed += 1
                except Exception as e:
                    print(f"WARNING: Failed to process {image_file} — {e}")
                    skipped.append(str(image_file))
                    continue

        rows.append(
            {
                "filename": mask_path.stem,
                "source": source,
                "mask_path": str(mask_path),
                "tree_fraction": float(np.mean(mask > 0)),
            }
        )

    print(f"\nMasks reused from legacy: {reused} | recomputed: {computed}")

    if skipped:
        print(f"\nSkipped {len(skipped)} image(s) due to errors:")
        for s in skipped:
            print(f"  {s}")

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
    parser.add_argument(
        "--no-reuse-legacy",
        dest="reuse_legacy",
        action="store_false",
        help="Recompute every mask instead of reusing non-colliding legacy masks.",
    )
    args = parser.parse_args()

    run(
        annotation_csvs=args.annotations_csv,
        output_dir=args.output_dir,
        root_dir=args.root_dir,
        stats_csv=args.stats_csv,
        device=args.device,
        overwrite=args.overwrite,
        reuse_legacy=args.reuse_legacy,
    )


if __name__ == "__main__":
    main()
