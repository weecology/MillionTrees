"""Render the trained MaskRCNNPolygonTrainer overlay for a single named tile.

Reuses the standard eval pipeline (``evaluate`` -> ``save_eval_visualizations``) but
restricts the test subset to one filename, so the per-source viz cap never hides it.
Useful for reproducing a specific overlay (e.g. one a baseline produced) from a checkpoint.
"""

import argparse
import os
import sys

import numpy as np
import torch

from training.polygons.train import MaskRCNNPolygonTrainer, evaluate
from milliontrees import get_dataset
from milliontrees.datasets.milliontrees_dataset import MillionTreesSubset

sys.modules['__main__'].MaskRCNNPolygonTrainer = MaskRCNNPolygonTrainer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--root-dir",
                    default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    ap.add_argument("--split-scheme", default="zeroshot",
                    choices=["random", "zeroshot", "crossgeometry"])
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--filename", required=True,
                    help="Exact image filename in the split CSV, e.g. "
                         "Hayachine_HY-EC1_Ortho_59_Takeshige_et_al_2025.png")
    ap.add_argument("--viz-dir", required=True)
    ap.add_argument("--score-threshold", type=float, default=0.1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading checkpoint: {args.checkpoint}")
    model = MaskRCNNPolygonTrainer.load_from_checkpoint(args.checkpoint, weights_only=False)
    model.model.roi_heads.score_thresh = args.score_threshold
    model.eval()
    model = model.to(device)

    dataset = get_dataset(
        "TreePolygons",
        root_dir=args.root_dir,
        split_scheme=args.split_scheme,
        image_size=args.image_size,
    )
    test_subset = dataset.get_subset("test")

    # test_subset.indices index into dataset; _input_array is the per-index filename.
    keep = [int(didx) for didx in test_subset.indices
            if dataset._input_array[int(didx)] == args.filename]
    if not keep:
        raise SystemExit(f"{args.filename!r} not found in {args.split_scheme} test subset")
    print(f"Matched {len(keep)} tile(s) for {args.filename}")

    # MillionTreesSubset (not torch Subset) so get_eval_loader can use dataset.collate.
    sub = MillionTreesSubset(dataset, np.asarray(keep), None, dataset.geometry_name)
    results, results_str = evaluate(
        model, dataset, sub,
        batch_size=1, device=device,
        viz_dir=args.viz_dir, viz_n_per_source=None,
    )
    print(results_str)
    print(f"Viz written under: {args.viz_dir}")


if __name__ == "__main__":
    main()
