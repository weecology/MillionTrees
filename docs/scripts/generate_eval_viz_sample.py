#!/usr/bin/env python3
"""Regenerate docs/public/eval_visualization_sample.png using DeepForest + mini TreeBoxes.

Requires optional dev dependencies (``deepforest``). From the repo root::

    uv sync --dev
    uv run python docs/scripts/generate_eval_viz_sample.py --root-dir onboarding_data

Uses one eval batch, runs ``dataset.eval(..., viz_dir=...)``, then copies the first
written overlay PNG to ``docs/public/eval_visualization_sample.png``.

``include_unsupervised=True`` is set so ``onboarding_data/TreeBoxes_v0.12`` is used
instead of the ``TreeBoxes_supervised_v*`` directory name.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import warnings
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "docs" / "public" / "eval_visualization_sample.png"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        type=str,
        default="onboarding_data",
        help="MillionTrees data root (directory that contains TreeBoxes_v*).",
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        help="Use mini TreeBoxes URLs (requires working download host).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download dataset archive if the version folder is missing.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output PNG path.",
    )
    args = parser.parse_args()

    try:
        from deepforest import main as df_main
        from deepforest.utilities import format_geometry
    except ImportError as e:
        raise SystemExit(
            "deepforest is required. Install dev extras: uv sync --dev"
        ) from e

    from milliontrees import get_dataset
    from milliontrees.common.data_loaders import get_eval_loader

    warnings.filterwarnings("ignore")
    model = df_main.deepforest()
    model.load_model("weecology/deepforest-tree")
    model.eval()

    # Use full dataset folder name ``TreeBoxes_v*`` (matches repo ``onboarding_data``), not
    # ``TreeBoxes_supervised_v*`` used when include_unsupervised is False.
    dataset = get_dataset(
        "TreeBoxes",
        root_dir=args.root_dir,
        download=args.download,
        mini=args.mini,
        split_scheme="within-distribution",
        include_unsupervised=True,
    )
    test_subset = dataset.get_subset("test")
    test_loader = get_eval_loader("standard", test_subset, batch_size=4, num_workers=0)

    metadata, images, targets = next(iter(test_loader))
    predictions = model.predict_step(images, 0)

    all_y_pred, all_y_true = [], []
    for image_metadata, pred, target in zip(metadata, predictions, targets):
        if pred is None or len(pred["boxes"]) == 0:
            all_y_pred.append(
                {
                    "y": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                }
            )
        else:
            df = format_geometry(pred)
            all_y_pred.append(
                {
                    "y": torch.tensor(
                        df[["xmin", "ymin", "xmax", "ymax"]].values.astype("float32")
                    ),
                    "labels": torch.tensor(df.label.values.astype("int64")),
                    "scores": torch.tensor(df.score.values.astype("float32")),
                }
            )
        all_y_true.append(target)

    used_meta = test_subset.metadata_array[: len(all_y_true)]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _, _ = dataset.eval(
            all_y_pred,
            all_y_true,
            used_meta,
            viz_dir=str(tmp_path),
            viz_n_per_source=4,
        )
        pngs = sorted(tmp_path.rglob("*.png"))
        if not pngs:
            raise SystemExit("No visualization PNGs were written; check dataset / preds.")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pngs[0], args.out)
        print(f"Wrote {args.out} (from {pngs[0].relative_to(tmp_path)})")


if __name__ == "__main__":
    main()
