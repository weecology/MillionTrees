"""Evaluate the DeepForest TreeFormer point model on TreePoints.

DeepForest now ships a TreeFormer-based keypoint head that predicts tree
points directly (no box->centroid conversion). The architecture is
``treeformer`` and the default checkpoint is
``weecology/deepforest-tree-point`` on HuggingFace.

The point model requires DeepForest > 2.1.0 (the keypoint workflow was
merged after that release). Until the next PyPI release, install from
git:

    uv add 'deepforest @ git+https://github.com/weecology/DeepForest.git@main'

Note: the bundled ``conf/point.yaml`` is missing from the published
wheel (MANIFEST.in upstream issue), so we pass the same settings via
``config_args`` instead of ``config='point'``.

This script mirrors ``baseline_points.py`` but uses the native point
predictions instead of converting RetinaNet boxes to centroids.
"""

import argparse
import os
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from deepforest import main as df_main
from deepforest.utilities import format_geometry, read_file
from deepforest.visualize import plot_results

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader


def _map_labels_to_int(values: pd.Series,
                       model: "df_main.deepforest") -> np.ndarray:
    if values.dtype == object:
        return values.apply(lambda x: model.label_dict.get(x, 0)).values.astype(
            np.int64)
    return values.values.astype(np.int64)


def format_treeformer_predictions(
    images: np.ndarray,
    metadata: torch.Tensor,
    targets: List[dict],
    model: "df_main.deepforest",
    dataset,
    batch_index: int,
) -> Tuple[List[dict], List[pd.DataFrame]]:
    """Run TreeFormer on a batch and convert to MillionTrees point format."""
    warnings.filterwarnings("ignore")

    images_tensor = torch.tensor(images)
    predictions = model.predict_step(images_tensor, batch_index)

    batch_y_pred: List[dict] = []
    formatted_predictions: List[pd.DataFrame] = []

    for image_metadata, image_pred in zip(metadata, predictions):
        basename = dataset._filename_id_to_code[int(image_metadata[0])]

        if len(image_pred["points"]) == 0:
            y_pred = {
                "y": torch.zeros((0, 2), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,), dtype=torch.float32),
            }
            formatted_pred = pd.DataFrame(columns=["x", "y", "score", "label"])
        else:
            formatted_pred = format_geometry(image_pred, geom_type="point")
            formatted_pred["image_path"] = basename
            formatted_pred = read_file(
                formatted_pred,
                root_dir=os.path.join(dataset._data_dir._str, "images"),
                image_path=basename,
            )
            y_pred = {
                "y":
                    image_pred["points"].detach().cpu().to(torch.float32),
                "labels":
                    torch.tensor(_map_labels_to_int(formatted_pred.label,
                                                    model)),
                "scores":
                    image_pred["scores"].detach().cpu().to(torch.float32),
            }

        batch_y_pred.append(y_pred)
        formatted_predictions.append(formatted_pred)

    return batch_y_pred, formatted_predictions


def plot_eval_result(
    pred_df: pd.DataFrame,
    image_targets: dict,
    image_tensor: torch.Tensor,
    dataset,
    batch_index: int,
    output_dir: str = None,
):
    basename = (pred_df.image_path.unique()[0]
                if isinstance(pred_df, pd.DataFrame) and len(pred_df) > 0 else
                "empty")

    gt_df = pd.DataFrame(image_targets["y"].numpy(), columns=["x", "y"])
    gt_df["image_path"] = basename
    gt_df = read_file(
        gt_df,
        root_dir=os.path.join(dataset._data_dir._str, "images"),
        image_path=basename,
        label="Tree",
    )
    gt_df["label"] = "Tree"
    gt_df["score"] = 1

    if isinstance(pred_df, pd.DataFrame) and len(pred_df) > 0:
        pred_vis_df = pred_df.copy()
        if "label" not in pred_vis_df.columns:
            pred_vis_df["label"] = "Tree"
    else:
        pred_vis_df = pred_df

    image = image_tensor.permute(1, 2, 0).numpy() * 255

    pred_vis_df.root_dir = os.path.join(dataset._data_dir._str, "images")
    fig = plot_results(pred_vis_df, gt_df, image=image.astype("int32"))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{batch_index:06d}_{basename}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DeepForest TreeFormer point model on TreePoints.")
    parser.add_argument(
        "--root-dir",
        type=str,
        default=os.environ.get("MT_ROOT",
                               "/orange/ewhite/web/public/MillionTrees/"),
        help="Dataset root directory",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--plot-interval",
        type=int,
        default=1000,
        help="Plot every Nth image; set 0 to disable plotting",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--mini",
                        action="store_true",
                        help="Use mini datasets for fast dev")
    parser.add_argument("--download",
                        action="store_true",
                        help="Download dataset if missing")
    parser.add_argument(
        "--split-scheme",
        type=str,
        default="random",
        choices=["random", "zeroshot", "crossgeometry"],
        help="Dataset split scheme",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=None,
        help=("Override the model's peak score threshold (defaults to the "
              "value baked into config.score_thresh)."),
    )
    args = parser.parse_args()

    # Equivalent to conf/point.yaml in DeepForest main. Passed via
    # config_args because the YAML is not yet bundled in the published wheel.
    config_args = {
        "architecture": "treeformer",
        "model": {
            "name": "weecology/deepforest-tree-point",
            "revision": "main",
        },
        "patch_size": 512,
        "patch_overlap": 0.1,
        "score_thresh": (args.score_thresh
                         if args.score_thresh is not None else 0.3),
        "point": {
            "backbone": "pvt_v2_b3",
            "score_integration_radius": 5,
            "nms_distance_thresh": 5.0,
            "distance_threshold": 10.0,
        },
    }
    model = df_main.deepforest(config_args=config_args)
    model.eval()

    point_dataset = get_dataset(
        "TreePoints",
        root_dir=args.root_dir,
        mini=args.mini,
        download=args.download,
        split_scheme=args.split_scheme,
    )
    test_subset = point_dataset.get_subset("test")
    test_loader = get_eval_loader("standard",
                                  test_subset,
                                  batch_size=args.batch_size)

    print(f"There are {len(test_loader)} batches in the test loader")

    all_y_pred: List[dict] = []
    all_y_true: List[dict] = []

    batch_index = 0
    for batch in test_loader:
        metadata, images, targets = batch
        mt_preds, df_preds = format_treeformer_predictions(
            images, metadata, targets, model, point_dataset, batch_index)

        for y_pred, pred_df, image_targets, image in zip(mt_preds, df_preds,
                                                         targets, images):
            if (args.plot_interval and args.plot_interval > 0 and
                (batch_index % args.plot_interval == 0)):
                plot_eval_result(pred_df, image_targets, image, point_dataset,
                                 batch_index, args.output_dir)

            all_y_pred.append(y_pred)
            all_y_true.append(image_targets)
            batch_index += 1

        if args.max_batches is not None and batch_index >= args.max_batches:
            break

    results, results_str = point_dataset.eval(
        all_y_pred, all_y_true,
        test_subset.metadata_array[:len(all_y_true)])
    print(results_str)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "results_treeformer_points.txt"),
                  "w",
                  encoding="utf-8") as f:
            f.write(results_str)


if __name__ == "__main__":
    main()
