import os
import argparse
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from deepforest import main as df_main
from deepforest.utilities import read_file, format_geometry
from deepforest.visualize import plot_results
import geopandas as gpd

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader


def _map_labels_to_int(values: pd.Series, model: "df_main.deepforest") -> np.ndarray:
    if values.dtype == object:
        return values.apply(lambda x: model.label_dict.get(x, 0)).values.astype(np.int64)
    return values.values.astype(np.int64)


def format_deepforest_predictions(
    images: np.ndarray,
    metadata: torch.Tensor,
    targets: List[dict],
    model: "df_main.deepforest",
    dataset,
    batch_index: int,
) -> Tuple[List[dict], List[pd.DataFrame]]:
    """Run DeepForest on a batch and convert to MillionTrees point format."""
    warnings.filterwarnings("ignore")

    images_tensor = torch.tensor(images)
    predictions = model.predict_step(images_tensor, batch_index)

    batch_y_pred: List[dict] = []
    formatted_predictions: List[pd.DataFrame] = []

    for image_metadata, image_pred, image_targets, image in zip(
            metadata, predictions, targets, images_tensor):
        basename = dataset._filename_id_to_code[int(image_metadata[0])]

        if len(image_pred["boxes"]) == 0:
            y_pred = {
                "y": torch.zeros((0, 2), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,), dtype=torch.float32),
            }
            formatted_pred = pd.DataFrame(columns=["x", "y", "score", "label"])  # empty
        else:
            formatted_pred = format_geometry(image_pred)
            formatted_pred.root_dir = os.path.join(dataset._data_dir._str, "images")
            formatted_pred = read_file(formatted_pred)
            
            # Convert boxes to centroids
            formatted_pred["geometry"] = gpd.GeoSeries(formatted_pred["geometry"]).centroid
            formatted_pred[["x", "y"]] = formatted_pred["geometry"].apply(lambda g: pd.Series([g.x, g.y]))
            formatted_pred["image_path"] = basename

            y_pred = {
                "y": torch.tensor(formatted_pred[["x", "y"]].values.astype("float32")),
                "labels": torch.tensor(_map_labels_to_int(formatted_pred.label, model)),
                "scores": torch.tensor(formatted_pred.score.values.astype("float32")),
            }

        batch_y_pred.append(y_pred)
        formatted_predictions.append(formatted_pred)

    return batch_y_pred, formatted_predictions


def plot_eval_result(
    y_pred: dict,
    pred_df: pd.DataFrame,
    image_targets: dict,
    image_tensor: torch.Tensor,
    dataset,
    batch_index: int,
    output_dir: str = None,
):
    basename = (
        pred_df.image_path.unique()[0]
        if isinstance(pred_df, pd.DataFrame) and len(pred_df) > 0
        else "empty"
    )

    # Ground truth
    gt_df = read_file(pd.DataFrame(image_targets["y"].numpy(), columns=["x", "y"]))
    gt_df["label"] = "Tree"
    gt_df["score"] = 1

    # Predictions
    if isinstance(pred_df, pd.DataFrame) and len(pred_df) > 0:
        pred_vis_df = read_file(pred_df, root_dir=os.path.join(dataset._data_dir._str, "images"))
        pred_vis_df["geometry"] = gpd.GeoSeries(pred_vis_df["geometry"]).centroid
        pred_vis_df[["x", "y"]] = pred_vis_df["geometry"].apply(lambda g: pd.Series([g.x, g.y]))
        if "label" not in pred_vis_df.columns:
            pred_vis_df["label"] = "Tree"
    else:
        pred_vis_df = pred_df

    # Image channel-last, 0-255
    image = image_tensor.permute(1, 2, 0).numpy() * 255

    pred_vis_df.root_dir = os.path.join(dataset._data_dir._str, "images")
    fig = plot_results(pred_vis_df, gt_df, image=image.astype("int32"))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{batch_index:06d}_{basename}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

def main():
    parser = argparse.ArgumentParser(description="Run baseline DeepForest on TreePoints using centroid conversion.")
    parser.add_argument(
        "--root-dir",
        type=str,
        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees/"),
        help="Dataset root directory",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--plot-interval", type=int, default=1000, help="Plot every Nth image; set 0 to disable plotting")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--mini", action="store_true", help="Use mini datasets for fast dev")
    parser.add_argument("--download", action="store_true", help="Download dataset if missing")
    args = parser.parse_args()

    # Load model
    model = df_main.deepforest()
    model.load_model("weecology/deepforest-tree")
    model.eval()

    # Load dataset
    point_dataset = get_dataset("TreePoints", root_dir=args.root_dir, mini=args.mini, download=args.download)
    test_subset = point_dataset.get_subset("test")
    test_loader = get_eval_loader("standard", test_subset, batch_size=args.batch_size)

    print(f"There are {len(test_loader)} batches in the test loader")

    all_y_pred: List[dict] = []
    all_y_true: List[dict] = []

    batch_index = 0
    for batch in test_loader:
        metadata, images, targets = batch
        mt_preds, df_preds = format_deepforest_predictions(images, metadata, targets, model, point_dataset, batch_index)

        for image_metadata, y_pred, pred, image_targets, image in zip(metadata, mt_preds, df_preds, targets, images):
            if args.plot_interval and args.plot_interval > 0 and (batch_index % args.plot_interval == 0):
                plot_eval_result(y_pred, pred, image_targets, image, point_dataset, batch_index, args.output_dir)

            all_y_pred.append(y_pred)
            all_y_true.append(image_targets)
            batch_index += 1

        if args.max_batches is not None and batch_index >= args.max_batches:
            break

    # Evaluate using dataset's metric implementation
    results, results_str = point_dataset.eval(all_y_pred, all_y_true, test_subset.metadata_array[:len(all_y_true)])
    print(results_str)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "results_points.txt"), "w", encoding="utf-8") as f:
            f.write(results_str)


if __name__ == "__main__":
    main()


