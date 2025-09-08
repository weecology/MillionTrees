import os
import argparse
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from deepforest import main as df_main
from deepforest.utilities import read_file, format_geometry
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader


def format_deepforest_predictions(
    images: np.ndarray,
    metadata: torch.Tensor,
    targets: List[dict],
    model: "df_main.deepforest",
    dataset,
    batch_index: int,
) -> Tuple[List[dict], List[pd.DataFrame]]:
    """Run DeepForest on a batch and convert to MillionTrees format.

    Returns a tuple of:
    - list of y_pred dicts for MillionTrees evaluation
    - list of original prediction DataFrames (for plotting)
    """
    warnings.filterwarnings("ignore")

    images_tensor = torch.tensor(images)
    predictions = model.predict_step(images_tensor, batch_index)

    batch_y_pred: List[dict] = []
    formatted_predictions: List[pd.DataFrame] = []

    for image_metadata, pred, image_targets, image in zip(
            metadata, predictions, targets, images_tensor):
        basename = dataset._filename_id_to_code[int(image_metadata[0])]

        if pred is None or len(pred["boxes"]) == 0:
            y_pred = {
                "y": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,), dtype=torch.float32),
            }
            formatted_pred = pd.DataFrame(
                columns=["xmin", "ymin", "xmax", "ymax", "score", "label"])  # empty
            formatted_pred.root_dir = os.path.join(dataset._data_dir._str,
                                                   "images")
            formatted_pred["image_path"] = basename
        else:
            formatted_pred = format_geometry(pred)
            formatted_pred.root_dir = os.path.join(dataset._data_dir._str,
                                                   "images")
            formatted_pred["image_path"] = basename

            y_pred = {
                "y": torch.tensor(
                    formatted_pred[["xmin", "ymin", "xmax",
                                     "ymax"]].values.astype("float32")),
                "labels": torch.tensor(formatted_pred.label.values.astype(
                    np.int64)),
                "scores": torch.tensor(
                    formatted_pred.score.values.astype("float32")),
            }

        batch_y_pred.append(y_pred)
        formatted_predictions.append(formatted_pred)

    return batch_y_pred, formatted_predictions


def _draw_boxes(ax, boxes: np.ndarray, color: str):
    for box in boxes:
        if len(box) != 4:
            continue
        xmin, ymin, xmax, ymax = box
        width = max(0.0, float(xmax) - float(xmin))
        height = max(0.0, float(ymax) - float(ymin))
        rect = patches.Rectangle((float(xmin), float(ymin)), width, height,
                                 linewidth=0.8,
                                 edgecolor=color,
                                 facecolor='none',
                                 alpha=0.8)
        ax.add_patch(rect)


def save_gallery(thumbnails: List[dict], rows: int, cols: int, dpi: int,
                 output_path: str):
    if len(thumbnails) == 0:
        return
    n = min(len(thumbnails), rows * cols)
    fig, axes = plt.subplots(rows,
                             cols,
                             figsize=(cols * 3, rows * 3),
                             dpi=dpi)
    axes = np.atleast_2d(axes)

    for idx in range(rows * cols):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        if idx < n:
            thumb = thumbnails[idx]
            ax.imshow(thumb["image"], interpolation='nearest')
            _draw_boxes(ax, thumb.get("gt_boxes", np.zeros((0, 4))),
                        color='orange')
            _draw_boxes(ax, thumb.get("pred_boxes", np.zeros((0, 4))),
                        color='royalblue')
            ax.set_title(thumb.get("title", ""), fontsize=8)
        ax.axis('off')

    plt.tight_layout(pad=0.2)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline DeepForest evaluation on TreeBoxes.")
    parser.add_argument(
        "--root-dir",
        type=str,
        default=os.environ.get("MT_ROOT",
                               "/orange/ewhite/web/public/MillionTrees"),
        help="Dataset root directory",
    )
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument(
        "--plot-interval",
        type=int,
        default=250,
        help="Plot every Nth image; set 0 to disable plotting",
    )
    parser.add_argument("--gallery-rows", type=int, default=2)
    parser.add_argument("--gallery-cols", type=int, default=3)
    parser.add_argument("--gallery-dpi", type=int, default=72)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    # Load model
    model = df_main.deepforest()
    model.load_model("weecology/deepforest-tree")
    model.eval()

    # Load dataset
    box_dataset = get_dataset("TreeBoxes",
                              download=False,
                              root_dir=args.root_dir)
    test_subset = box_dataset.get_subset("test")
    test_loader = get_eval_loader("standard",
                                  test_subset,
                                  batch_size=args.batch_size)

    print(f"There are {len(test_loader)} batches in the test loader")

    all_y_pred: List[dict] = []
    all_y_true: List[dict] = []

    batch_index = 0
    thumbnails: List[dict] = []
    max_thumbnails = args.gallery_rows * args.gallery_cols if args.output_dir else 0
    for batch in test_loader:
        metadata, images, targets = batch
        mt_preds, df_preds = format_deepforest_predictions(images, metadata,
                                                           targets, model,
                                                           box_dataset,
                                                           batch_index)

        for image_metadata, y_pred, pred, image_targets, image in zip(
                metadata, mt_preds, df_preds, targets, images):
            
            # Collect thumbnails for gallery
            if args.output_dir and len(thumbnails) < max_thumbnails:
                basename = (pred.image_path.unique()[0]
                            if isinstance(pred, pd.DataFrame)
                            and len(pred) > 0 else "")
                image_np = (image.permute(1, 2, 0).numpy() * 255).clip(
                    0, 255).astype("uint8")
                pred_boxes = y_pred.get("y", torch.zeros((0, 4))).detach().cpu().numpy()
                gt_boxes = image_targets["y"].detach().cpu().numpy()
                recall = box_dataset.metrics["recall"]._recall(
                    image_targets["y"], y_pred.get("y", torch.zeros((0, 4))),
                    iou_threshold=0.3)
                title = f"{basename} R@0.3={float(recall):.2f}"
                thumbnails.append({
                    "image": image_np,
                    "pred_boxes": pred_boxes,
                    "gt_boxes": gt_boxes,
                    "title": title
                })

            all_y_pred.append(y_pred)
            all_y_true.append(image_targets)
            batch_index += 1

        if args.max_batches is not None and batch_index >= args.max_batches:
            break

    results, results_str = box_dataset.eval(all_y_pred, all_y_true,
                                            test_subset.metadata_array)
    print(results_str)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "results_boxes.txt"), "w",
                  encoding="utf-8") as f:
            f.write(results_str)
        # Save a compact gallery image
        gallery_path = os.path.join(args.output_dir, "gallery_boxes.png")
        save_gallery(thumbnails, args.gallery_rows, args.gallery_cols,
                     args.gallery_dpi, gallery_path)


if __name__ == "__main__":
    main()


