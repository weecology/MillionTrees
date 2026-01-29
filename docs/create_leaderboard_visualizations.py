#!/usr/bin/env python3
"""Create panel visualizations for leaderboard showing predictions across datasets and splits."""

import os
import argparse
import warnings
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from PIL import Image
import pandas as pd
from deepforest import main as df_main
from deepforest.utilities import read_file, format_geometry
from deepforest.visualize import plot_results
import geopandas as gpd

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader

warnings.filterwarnings("ignore")


def get_sample_image(dataset, split_scheme: str, index: int = 0, dataset_type: str = "points"):
    """Get a single sample image from the test set.
    
    For polygons, try multiple indices to find one with visible annotations.
    """
    test_dataset = dataset.get_subset("test")
    test_loader = get_eval_loader("standard", test_dataset, batch_size=1)
    
    # For polygons, try different indices to find one with visible annotations
    if dataset_type == "polygons":
        # Try indices 0, 1, 2, 3, 4 to find one with annotations
        for idx in [index, 1, 2, 3, 4, 5]:
            for i, (metadata, images, targets) in enumerate(test_loader):
                if i == idx:
                    # Check if this image has annotations
                    gt = targets[0].get("y", targets[0].get("bboxes", None))
                    if gt is not None and len(gt) > 0:
                        return metadata[0], images[0], targets[0]
    
    # Default behavior for points/boxes or if no good polygon image found
    for i, (metadata, images, targets) in enumerate(test_loader):
        if i == index:
            return metadata[0], images[0], targets[0]
    return None, None, None


def _map_labels_to_int(values: pd.Series, model) -> np.ndarray:
    """Map label values to integers using model's label_dict."""
    if hasattr(model, 'label_dict') and model.label_dict:
        return values.map(model.label_dict).values.astype(np.int64)
    # If no label_dict, assume all labels are 0 (Tree class)
    return np.zeros(len(values), dtype=np.int64)


def predict_deepforest_points(image_tensor: torch.Tensor, model, dataset):
    """Run DeepForest prediction on points dataset."""
    channels_first = image_tensor.permute(1, 2, 0).numpy() * 255
    pred = model.predict_image(channels_first)
    
    if pred is None or len(pred) == 0:
        return {"y": torch.zeros((0, 2), dtype=torch.float32), 
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,), dtype=torch.float32)}
    
    pred["geometry"] = gpd.GeoSeries(pred["geometry"]).centroid
    pred[["x", "y"]] = pred["geometry"].apply(lambda g: pd.Series([g.x, g.y]))
    
    return {
        "y": torch.tensor(pred[["x", "y"]].values.astype("float32")),
        "labels": torch.tensor(_map_labels_to_int(pred.label, model)),
        "scores": torch.tensor(pred.score.values.astype("float32"))
    }


def predict_deepforest_boxes(image_tensor: torch.Tensor, model, dataset):
    """Run DeepForest prediction on boxes dataset."""
    channels_first = image_tensor.permute(1, 2, 0).numpy() * 255
    pred = model.predict_image(channels_first)
    
    if pred is None or len(pred) == 0:
        return {"y": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,), dtype=torch.float32)}
    
    return {
        "y": torch.tensor(pred[["xmin", "ymin", "xmax", "ymax"]].values.astype("float32")),
        "labels": torch.tensor(_map_labels_to_int(pred.label, model)),
        "scores": torch.tensor(pred.score.values.astype("float32"))
    }


def predict_deepforest_polygons(image_tensor: torch.Tensor, model, dataset):
    """Run DeepForest prediction on polygons dataset (returns boxes)."""
    channels_first = image_tensor.permute(1, 2, 0).numpy() * 255
    pred = model.predict_image(channels_first)
    
    if pred is None or len(pred) == 0:
        return {"y": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,), dtype=torch.float32)}
    
    return {
        "y": torch.tensor(pred[["xmin", "ymin", "xmax", "ymax"]].values.astype("float32")),
        "labels": torch.tensor(_map_labels_to_int(pred.label, model)),
        "scores": torch.tensor(pred.score.values.astype("float32"))
    }


def predict_sam3_points(image_tensor: torch.Tensor, model, processor, device, text_prompt="tree"):
    """Run SAM3 prediction on points dataset."""
    from PIL import Image as PILImage
    
    # Convert to PIL
    img = (image_tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    pil_image = PILImage.fromarray(img)
    
    inputs = processor(images=[pil_image], text=[text_prompt], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=0.5, mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )
    
    res = results[0]
    masks = res.get("masks", None)
    scores = res.get("scores", None)
    
    if masks is None or len(masks) == 0:
        return {"y": torch.zeros((0, 2), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,), dtype=torch.float32)}
    
    # Convert masks to centroids
    masks_t = torch.as_tensor(masks, dtype=torch.uint8, device=device)
    if masks_t.dim() == 2:
        masks_t = masks_t.unsqueeze(0)
    if masks_t.dim() == 4 and masks_t.shape[1] == 1:
        masks_t = masks_t[:, 0]
    
    # Calculate centroids
    centroids = []
    for mask in masks_t:
        y_coords, x_coords = torch.where(mask > 0)
        if len(y_coords) > 0:
            centroid_y = y_coords.float().mean().item()
            centroid_x = x_coords.float().mean().item()
            centroids.append([centroid_x, centroid_y])
    
    if len(centroids) == 0:
        return {"y": torch.zeros((0, 2), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,), dtype=torch.float32)}
    
    return {
        "y": torch.tensor(centroids, dtype=torch.float32),
        "labels": torch.zeros((len(centroids),), dtype=torch.int64),
        "scores": torch.as_tensor(scores[:len(centroids)], dtype=torch.float32).cpu()
    }


def predict_sam3_boxes(image_tensor: torch.Tensor, model, processor, device, text_prompt="tree"):
    """Run SAM3 prediction on boxes dataset."""
    from PIL import Image as PILImage
    from torchvision.ops import masks_to_boxes
    
    img = (image_tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    pil_image = PILImage.fromarray(img)
    
    inputs = processor(images=[pil_image], text=[text_prompt], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=0.5, mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )
    
    res = results[0]
    masks = res.get("masks", None)
    scores = res.get("scores", None)
    
    if masks is None or len(masks) == 0:
        return {"y": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,), dtype=torch.float32)}
    
    masks_t = torch.as_tensor(masks, dtype=torch.uint8, device=device)
    if masks_t.dim() == 2:
        masks_t = masks_t.unsqueeze(0)
    if masks_t.dim() == 4 and masks_t.shape[1] == 1:
        masks_t = masks_t[:, 0]
    
    masks_bool = masks_t.bool()
    boxes = masks_to_boxes(masks_bool).cpu()
    
    return {
        "y": boxes,
        "labels": torch.zeros((len(boxes),), dtype=torch.int64),
        "scores": torch.as_tensor(scores[:len(boxes)], dtype=torch.float32).cpu()
    }


def predict_sam3_polygons(image_tensor: torch.Tensor, model, processor, device, text_prompt="tree"):
    """Run SAM3 prediction on polygons dataset."""
    from PIL import Image as PILImage
    
    img = (image_tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    pil_image = PILImage.fromarray(img)
    
    inputs = processor(images=[pil_image], text=[text_prompt], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=0.5, mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )
    
    res = results[0]
    masks = res.get("masks", None)
    scores = res.get("scores", None)
    
    if masks is None or len(masks) == 0:
        return {"y": torch.zeros((0, 448, 448), dtype=torch.bool),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,), dtype=torch.float32)}
    
    masks_t = torch.as_tensor(masks, dtype=torch.uint8, device=device)
    if masks_t.dim() == 2:
        masks_t = masks_t.unsqueeze(0)
    if masks_t.dim() == 4 and masks_t.shape[1] == 1:
        masks_t = masks_t[:, 0]
    
    masks_bool = masks_t.bool().cpu()
    
    return {
        "y": masks_bool,
        "labels": torch.zeros((masks_bool.shape[0],), dtype=torch.int64),
        "scores": torch.as_tensor(scores[:masks_bool.shape[0]], dtype=torch.float32).cpu()
    }


def plot_points_prediction(ax, image, gt, pred_df, title=""):
    """Plot points predictions on an axis."""
    ax.imshow(image)
    ax.set_title(title, fontsize=8)
    ax.axis('off')
    
    # Ground truth - purple for color blindness
    if len(gt) > 0:
        gt_points = gt.numpy()
        ax.scatter(gt_points[:, 0], gt_points[:, 1], c='purple', s=20, marker='+', 
                  linewidths=2, label='Ground Truth', alpha=0.7)
    
    # Predictions - orange for color blindness
    if isinstance(pred_df, dict) and len(pred_df.get("y", [])) > 0:
        pred_points = pred_df["y"].numpy()
        scores = pred_df.get("scores", torch.ones(len(pred_points))).numpy()
        ax.scatter(pred_points[:, 0], pred_points[:, 1], c='orange', s=15, marker='o',
                  linewidths=1, label='Prediction', alpha=0.6, edgecolors='darkorange')
    elif isinstance(pred_df, pd.DataFrame) and len(pred_df) > 0:
        if "x" in pred_df.columns and "y" in pred_df.columns:
            ax.scatter(pred_df["x"], pred_df["y"], c='orange', s=15, marker='o',
                      linewidths=1, label='Prediction', alpha=0.6, edgecolors='darkorange')


def plot_boxes_prediction(ax, image, gt, pred, title=""):
    """Plot boxes predictions on an axis."""
    ax.imshow(image)
    ax.set_title(title, fontsize=8)
    ax.axis('off')
    
    # Ground truth - purple for color blindness
    if isinstance(gt, torch.Tensor) and len(gt) > 0:
        gt_boxes = gt.numpy()
        for i, box in enumerate(gt_boxes):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                    linewidth=2, edgecolor='purple', 
                                    facecolor='none', label='Ground Truth' if i == 0 else '')
            ax.add_patch(rect)
    
    # Predictions - orange for color blindness
    if isinstance(pred, dict) and len(pred.get("y", [])) > 0:
        pred_boxes = pred["y"].numpy()
        for i, box in enumerate(pred_boxes):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                    linewidth=1.5, edgecolor='orange',
                                    facecolor='none', linestyle='--', label='Prediction' if i == 0 else '')
            ax.add_patch(rect)


def plot_polygons_prediction(ax, image, gt, pred, title=""):
    """Plot polygons predictions on an axis."""
    ax.imshow(image)
    ax.set_title(title, fontsize=8)
    ax.axis('off')
    
    # Ground truth masks - purple for color blindness
    # Use custom colormap for purple
    from matplotlib.colors import ListedColormap
    purple_cmap = ListedColormap(['none', 'purple'])
    orange_cmap = ListedColormap(['none', 'orange'])
    
    if isinstance(gt, torch.Tensor) and len(gt) > 0:
        gt_masks = gt.numpy()
        for i, mask in enumerate(gt_masks):
            # Convert boolean mask to 0/1 for colormap
            mask_float = mask.astype(float)
            ax.imshow(mask_float, alpha=0.4, cmap=purple_cmap, vmin=0, vmax=1, 
                     label='Ground Truth' if i == 0 else '')
    
    # Predictions (can be boxes or masks) - orange for color blindness
    if isinstance(pred, dict):
        if "y" in pred:
            pred_y = pred["y"]
            if isinstance(pred_y, torch.Tensor):
                if pred_y.dim() == 3:  # Masks
                    for i, mask in enumerate(pred_y.numpy()):
                        mask_float = mask.astype(float)
                        ax.imshow(mask_float, alpha=0.4, cmap=orange_cmap, vmin=0, vmax=1,
                                 label='Prediction' if i == 0 else '')
                elif pred_y.dim() == 2 and pred_y.shape[1] == 4:  # Boxes
                    for i, box in enumerate(pred_y.numpy()):
                        x1, y1, x2, y2 = box
                        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                                linewidth=1.5, edgecolor='orange',
                                                facecolor='none', linestyle='--',
                                                label='Prediction' if i == 0 else '')
                        ax.add_patch(rect)


def _create_single_task_figure(
    root_dir: str,
    output_path: str,
    dataset_name: str,
    dataset_type: str,
    df_model,
    sam3_model,
    sam3_processor,
    device: str,
) -> None:
    """Create one figure for a single task: 3 rows (splits) x 2 cols (DeepForest, SAM3)."""
    splits = ["random", "zeroshot", "crossgeometry"]
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.06, wspace=0.01,
                  left=0.04, right=0.97, top=0.96, bottom=0.06)
    for row, split in enumerate(splits):
        try:
            dataset = get_dataset(dataset_name, root_dir=root_dir, split_scheme=split, download=False)
            metadata, image_tensor, target = get_sample_image(
                dataset, split, index=0, dataset_type=dataset_type
            )
            if metadata is None:
                continue
            image = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            gt = target.get("y", target.get("bboxes"))
            ax_df = fig.add_subplot(gs[row, 0])
            if dataset_type == "points":
                pred_df = predict_deepforest_points(image_tensor, df_model, dataset)
                plot_points_prediction(ax_df, image, target["y"], pred_df, "DeepForest")
            elif dataset_type == "boxes":
                pred_df = predict_deepforest_boxes(image_tensor, df_model, dataset)
                plot_boxes_prediction(ax_df, image, target["y"], pred_df, "DeepForest")
            else:
                pred_df = predict_deepforest_polygons(image_tensor, df_model, dataset)
                plot_polygons_prediction(ax_df, image, gt, pred_df, "DeepForest")
            ax_sam3 = fig.add_subplot(gs[row, 1])
            if sam3_model is not None:
                if dataset_type == "points":
                    pred_sam3 = predict_sam3_points(image_tensor, sam3_model, sam3_processor, device)
                    plot_points_prediction(ax_sam3, image, target["y"], pred_sam3, "SAM3")
                elif dataset_type == "boxes":
                    pred_sam3 = predict_sam3_boxes(image_tensor, sam3_model, sam3_processor, device)
                    plot_boxes_prediction(ax_sam3, image, target["y"], pred_sam3, "SAM3")
                else:
                    pred_sam3 = predict_sam3_polygons(image_tensor, sam3_model, sam3_processor, device)
                    plot_polygons_prediction(ax_sam3, image, gt, pred_sam3, "SAM3")
            else:
                ax_sam3.text(0.5, 0.5, "SAM3\nNot Available",
                             ha='center', va='center', transform=ax_sam3.transAxes)
                ax_sam3.axis('off')
        except Exception as e:
            print(f"Error processing {dataset_name} {split}: {e}")
    for row, split in enumerate(splits):
        fig.text(0.01, 0.83 - row * 0.31, split.replace('_', ' ').title(),
                 ha='center', va='center', fontsize=10, fontweight='bold', rotation=90)
    fig.suptitle(f"{dataset_name}: Model Predictions by Split", fontsize=12, fontweight='bold', y=0.995)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='purple', alpha=0.4, label='Ground Truth'),
        Patch(facecolor='orange', alpha=0.4, label='Prediction'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, 0.005), borderaxespad=0.2)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {dataset_name} visualization to {output_path}")


def create_task_figures(
    root_dir: str,
    output_dir: str,
    device: str = "cuda",
    hf_token: str = None,
) -> None:
    """Create one figure per task (TreePoints, TreeBoxes, TreePolygons) for manuscript use."""
    df_model = df_main.deepforest()
    df_model.load_model("weecology/deepforest-tree")
    df_model.eval()
    try:
        from transformers import Sam3Processor, Sam3Model
        if hf_token:
            sam3_model = Sam3Model.from_pretrained("facebook/sam3", token=hf_token).to(device)
            sam3_processor = Sam3Processor.from_pretrained("facebook/sam3", token=hf_token)
        else:
            sam3_model = Sam3Model.from_pretrained("facebook/sam3").to(device)
            sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")
    except Exception as e:
        print(f"Warning: Could not load SAM3: {e}")
        sam3_model = None
        sam3_processor = None
    os.makedirs(output_dir, exist_ok=True)
    for dataset_name, dataset_type, basename in [
        ("TreePoints", "points", "leaderboard_predictions_points.png"),
        ("TreeBoxes", "boxes", "leaderboard_predictions_boxes.png"),
        ("TreePolygons", "polygons", "leaderboard_predictions_polygons.png"),
    ]:
        out_path = os.path.join(output_dir, basename)
        _create_single_task_figure(
            root_dir, out_path, dataset_name, dataset_type,
            df_model, sam3_model, sam3_processor, device,
        )


def create_panel_figure(root_dir: str, output_path: str, device: str = "cuda", hf_token: str = None):
    """Create a panel figure with predictions for all datasets and splits."""
    df_model = df_main.deepforest()
    df_model.load_model("weecology/deepforest-tree")
    df_model.eval()
    try:
        from transformers import Sam3Processor, Sam3Model
        if hf_token:
            sam3_model = Sam3Model.from_pretrained("facebook/sam3", token=hf_token).to(device)
            sam3_processor = Sam3Processor.from_pretrained("facebook/sam3", token=hf_token)
        else:
            sam3_model = Sam3Model.from_pretrained("facebook/sam3").to(device)
            sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")
    except Exception as e:
        print(f"Warning: Could not load SAM3: {e}")
        sam3_model = None
        sam3_processor = None
    splits = ["random", "zeroshot", "crossgeometry"]
    datasets_info = [
        ("TreePoints", "points"),
        ("TreeBoxes", "boxes"),
        ("TreePolygons", "polygons")
    ]
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 6, figure=fig, hspace=0.06, wspace=0.02,
                  left=0.03, right=0.99, top=0.96, bottom=0.05)
    for row, (dataset_name, dataset_type) in enumerate(datasets_info):
        for col, split in enumerate(splits):
            try:
                dataset = get_dataset(dataset_name, root_dir=root_dir, split_scheme=split, download=False)
                metadata, image_tensor, target = get_sample_image(dataset, split, index=0, dataset_type=dataset_type)
                if metadata is None:
                    continue
                image = image_tensor.permute(1, 2, 0).numpy() * 255
                image = image.astype(np.uint8)
                ax_df = fig.add_subplot(gs[row, col*2])
                if dataset_type == "points":
                    pred_df = predict_deepforest_points(image_tensor, df_model, dataset)
                    plot_points_prediction(ax_df, image, target["y"], pred_df, f"{split} - DeepForest")
                elif dataset_type == "boxes":
                    pred_df = predict_deepforest_boxes(image_tensor, df_model, dataset)
                    plot_boxes_prediction(ax_df, image, target["y"], pred_df, f"{split} - DeepForest")
                else:
                    pred_df = predict_deepforest_polygons(image_tensor, df_model, dataset)
                    plot_polygons_prediction(ax_df, image, target.get("y", target.get("bboxes")), pred_df, f"{split} - DeepForest")
                ax_sam3 = fig.add_subplot(gs[row, col*2+1])
                if sam3_model is not None:
                    if dataset_type == "points":
                        pred_sam3 = predict_sam3_points(image_tensor, sam3_model, sam3_processor, device)
                        plot_points_prediction(ax_sam3, image, target["y"], pred_sam3, f"{split} - SAM3")
                    elif dataset_type == "boxes":
                        pred_sam3 = predict_sam3_boxes(image_tensor, sam3_model, sam3_processor, device)
                        plot_boxes_prediction(ax_sam3, image, target["y"], pred_sam3, f"{split} - SAM3")
                    else:
                        pred_sam3 = predict_sam3_polygons(image_tensor, sam3_model, sam3_processor, device)
                        plot_polygons_prediction(ax_sam3, image, target.get("y", target.get("bboxes")), pred_sam3, f"{split} - SAM3")
                else:
                    ax_sam3.text(0.5, 0.5, "SAM3\nNot Available", ha='center', va='center', transform=ax_sam3.transAxes)
                    ax_sam3.axis('off')
            except Exception as e:
                print(f"Error processing {dataset_name} {split}: {e}")
                continue
    for row, (dataset_name, _) in enumerate(datasets_info):
        fig.text(0.008, 0.83 - row*0.31, dataset_name, rotation=90, ha='center', va='center', fontsize=11, fontweight='bold')
    for col, split in enumerate(splits):
        fig.text(0.17 + col*0.32, 0.99, split.replace('_', ' ').title(), ha='center', va='center', fontsize=9, fontweight='bold')
    plt.suptitle("Model Predictions Across Datasets and Splits", fontsize=13, fontweight='bold', y=0.998)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='purple', alpha=0.4, label='Ground Truth'),
        Patch(facecolor='orange', alpha=0.4, label='Prediction'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, 0.004), borderaxespad=0.2, handletextpad=0.5, columnspacing=1.0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create leaderboard visualization panels")
    parser.add_argument("--root-dir", type=str,
                       default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees/"),
                       help="Dataset root directory")
    parser.add_argument("--output", type=str, default="docs/leaderboard_predictions.png",
                       help="Output path for the combined figure")
    parser.add_argument("--output-dir", type=str, default="docs",
                       help="Output directory for per-task figures (used with --split-by-task)")
    parser.add_argument("--split-by-task", action="store_true",
                       help="Create one figure per task for manuscript")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                       help="Device for SAM3")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"),
                       help="Hugging Face token for SAM3")
    args = parser.parse_args()
    if args.split_by_task:
        create_task_figures(args.root_dir, args.output_dir, args.device, args.hf_token)
    else:
        create_panel_figure(args.root_dir, args.output, args.device, args.hf_token)


if __name__ == "__main__":
    main()
