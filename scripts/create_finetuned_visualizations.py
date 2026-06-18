"""Panel figures for fine-tuned MillionTrees models (manuscript / leaderboard).

One figure per geometry (TreePoints, TreeBoxes, TreePolygons). Each figure uses the
test split for the within-distribution and out-of-distribution fine-tuning tasks: rows are splits, columns
are ground truth vs fine-tuned predictions on the same image.

Loads checkpoints from ``training/<geometry>/outputs/<split>/checkpoints/``.
Writes PNG and SVG to ``docs/`` (and optional per-panel SVGs for layout in Illustrator).

Not run by ``submit_all.sh`` automatically; use ``slurm/visualize_finetuned.sbatch`` or
the visualization step in ``slurm/run_benchmark.sbatch`` after training finishes.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
FINETUNE_SPLITS = ("within-distribution", "out-of-distribution")

GEOMETRY_CONFIG = {
    "points": {
        "dataset": "TreePoints",
        "model_label": "TreeFormer",
        "basename": "leaderboard_predictions_points",
        "train_module": "training.points.train",
    },
    "boxes": {
        "dataset": "TreeBoxes",
        "model_label": "DeepForest",
        "basename": "leaderboard_predictions_boxes",
        "train_module": "training.boxes.train",
    },
    "polygons": {
        "dataset": "TreePolygons",
        "model_label": "Mask R-CNN",
        "basename": "leaderboard_predictions_polygons",
        "train_module": "training.polygons.train",
    },
}


def find_checkpoint(split_output_dir: Path) -> Path | None:
    ckpt_dir = split_output_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        return None
    preferred = [p for p in ckpts if "last" not in p.name.lower()]
    pool = preferred or ckpts
    return max(pool, key=lambda p: p.stat().st_mtime)


def get_test_sample(dataset, index: int = 0, *, geometry: str):
    test_subset = dataset.get_subset("test")
    loader = get_eval_loader("standard", test_subset, batch_size=1)
    indices = [index]
    if geometry == "polygons":
        indices = [index, 1, 2, 3, 4, 5]
    for i, (metadata, images, targets) in enumerate(loader):
        if i not in indices:
            continue
        t = targets[0]
        gt = t.get("y", t.get("bboxes"))
        if geometry == "polygons" and gt is not None and len(gt) > 0:
            return metadata[0], images[0], t
        if geometry != "polygons" and i == index:
            return metadata[0], images[0], t
    for i, (metadata, images, targets) in enumerate(loader):
        if i == index:
            return metadata[0], images[0], targets[0]
    return None, None, None


def image_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    return (image_tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)


def plot_points(ax, image, gt, pred, title: str):
    ax.imshow(image)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    if isinstance(gt, torch.Tensor) and len(gt) > 0:
        pts = gt.numpy()
        ax.scatter(pts[:, 0], pts[:, 1], c="purple", s=22, marker="+", linewidths=2, alpha=0.85)
    if isinstance(pred, dict) and len(pred.get("y", [])) > 0:
        pts = pred["y"].numpy()
        ax.scatter(
            pts[:, 0], pts[:, 1], c="orange", s=14, marker="o",
            linewidths=1, alpha=0.75, edgecolors="darkorange",
        )


def plot_boxes(ax, image, gt, pred, title: str):
    ax.imshow(image)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    if isinstance(gt, torch.Tensor) and len(gt) > 0:
        for box in gt.numpy():
            x1, y1, x2, y2 = box
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="purple", facecolor="none",
            ))
    if isinstance(pred, dict) and len(pred.get("y", [])) > 0:
        for box in pred["y"].numpy():
            x1, y1, x2, y2 = box
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor="orange",
                facecolor="none", linestyle="--",
            ))


def plot_polygons(ax, image, gt, pred, title: str):
    ax.imshow(image)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    purple_cmap = ListedColormap(["none", "purple"])
    orange_cmap = ListedColormap(["none", "orange"])
    if isinstance(gt, torch.Tensor) and len(gt) > 0:
        for mask in gt.numpy():
            ax.imshow(mask.astype(float), alpha=0.4, cmap=purple_cmap, vmin=0, vmax=1)
    if isinstance(pred, dict) and len(pred.get("y", [])) > 0:
        pred_y = pred["y"]
        if isinstance(pred_y, torch.Tensor) and pred_y.dim() == 3:
            for mask in pred_y.numpy():
                ax.imshow(mask.astype(float), alpha=0.4, cmap=orange_cmap, vmin=0, vmax=1)


def load_finetuned_model(geometry: str, checkpoint: Path, device: torch.device):
    if geometry == "boxes":
        from deepforest import main as df_main
        model = df_main.deepforest.load_from_checkpoint(str(checkpoint), weights_only=False)
        model.eval()
        model.to(device)
        return model, "predict_batch"

    if geometry == "points":
        from deepforest import main as df_main
        from training.points.train import predict_batch

        model = df_main.deepforest.load_from_checkpoint(str(checkpoint), weights_only=False)
        model.eval()
        model.to(device)
        return (model, predict_batch), "points_predict"

    if geometry == "polygons":
        from training.polygons.train import MaskRCNNPolygonTrainer, format_predictions_for_eval

        model = MaskRCNNPolygonTrainer.load_from_checkpoint(str(checkpoint), weights_only=False)
        model.eval()
        model.to(device)
        return (model, format_predictions_for_eval), "polygons_predict"

    raise ValueError(f"Unknown geometry: {geometry}")


def run_prediction(geometry: str, model_bundle, image_tensor: torch.Tensor, device: torch.device):
    if geometry == "boxes":
        from training.boxes.train import predict_batch

        model = model_bundle
        images = image_tensor.unsqueeze(0).to(device)
        return predict_batch(model, images)[0]

    if geometry == "points":
        model, predict_fn = model_bundle
        images = image_tensor.unsqueeze(0).to(device)
        return predict_fn(model, images)[0]

    model, format_fn = model_bundle
    images = image_tensor.unsqueeze(0).to(device)
    return format_fn(images, model, device)[0]


def save_figure(fig, base_path: Path):
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{base_path}.png", dpi=150, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{base_path}.svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {base_path}.png and {base_path}.svg")


def export_standalone_panel(
    plot_fn,
    image: np.ndarray,
    gt,
    pred,
    title: str,
    panel_path: Path,
    *,
    predictions_only: bool,
):
    """One SVG per panel so panels can be aligned separately in Illustrator."""
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    empty = {"y": torch.zeros((0, 2))}
    if predictions_only:
        plot_fn(ax, image, gt, pred, title)
    else:
        plot_fn(ax, image, gt, empty, title)
    fig.savefig(panel_path, format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def create_geometry_figure(
    geometry: str,
    root_dir: str,
    output_dir: Path,
    panel_dir: Path | None,
    device: torch.device,
    sample_index: int,
):
    cfg = GEOMETRY_CONFIG[geometry]
    splits = FINETUNE_SPLITS
    n_rows = len(splits)
    fig = plt.figure(figsize=(8, 4 * n_rows))
    gs = GridSpec(n_rows, 2, figure=fig, hspace=0.12, wspace=0.04,
                  left=0.08, right=0.98, top=0.94, bottom=0.06)

    plot_fn = {"points": plot_points, "boxes": plot_boxes, "polygons": plot_polygons}[geometry]
    col_titles = ("Ground truth", f"{cfg['model_label']} (fine-tuned)")

    for row, split in enumerate(splits):
        ckpt_path = find_checkpoint(ROOT / "training" / geometry / "outputs" / split)
        if ckpt_path is None:
            print(f"Skip {geometry}/{split}: no checkpoint under training/{geometry}/outputs/{split}")
            for col in range(2):
                ax = fig.add_subplot(gs[row, col])
                ax.text(0.5, 0.5, f"No checkpoint\n({split})", ha="center", va="center")
                ax.axis("off")
            continue

        print(f"{geometry}/{split}: {ckpt_path}")
        model_bundle = load_finetuned_model(geometry, ckpt_path, device)

        dataset = get_dataset(cfg["dataset"], root_dir=root_dir, split_scheme=split, download=False)
        metadata, image_tensor, target = get_test_sample(dataset, sample_index, geometry=geometry)
        if metadata is None:
            print(f"  No test sample for {split}")
            continue

        image = image_to_uint8(image_tensor)
        gt = target["y"]
        pred = run_prediction(geometry, model_bundle, image_tensor, device)

        ax_gt = fig.add_subplot(gs[row, 0])
        plot_fn(ax_gt, image, gt, {"y": torch.zeros((0, 2))}, col_titles[0])
        ax_pred = fig.add_subplot(gs[row, 1])
        plot_fn(ax_pred, image, gt, pred, col_titles[1])

        if panel_dir is not None:
            export_standalone_panel(
                plot_fn, image, gt, pred, col_titles[0],
                panel_dir / f"{geometry}_{split}_ground_truth.svg",
                predictions_only=False,
            )
            export_standalone_panel(
                plot_fn, image, gt, pred, col_titles[1],
                panel_dir / f"{geometry}_{split}_finetuned.svg",
                predictions_only=True,
            )

        fig.text(0.02, 0.88 - row * (0.82 / max(n_rows - 1, 1)), split.title(),
                 rotation=90, va="center", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle(
        f"{cfg['dataset']}: fine-tuned predictions ({cfg['model_label']})",
        fontsize=12, fontweight="bold",
    )
    fig.legend(
        handles=[
            Patch(facecolor="purple", alpha=0.5, label="Ground truth"),
            Patch(facecolor="orange", alpha=0.5, label="Fine-tuned prediction"),
        ],
        loc="lower center", ncol=2, fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.0),
    )

    base = output_dir / cfg["basename"]
    save_figure(fig, base)


def main():
    parser = argparse.ArgumentParser(description="Fine-tuned model panel figures (PNG + SVG).")
    parser.add_argument("--root-dir", type=str, default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "docs"),
                        help="Directory for combined leaderboard_predictions_* files")
    parser.add_argument("--panel-dir", type=str, default=None,
                        help="If set, also write one SVG per panel (for manuscript layout)")
    parser.add_argument("--geometries", nargs="+", default=list(GEOMETRY_CONFIG),
                        choices=list(GEOMETRY_CONFIG))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Test-set image index (polygons tries several if empty)")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    panel_dir = Path(args.panel_dir) if args.panel_dir else None

    for geometry in args.geometries:
        create_geometry_figure(
            geometry, args.root_dir, output_dir, panel_dir, device, args.sample_index,
        )


if __name__ == "__main__":
    main()
