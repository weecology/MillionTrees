from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def bytes_to_gb(size_in_bytes: Optional[int | float]) -> float:
    """Convert bytes to gigabytes."""
    if not size_in_bytes:
        return 0.0
    return float(size_in_bytes) / (1024**3)


def print_dataset_summary(
    dataset_name: str,
    version: str,
    data_dir: Path | str,
    split_scheme: str,
    n_annotations: int,
    n_total_images: int,
    n_train_images: int,
    n_test_images: int,
    n_available_sources: int,
    n_selected_sources: int,
    mini: bool = False,
    include_patterns: Optional[list[str]] = None,
    exclude_patterns: Optional[list[str]] = None,
) -> None:
    """Print a concise onboarding summary for dataset initialization."""
    mode = "mini" if mini else "full"
    print(f"[MillionTrees] Loaded {dataset_name} v{version} ({mode})")
    print(f"[MillionTrees] Data dir: {data_dir}")
    print(f"[MillionTrees] Split: {split_scheme} | Images train/test/total: "
          f"{n_train_images}/{n_test_images}/{n_total_images}")
    print(
        f"[MillionTrees] Annotations: {n_annotations} | Sources selected/available: "
        f"{n_selected_sources}/{n_available_sources}")
    if include_patterns:
        print(f"[MillionTrees] include_sources: {include_patterns}")
    if exclude_patterns:
        print(f"[MillionTrees] exclude_sources: {exclude_patterns}")


def plot_release_size_summary(
    output_path: str | Path,
    dataset_sizes: dict[str, int | float],
    title: str = "MillionTrees Dataset Release Sizes (Compressed)",
) -> Path:
    """Create a bar chart for release compressed sizes."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = list(dataset_sizes.keys())
    sizes_gb = [bytes_to_gb(dataset_sizes[n]) for n in names]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=180)
    bars = ax.bar(names, sizes_gb, color=["#1f77b4", "#2ca02c", "#ff7f0e"])
    ax.set_ylabel("Compressed Size (GB)")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, size in zip(bars, sizes_gb):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{size:.1f} GB",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def get_latest_release_sizes() -> dict[str, int]:
    """Collect latest compressed sizes from dataset class metadata."""
    from milliontrees.datasets.TreeBoxes import TreeBoxesDataset
    from milliontrees.datasets.TreePoints import TreePointsDataset
    from milliontrees.datasets.TreePolygons import TreePolygonsDataset

    dataset_classes = {
        "TreeBoxes": TreeBoxesDataset,
        "TreePoints": TreePointsDataset,
        "TreePolygons": TreePolygonsDataset,
    }

    sizes: dict[str, int] = {}
    for name, klass in dataset_classes.items():
        versions = klass._versions_dict
        latest_version = max(versions.keys(),
                             key=lambda v: tuple(map(int, v.split("."))))
        sizes[name] = int(versions[latest_version]["compressed_size"])
    return sizes


def save_sample_visualization(
    dataset,
    output_path: str | Path,
    split: str = "train",
    index: int = 0,
) -> Path:
    """Save a single sample image with overlaid annotations."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subset = dataset.get_subset(split)
    _, image, targets = subset[index]

    image_np = image.detach().cpu().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))

    fig, ax = plt.subplots(figsize=(4, 4), dpi=180)
    ax.imshow(image_np)

    if dataset.dataset_name == "TreeBoxes":
        boxes = targets["y"].detach().cpu().numpy()
        for xmin, ymin, xmax, ymax in boxes:
            ax.add_patch(
                plt.Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=0.6,
                ))
    elif dataset.dataset_name == "TreePoints":
        points = targets["y"].detach().cpu().numpy()
        if len(points):
            ax.scatter(points[:, 0], points[:, 1], c="yellow", s=4)
    else:
        masks = targets["y"].detach().cpu().numpy()
        if len(masks):
            overlay = masks.sum(axis=0)
            ax.imshow(np.ma.masked_where(overlay == 0, overlay),
                      cmap="magma",
                      alpha=0.35)

    ax.set_title(f"{dataset.dataset_name} {split} sample")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
