#!/usr/bin/env python3
"""Visualize annotation counts per source from the most recent MillionTrees release."""

import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root / "src"))

from milliontrees.datasets.TreeBoxes import TreeBoxesDataset
from milliontrees.datasets.TreePoints import TreePointsDataset
from milliontrees.datasets.TreePolygons import TreePolygonsDataset

DATASETS = [
    ("TreeBoxes", TreeBoxesDataset, "Boxes"),
    ("TreePoints", TreePointsDataset, "Points"),
    ("TreePolygons", TreePolygonsDataset, "Polygons"),
]


def get_latest_version(dataset_class):
    versions = getattr(dataset_class, "_versions_dict", {})
    if not versions:
        return None
    def sort_key(v):
        try:
            return tuple(map(int, v.split(".")))
        except ValueError:
            return (0, 0)
    return max((v for v in versions if v != "0.0"), key=sort_key, default=None)


def derive_local_zip_path(url: str) -> Optional[Path]:
    if not url or "data.rc.ufl.edu/pub/" not in url:
        return None
    tail = url.split("data.rc.ufl.edu/pub/", 1)[1]
    if tail.startswith("ewhite/"):
        tail = tail[len("ewhite/"):]
    return Path("/orange/ewhite/web/public") / tail


def read_source_counts_from_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    if "source" not in df.columns:
        return pd.DataFrame()
    return df.groupby("source", as_index=False).size().rename(columns={"size": "count"})


def load_release_data(root_dir: Path) -> pd.DataFrame:
    rows = []
    for dataset_name, dataset_class, annotation_type in DATASETS:
        version = get_latest_version(dataset_class)
        if not version:
            continue
        extracted_csv = root_dir / f"{dataset_name}_v{version}" / "random.csv"
        if extracted_csv.exists():
            counts = read_source_counts_from_csv(extracted_csv)
        else:
            versions_dict = getattr(dataset_class, "_versions_dict", {})
            info = versions_dict.get(version, {})
            url = info.get("download_url", "")
            zip_path = derive_local_zip_path(url)
            if zip_path and zip_path.exists():
                import zipfile, io, csv
                with zipfile.ZipFile(zip_path, "r") as zf:
                    names = [n for n in zf.namelist() if n.lower().endswith("random.csv")]
                    if not names:
                        continue
                    name = min(names, key=lambda s: s.count("/"))
                    with zf.open(name) as f:
                        reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
                        df = pd.DataFrame(reader)
                    if "source" not in df.columns:
                        continue
                    counts = df.groupby("source", as_index=False).size().rename(columns={"size": "count"})
            else:
                print(f"Warning: No data for {dataset_name}")
                continue
        counts["annotation_type"] = annotation_type
        rows.append(counts)
    if not rows:
        raise FileNotFoundError(f"No release data at {root_dir}")
    df = pd.concat(rows, ignore_index=True)
    # Exclude unsupervised sources
    mask = df["source"].str.contains("unsupervised", case=False, na=False)
    return df[~mask]


def create_annotation_plot(df: pd.DataFrame, output_path: Path) -> None:
    pivot = df.pivot(index="source", columns="annotation_type", values="count").fillna(0)
    for col in ["Polygons", "Boxes", "Points"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["Polygons", "Boxes", "Points"]]
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=True).drop(columns="total")

    fig, ax = plt.subplots(figsize=(10, max(8, len(pivot) * 0.35)))
    pivot.plot(ax=ax, kind="barh", stacked=True,
               color=["#e9967a", "#daa520", "#90ee90"],
               width=0.7, legend=True)
    ax.set_xlabel("Annotations", fontsize=11)
    ax.set_ylabel("")
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.legend(title="Annotation Type", loc="upper right", frameon=True)
    ax.set_xlim(0, None)
    n_sources = df["source"].nunique()
    total_ann = int(df["count"].sum())
    ax.text(0.02, 0.98, f"Total Datasets: {n_sources}\nTotal Annotations: {total_ann:,}",
            transform=ax.transAxes, verticalalignment="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    plt.tight_layout()
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=Path,
                        default=Path(os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees")),
                        help="Root dir with extracted datasets")
    parser.add_argument("-o", "--output", type=Path, default=Path("annotation_counts_by_source.svg"))
    args = parser.parse_args()
    df = load_release_data(args.root_dir)
    print(f"Loaded {df['count'].sum():,} annotations from {df['source'].nunique()} sources")
    create_annotation_plot(df, args.output)


if __name__ == "__main__":
    main()
