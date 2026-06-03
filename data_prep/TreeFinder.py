"""
Reproducible processor for Wang et al. 2025 (TreeFinder).

Dataset: https://www.kaggle.com/datasets/zhihaow/tree-finder
Mirror: https://huggingface.co/datasets/zhwang1/TreeFinder

Tiles are 224×224 GeoTIFFs with bands [R, G, B, NIR, dead-tree mask].
Dead-tree pixels (label=1) are vectorized to polygons for TreePolygons.

Download (Hugging Face) then run:
  TREE_FINDER_ROOT=/path/to/TreeFinder uv run python data_prep/TreeFinder.py

Expected layout under TREE_FINDER_ROOT:
  tile_info224_v3.csv
  tiles224_v3.zip  (contains tiles224_v3/tiles224_v3/*.tif)
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Polygon

SOURCE_NAME = "Wang et al. 2025"
ZIP_TILE_PREFIX = "tiles224_v3/tiles224_v3/"
MIN_POLYGON_AREA = 4
NO_DATA_LABEL = 255

DATASET_ROOT = Path(
    os.environ.get("TREE_FINDER_ROOT",
                   "/orange/ewhite/DeepForest/TreeFinder")).resolve()
META_CSV = DATASET_ROOT / "tile_info224_v3.csv"
ZIP_PATH = DATASET_ROOT / "tiles224_v3.zip"
IMAGES_DIR = DATASET_ROOT / "images"
OUT_CSV = DATASET_ROOT / "annotations.csv"
PREVIEWS_DIR = DATASET_ROOT / "previews"


def _polygons_from_label(label: np.ndarray) -> list[Polygon]:
    mask = (label == 1).astype(np.uint8)
    num, labels = cv2.connectedComponents(mask, connectivity=8)
    polygons: list[Polygon] = []
    for comp_id in range(1, num):
        comp = (labels == comp_id).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < MIN_POLYGON_AREA:
            continue
        pts = contour.reshape(-1, 2)
        poly = Polygon([(float(x), float(y)) for x, y in pts])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < MIN_POLYGON_AREA:
            continue
        polygons.append(poly)
    return polygons


def _read_tile_arrays(tif_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(io.BytesIO(tif_bytes)) as src:
        arr = src.read()
    rgb = np.transpose(arr[:3], (1, 2, 0))
    label = arr[4].astype(np.uint8)
    label[label == NO_DATA_LABEL] = 0
    return rgb, label


def build_annotations(max_tiles: int | None = None) -> pd.DataFrame:
    if not META_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {META_CSV}")
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Tile archive not found: {ZIP_PATH}")

    meta = pd.read_csv(META_CSV)
    meta = meta[meta["LabelSize"] > 0]
    if max_tiles is not None:
        meta = meta.head(max_tiles)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    with zipfile.ZipFile(ZIP_PATH) as zf:
        for _, record in meta.iterrows():
            zip_name = ZIP_TILE_PREFIX + record["FileName"]
            tif_bytes = zf.read(zip_name)
            rgb, label = _read_tile_arrays(tif_bytes)
            polygons = _polygons_from_label(label)
            if not polygons:
                continue

            png_name = Path(record["FileName"]).with_suffix(".png").name
            png_path = IMAGES_DIR / png_name
            if not png_path.exists():
                cv2.imwrite(str(png_path),
                            cv2.cvtColor(rgb.astype(np.uint8),
                                         cv2.COLOR_RGB2BGR))

            for poly in polygons:
                xmin, ymin, xmax, ymax = poly.bounds
                rows.append({
                    "image_path": str(png_path.resolve()),
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "label": "Tree",
                    "geometry": poly.wkt,
                    "source": SOURCE_NAME,
                })

    if not rows:
        raise RuntimeError("No polygon annotations were produced.")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    return df


def generate_previews(df: pd.DataFrame, out_dir: Path, n: int = 1) -> list[str]:
    from deepforest.utilities import read_file
    from deepforest.visualize import plot_results

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for image_path in df["image_path"].drop_duplicates().head(n):
        subset = df[df["image_path"] == image_path]
        img = cv2.imread(image_path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gdf = read_file(subset.copy(), root_dir=str(Path(image_path).parent))
        gdf.root_dir = str(Path(image_path).parent)
        basename = Path(image_path).stem
        plot_results(gdf,
                     savedir=str(out_dir),
                     basename=basename,
                     image=rgb,
                     show=False)
        outputs.append(str(out_dir / f"{basename}.png"))
    return outputs


def main():
    df = build_annotations()
    print(f"Wrote {len(df)} annotations across "
          f"{df['image_path'].nunique()} images -> {OUT_CSV}")
    previews = generate_previews(df, PREVIEWS_DIR, n=3)
    for path in previews:
        print(f"Preview: {path}")


if __name__ == "__main__":
    main()
