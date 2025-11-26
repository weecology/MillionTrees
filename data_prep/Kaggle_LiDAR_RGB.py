"""
Reproducible processor for: Dubrovin et al. 2024 (Kaggle LiDAR+RGB)
Dataset: https://www.kaggle.com/datasets/sentinel3734/tree-detection-lidar-rgb

This script:
- Reads field_survey.geojson (point locations and plot IDs)
- Matches points to ortho/*.tif by plot ID
- Converts geospatial coordinates to pixel coordinates for each ortho
- Writes annotations.csv with columns: image_path (absolute), geometry (WKT POINT), source
- Generates a few preview overlays in previews/

No command-line arguments required. Optionally override dataset root via env var:
  DUBROVIN_DATASET_ROOT=/path/to/root python3 data_prep/Kaggle_LiDAR_RGB.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import cv2
import pandas as pd
try:
    import rasterio  # type: ignore
except Exception:
    rasterio = None  # lazy fallback if environment lacks rasterio
from shapely.geometry import Point
from shapely import wkt as shapely_wkt


DATASET_ROOT = Path(os.environ.get("DUBROVIN_DATASET_ROOT", "/orange/ewhite/DeepForest/Kaggle_LiDAR_RGB")).resolve()
FIELD_SURVEY = DATASET_ROOT / "field_survey.geojson"
ORTHO_DIR = DATASET_ROOT / "ortho"
OUT_CSV = DATASET_ROOT / "annotations.csv"
PREVIEWS_DIR = DATASET_ROOT / "previews"
SOURCE_NAME = "Dubrovin et al. 2024"


def _load_points(geojson_path: Path) -> list[dict]:
    with open(geojson_path, "r") as f:
        data = json.load(f)
    features = data.get("features", [])
    points: list[dict] = []
    for feat in features:
        geom = feat.get("geometry", {})
        props = feat.get("properties", {}) or {}
        if not geom or geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates", [])
        if not coords or len(coords) != 2:
            continue
        x, y = float(coords[0]), float(coords[1])
        plot_val = props.get("plot")
        if plot_val is None:
            continue
        try:
            plot_num = int(float(plot_val))
        except Exception:
            continue
        plot_id = str(plot_num).zfill(2)
        points.append({"plot_id": plot_id, "x": x, "y": y})
    if not points:
        raise RuntimeError("No valid Point features with 'plot' found in field_survey.geojson")
    return points


def _iter_orthos(ortho_dir: Path) -> Iterable[Path]:
    yield from sorted(ortho_dir.glob("*.tif"))


def build_annotations() -> str:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")
    if not FIELD_SURVEY.exists():
        raise FileNotFoundError(f"GeoJSON not found: {FIELD_SURVEY}")
    if not ORTHO_DIR.exists():
        raise FileNotFoundError(f"Ortho directory not found: {ORTHO_DIR}")

    if rasterio is None:
        # Fallback: if annotations already exist, reuse them
        if OUT_CSV.exists():
            return str(OUT_CSV)
        raise ModuleNotFoundError("rasterio is required to rebuild annotations from GeoJSON; install rasterio or provide annotations.csv")

    all_points = _load_points(FIELD_SURVEY)
    rows: list[dict] = []

    for ortho in _iter_orthos(ORTHO_DIR):
        # Plot ID is last token after underscore (e.g., ortho_03.tif -> "03")
        stem_parts = ortho.stem.split("_")
        plot_id = stem_parts[-1] if stem_parts else ""
        plot_id = str(plot_id).zfill(2)

        pts = [p for p in all_points if p["plot_id"] == plot_id]
        if not pts:
            continue

        with rasterio.open(ortho) as ds:
            for p in pts:
                # Convert from geospatial coords to pixel row/col
                row, col = ds.index(p["x"], p["y"])
                # geometry uses (x=col, y=row) in pixel coordinates
                geom_wkt = f"POINT ({float(col)} {float(row)})"
                rows.append({"image_path": str(ortho.resolve()), "geometry": geom_wkt, "source": SOURCE_NAME})

    if not rows:
        raise RuntimeError("No point annotations matched ortho images.")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    return str(OUT_CSV)


def generate_previews(csv_path: str, out_dir: Path, n: int = 3) -> list[str]:
    df = pd.read_csv(csv_path)
    if df.empty:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    images = df["image_path"].drop_duplicates().head(n).tolist()
    outputs: list[str] = []

    for i, img_path in enumerate(images, start=1):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        subset = df[df["image_path"] == img_path]
        for _, r in subset.iterrows():
            try:
                geom = shapely_wkt.loads(r["geometry"])
            except Exception:
                continue
            if isinstance(geom, Point):
                x, y = int(round(geom.x)), int(round(geom.y))
                cv2.circle(img_rgb, (x, y), 3, (255, 0, 0), -1)
        out_path = out_dir / f"preview_{i}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        outputs.append(str(out_path))
    return outputs


def main():
    out_csv = build_annotations()
    print(f"Wrote: {out_csv}")
    previews = generate_previews(out_csv, PREVIEWS_DIR, n=3)
    for p in previews:
        print(f"Plot: {p}")


if __name__ == "__main__":
    main()