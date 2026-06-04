"""
Dataset prep for: Ball et al. 2023 (DetectTree2) — polygons
Sites: Sepilok East, Sepilok West (Borneo, Malaysia), Danum Valley (Borneo, Malaysia)
Root: /orange/ewhite/DeepForest/DetectTree2

Both rasters are 4-band float32 (values 0-65535). This script:
  1. Converts each raster to 3-band uint8 via windowed I/O (cached to disk)
  2. Projects polygon annotations from .gpkg files to pixel coordinates
  3. Tiles each corrected raster into 500 px patches
  4. Writes pngs/annotations.csv and sample visualisations to pngs/viz/
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from deepforest.preprocess import split_raster
from deepforest.utilities import read_file

DATASET_ROOT = Path("/orange/ewhite/DeepForest/DetectTree2").resolve()
PNG_DIR = DATASET_ROOT / "pngs"
VIZ_DIR = PNG_DIR / "viz"
SOURCE_NAME = "Ball et al. 2023"
PATCH_SIZE = 1000

SITES: dict[str, dict] = {
    "Sepilok": {
        "raster": DATASET_ROOT / "Sep_MA14_21_orthomosaic_20141023_reprojected_full_res.tif",
        "gpkgs": [DATASET_ROOT / "SepilokEast.gpkg", DATASET_ROOT / "SepilokWest.gpkg"],
        "corrected": DATASET_ROOT / "Sep_corrected_rgb.tif",
    },
    "Danum": {
        "raster": DATASET_ROOT / "Dan_2014_RGB_project_to_CHM.tif",
        "gpkgs": [DATASET_ROOT / "Danum.gpkg"],
        "corrected": DATASET_ROOT / "Dan_corrected_rgb.tif",
    },
}


def convert_to_rgb_uint8(src_path: Path, dst_path: Path) -> None:
    """Convert 4-band float32 raster (0-65535) to 3-band uint8 via windowed I/O."""
    if dst_path.exists():
        print(f"  Using cached corrected raster: {dst_path.name}")
        return
    print(f"  Converting {src_path.name} -> {dst_path.name} ...")
    with rasterio.open(src_path) as src:
        meta = src.meta.copy()
        meta.update(count=3, dtype=rasterio.uint8, nodata=0)
        with rasterio.open(dst_path, "w", **meta) as dst:
            for _, window in src.block_windows(1):
                data = src.read(window=window)[:3]
                data = np.where(np.isnan(data), 0.0, data)
                data = np.clip(data / 65535.0 * 255, 0, 255).astype(np.uint8)
                dst.write(data, window=window)
    print("  Done.")


def build_annotations(site_name: str, raster_path: Path, gpkg_paths: list[Path]) -> pd.DataFrame:
    """Read .gpkg files and project polygons to pixel coordinates using the original raster."""
    frames = []
    for gpkg in gpkg_paths:
        gdf = gpd.read_file(gpkg)
        gdf["image_path"] = str(raster_path)
        gdf["label"] = "Tree"
        df = read_file(gdf, root_dir=str(raster_path.parent))
        df = df[df.geometry.is_valid]
        df["site"] = site_name
        frames.append(df)
    ann = pd.concat(frames, ignore_index=True)
    ann["source"] = SOURCE_NAME
    return ann


def tile_site(annotations: pd.DataFrame, corrected_path: Path) -> pd.DataFrame:
    """Tile a site's corrected raster and return per-tile annotations."""
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    # Normalise schema and redirect image_path to the corrected raster filename
    gdf = read_file(annotations, root_dir=str(DATASET_ROOT))
    gdf = gpd.GeoDataFrame(gdf)
    gdf["image_path"] = os.path.basename(corrected_path)

    with rasterio.open(corrected_path) as src:
        numpy_image = src.read()

    tiled = split_raster(
        image_name=os.path.basename(corrected_path),
        annotations_file=gdf,
        numpy_image=numpy_image,
        patch_size=PATCH_SIZE,
        allow_empty=False,
        save_dir=str(PNG_DIR),
        root_dir=str(DATASET_ROOT),
    )
    tiled["image_path"] = tiled["image_path"].apply(lambda x: str(PNG_DIR / os.path.basename(x)))
    tiled["source"] = SOURCE_NAME
    return tiled


def visualize(annotations: pd.DataFrame, n_samples: int = 3) -> None:
    """Save n_samples annotated tile panels per site to VIZ_DIR."""
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    for site, group in annotations.groupby("site"):
        tiles = group["image_path"].unique().tolist()
        chosen = random.sample(tiles, min(n_samples, len(tiles)))
        fig, axes = plt.subplots(1, len(chosen), figsize=(6 * len(chosen), 6))
        if len(chosen) == 1:
            axes = [axes]
        for ax, path in zip(axes, chosen):
            img = plt.imread(path)
            ax.imshow(img)
            for _, row in group[group["image_path"] == path].iterrows():
                if row.geometry is not None and not row.geometry.is_empty:
                    xs, ys = row.geometry.exterior.xy
                    ax.plot(xs, ys, color="red", linewidth=0.8)
            ax.set_title(os.path.basename(path), fontsize=7)
            ax.axis("off")
        fig.suptitle(f"{site} — {SOURCE_NAME}")
        out = VIZ_DIR / f"{site}_sample.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


def main() -> pd.DataFrame:
    all_tiled: list[pd.DataFrame] = []

    for site_name, cfg in SITES.items():
        print(f"\n=== {site_name} ===")
        convert_to_rgb_uint8(cfg["raster"], cfg["corrected"])
        ann = build_annotations(site_name, cfg["raster"], cfg["gpkgs"])
        print(f"  Annotations loaded: {len(ann)}")
        tiled = tile_site(ann, cfg["corrected"])
        tiled["site"] = site_name
        print(f"  Tiles with annotations: {tiled['image_path'].nunique()}")
        all_tiled.append(tiled)

    combined = pd.concat(all_tiled, ignore_index=True)
    out_csv = PNG_DIR / "annotations.csv"
    combined.to_csv(out_csv, index=False)
    print(f"\nWrote: {out_csv}  ({len(combined)} rows)")

    print("\nGenerating visualisations ...")
    visualize(combined)

    return combined


if __name__ == "__main__":
    main()
