"""
Dataset prep for: Ball et al. (Paracou) — polygons
Root: /orange/ewhite/DeepForest/paracou_ball

This script:
- Finds the main orthomosaic .tif
- Reads all polygon shapefiles in the root (and optional subdirs)
- Converts geospatial polygons to image coordinates
- Splits the .tif into 500px tiles with deepforest.preprocess.split_raster
- Writes tiled images to pngs/ and annotations to pngs/annotations.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import os
import pandas as pd
import geopandas as gpd
import rasterio
from deepforest.utilities import read_file
from deepforest.preprocess import split_raster

DATASET_ROOT = Path("/orange/ewhite/DeepForest/paracou_ball").resolve()
PNG_DIR = DATASET_ROOT / "pngs"
SOURCE_NAME = "Ball et al. 2023"
PATCH_SIZE = 500


def _find_raster(root: Path) -> Path:
    """Find a single .tif in the dataset root (or common subfolders)."""
    candidates: list[Path] = []
    candidates += sorted(root.glob("*.tif"))
    candidates += sorted((root / "ortho").glob("*.tif")) if (root / "ortho").exists() else []
    if not candidates:
        raise FileNotFoundError(f"No .tif found under {root}")
    return candidates[0]


def _iter_shapefiles(root: Path) -> Iterable[Path]:
    """Yield shapefile paths under the root (1 level deep)."""
    yield from sorted(root.glob("*.gpkg"))
    for sub in root.iterdir():
        if sub.is_dir():
            yield from sorted(sub.glob("*.gpkg"))


def build_annotations() -> pd.DataFrame:
    """Create a unified annotations DataFrame in image coordinates for the raster."""
    raster_path = _find_raster(DATASET_ROOT)
    shapefiles = list(_iter_shapefiles(DATASET_ROOT))
    if not shapefiles:
        raise FileNotFoundError(f"No shapefiles found under {DATASET_ROOT}")

    frames: list[pd.DataFrame] = []
    for shp in shapefiles:
        gdf = gpd.read_file(shp)
        # minimal expectations: polygon geometry per crown
        gdf["image_path"] = str(raster_path)
        gdf["label"] = "Tree"
        df = read_file(gdf)  # converts to image coordinates using image_path
        df = df[df.is_valid]
        frames.append(df)

    annotations = pd.concat(frames, ignore_index=True)
    annotations["source"] = SOURCE_NAME
    return annotations


def split_tif(annotations: pd.DataFrame) -> pd.DataFrame:
    """Split raster into tiles and map annotations to each tile."""
    raster_path = _find_raster(DATASET_ROOT)
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(raster_path) as src:
        numpy_image = src.read()

    # read_file again to ensure expected schema and local root_dir before splitting
    gdf = read_file(annotations, root_dir=str(DATASET_ROOT))
    gdf = gpd.GeoDataFrame(gdf)
    gdf["image_path"] = gdf["image_path"].apply(os.path.basename)

    split_ann = split_raster(
        image_name=os.path.basename(raster_path),
        annotations_file=gdf,
        numpy_image=numpy_image,
        patch_size=PATCH_SIZE,
        allow_empty=False,
        save_dir=PNG_DIR,
        root_dir=DATASET_ROOT,
    )
    # Post-process
    split_ann = pd.concat([split_ann], ignore_index=True) if isinstance(split_ann, pd.DataFrame) else split_ann
    split_ann["image_path"] = split_ann["image_path"].apply(lambda x: str(PNG_DIR / x))
    split_ann["source"] = SOURCE_NAME
    out_csv = PNG_DIR / "annotations.csv"
    split_ann.to_csv(out_csv, index=False)
    return split_ann


def main() -> pd.DataFrame:
    annotations = build_annotations()
    split_annotations = split_tif(annotations)
    print(f"Wrote: {PNG_DIR / 'annotations.csv'}")
    return split_annotations


if __name__ == "__main__":
    main()


