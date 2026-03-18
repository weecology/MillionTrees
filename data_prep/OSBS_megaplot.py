"""Prepare OSBS megaplot 2025 point dataset for MillionTrees.

Inputs (existing on /orange):
- Shapefile of visible trees (points):
  /orange/ewhite/DeepForest/OSBS_megaplot/VisibleTrees_OSBS_2025/VisibleTrees_2025.shp
- RGB mosaic:
  /orange/ewhite/DeepForest/OSBS_megaplot/2025/mosaic_2025.tif

This script:
- Reads the point shapefile
- Converts geospatial coordinates to pixel coordinates in the mosaic
- Writes a flat annotations.csv with POINT geometries in pixel space
- Splits the large mosaic into smaller tiles with deepforest.preprocess.split_raster
- Writes tiled annotations at:
  /orange/ewhite/DeepForest/OSBS_megaplot/2025/pngs/annotations.csv
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import Point

from deepforest.preprocess import split_raster
from deepforest.utilities import read_file


ROOT_DIR = Path(
    os.environ.get(
        "OSBS_MEGAPLOT_ROOT",
        "/orange/ewhite/DeepForest/OSBS_megaplot/2025",
    )
).resolve()
SHAPEFILE = Path(
    os.environ.get(
        "OSBS_MEGAPLOT_SHAPEFILE",
        "/orange/ewhite/DeepForest/OSBS_megaplot/VisibleTrees_OSBS_2025/VisibleTrees_2025.shp",
    )
).resolve()
MOSAIC = Path(
    os.environ.get(
        "OSBS_MEGAPLOT_MOSAIC",
        "/orange/ewhite/DeepForest/OSBS_megaplot/2025/mosaic_2025.tif",
    )
).resolve()
OUT_CSV = ROOT_DIR / "annotations_pixel.csv"
PNG_DIR = ROOT_DIR / "pngs"
TILED_CSV = PNG_DIR / "annotations.csv"
SOURCE_NAME = "OSBS megaplot 2025"


def _build_pixel_annotations() -> str:
    """Project point shapefile into mosaic pixel coordinates and write OUT_CSV."""
    if not SHAPEFILE.exists():
        raise FileNotFoundError(f"Shapefile not found: {SHAPEFILE}")
    if not MOSAIC.exists():
        raise FileNotFoundError(f"Mosaic not found: {MOSAIC}")

    gdf = gpd.read_file(SHAPEFILE)
    if gdf.empty:
        raise RuntimeError(f"No features found in shapefile: {SHAPEFILE}")

    with rasterio.open(MOSAIC) as ds:
        mosaic_crs = ds.crs
        if mosaic_crs is None:
            raise RuntimeError(f"Mosaic has no CRS defined: {MOSAIC}")

        if gdf.crs != mosaic_crs:
            gdf = gdf.to_crs(mosaic_crs)

        rows: List[Dict[str, object]] = []
        for geom in gdf.geometry:
            if not isinstance(geom, Point):
                # Skip non-point geometries
                continue
            x, y = float(geom.x), float(geom.y)
            row, col = ds.index(x, y)
            geom_wkt = f"POINT ({float(col)} {float(row)})"
            rows.append(
                {
                    "image_path": str(MOSAIC),
                    "geometry": geom_wkt,
                    "source": SOURCE_NAME,
                }
            )

    if not rows:
        raise RuntimeError("No point geometries converted to pixel space.")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    return str(OUT_CSV)


def _split_mosaic(out_csv: str) -> str:
    """Split large mosaic into tiles and write tiled annotations CSV."""
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(out_csv)
    if df.empty:
        raise RuntimeError(f"No annotations in {out_csv}")

    images = df["image_path"].unique()
    split_annotations = []

    for image in images:
        annotations = df[df["image_path"] == image].copy(deep=True)

        with rasterio.open(image) as src:
            numpy_image = src.read()

        gdf = read_file(annotations, root_dir=ROOT_DIR)
        gdf = gpd.GeoDataFrame(gdf)
        gdf["label"] = "Tree"
        # Use basenames for split_raster, as it writes relative filenames
        gdf["image_path"] = gdf["image_path"].apply(os.path.basename)

        split_image_annotations = split_raster(
            image_name=os.path.basename(image),
            annotations_file=gdf,
            numpy_image=numpy_image,
            patch_size=500,
            allow_empty=False,
            save_dir=PNG_DIR,
            root_dir=ROOT_DIR,
        )
        split_annotations.append(split_image_annotations)

    split_df = pd.concat(split_annotations)
    split_df["image_path"] = split_df["image_path"].apply(
        lambda x: str(PNG_DIR / x)
    )
    split_df["source"] = SOURCE_NAME
    split_df.to_csv(TILED_CSV, index=False)
    return str(TILED_CSV)


def main() -> str:
    pixel_csv = _build_pixel_annotations()
    tiled_csv = _split_mosaic(pixel_csv)
    print(f"Wrote pixel annotations: {pixel_csv}")
    print(f"Wrote tiled annotations: {tiled_csv}")
    return tiled_csv


if __name__ == "__main__":
    main()

