"""
Build MillionTrees-format annotation CSVs for NEON MultiTemporal evaluation.

Crawls all .shp files under multi-temporal/ (any subfolder). Imagery:
NEON/{site}/{year}/NEONPlots/Camera/L3/{site}_{plot}_{year}.tif
Annotations: *_boxes.shp, *_points.shp, *_polygons.shp (e.g. BART_023_2018_boxes.shp
matches NEON/BART/2018/NEONPlots/Camera/L3/BART_023_2018.tif). First shp found per
(stem, geom_type) wins when the same stem appears in multiple folders.

Images are used as-is (no cropping). All rows get existing_split="test" for evaluation.
"""
import os
import glob
import pandas as pd
import geopandas as gpd
import rasterio
from deepforest.utilities import read_file

BASE_DIR = "/orange/ewhite/DeepForest/MultiTemporal"
NEON_DIR = os.path.join(BASE_DIR, "NEON")
OUTPUT_DIR = os.path.join(BASE_DIR, "annotations")
SOURCE_NAME = "NEON MultiTemporal"

# Crawl this root for all *_boxes.shp, *_points.shp, *_polygons.shp (recursive)
MULTI_TEMPORAL_ROOT = os.path.join(BASE_DIR, "multi-temporal")

GEOM_SUFFIXES = {"box": "_boxes.shp", "point": "_points.shp", "polygon": "_polygons.shp"}


def _shp_basename_to_tif_path(shp_basename: str, geom_type: str) -> str | None:
    """
    From e.g. BART_023_2018_boxes.shp get stem BART_023_2018, then
    NEON/BART/2018/NEONPlots/Camera/L3/BART_023_2018.tif
    """
    suffix = GEOM_SUFFIXES.get(geom_type)
    if not suffix or not shp_basename.endswith(suffix):
        return None
    stem = shp_basename[: -len(suffix)]
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    year = parts[-1]
    site = parts[0]
    tif_path = os.path.join(
        NEON_DIR, site, year, "NEONPlots", "Camera", "L3", stem + ".tif"
    )
    return tif_path if os.path.isfile(tif_path) else None


def build_annotations_for_geometry(geom_type: str) -> pd.DataFrame:
    """
    Crawl MULTI_TEMPORAL_ROOT for all *_boxes.shp / *_points.shp / *_polygons.shp,
    match each to NEON/.../Camera/L3/{stem}.tif. First shp per (tif_path, geom_type) wins.
    """
    if not os.path.isdir(MULTI_TEMPORAL_ROOT):
        return pd.DataFrame()

    suffix = GEOM_SUFFIXES[geom_type]
    pattern = os.path.join(MULTI_TEMPORAL_ROOT, "**", "*" + suffix)
    shp_files = sorted(glob.glob(pattern))
    seen_tif = set()
    rows = []

    for shp_path in shp_files:
        basename = os.path.basename(shp_path)
        tif_path = _shp_basename_to_tif_path(basename, geom_type)
        if tif_path is None or tif_path in seen_tif:
            continue
        seen_tif.add(tif_path)

        gdf = gpd.read_file(shp_path)
        if len(gdf) == 0:
            continue
        gdf["image_path"] = os.path.basename(tif_path)
        gdf["label"] = "Tree"

        # If the crs is EPSG:4326, load the .tif and reproject the shapefile to the .tif crs
        if gdf.crs is not None and gdf.crs.to_string() == "EPSG:4326":
            with rasterio.open(tif_path) as src:
                gdf = gdf.to_crs(src.crs)

        # Remove any None geometries
        gdf = gdf[gdf.geometry.notna()]
        gdf = read_file(gdf, root_dir=os.path.dirname(tif_path))

        gdf["image_path"] = tif_path
        gdf["source"] = SOURCE_NAME
        rows.append(gdf)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out["geometry"] = out.geometry.apply(lambda g: g.wkt if g is not None else None)
    return out


def run():
    """Build TreeBoxes, TreePoints, TreePolygons CSVs from all .shp under multi-temporal/."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(MULTI_TEMPORAL_ROOT):
        print(f"Annotation root not found: {MULTI_TEMPORAL_ROOT}")
        return

    for geom_type, suffix in GEOM_SUFFIXES.items():
        df = build_annotations_for_geometry(geom_type)
        if df.empty:
            print(f"NEON MultiTemporal {suffix.strip('_')}: no matching (shp, tif) pairs; no CSV written.")
            continue
        name = "TreeBoxes" if geom_type == "box" else "TreePoints" if geom_type == "point" else "TreePolygons"
        out_path = os.path.join(OUTPUT_DIR, f"{name}_NEON_MultiTemporal.csv")
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(df)} rows, {df['image_path'].nunique()} images)")


if __name__ == "__main__":
    run()
