"""
Prepare Frey et al. 2026 (EcoSense) TLS-derived crown polygons for MillionTrees.

Terrestrial LiDAR (TLS) crown footprints segmented from a co-located drone
orthomosaic of a central-European (UTM zone 32N) closed-canopy forest plot
(EcoSense project, plot ``ortho_3919``). Inputs under ``base_dir``:

    ortho_3919.tif                 # 4-band RGBA drone orthomosaic (~1.5 cm GSD)
    crown_polygons_mamba2.gpkg     # 1601 crown polygons, EPSG:32632

Both share EPSG:32632, so no reprojection is needed. The orthomosaic is far too
large (38471 x 65536 px, 4-band) to hand to ``deepforest.preprocess.split_raster``
(which loads the whole raster through PIL), so this script tiles it directly with
rasterio windows into **2000 px crops with no overlap**, keeps only tiles that
contain at least one crown, converts each crown to tile-local image-pixel
coordinates, and writes the three MillionTrees geometry CSVs.

Every row is assigned ``existing_split="validation"``. Frey et al. 2026 is the
second reserved held-out TLS validation source (after Allen et al. 2025): it must
never appear in any train or test split. ``VALIDATION_SOURCES`` in
``package_datasets.py`` pins it to the reserved validation split.

Outputs (under ``base_dir/crops/``):

    annotations_polygons.csv
    annotations_boxes.csv      # axis-aligned bounds around each polygon
    annotations_points.csv     # polygon centroids
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely.wkt
from PIL import Image
from rasterio.windows import Window
from shapely.affinity import affine_transform
from shapely.geometry import box
from shapely.ops import transform as shapely_transform

SOURCE_NAME = "Frey et al. 2026"
PLOT_ID = "ortho_3919"
DEFAULT_BASE = Path("/orange/ewhite/DeepForest/EcoSense_TLS")
DEFAULT_RASTER = "ortho_3919.tif"
DEFAULT_CROWNS = "crown_polygons_mamba2.gpkg"

# Discard crown fragments smaller than this many square pixels after clipping to a
# tile edge — these are slivers where a crown barely grazes the tile, not usable
# training/eval targets.
MIN_PIXEL_AREA = 16.0


def _force_2d(geom):
    """Drop any Z coordinate; the source CRS is a 3D-promoted EPSG:32632."""
    if geom is None or geom.is_empty:
        return None
    if not geom.has_z:
        return geom
    return shapely_transform(lambda x, y, z=None: (x, y), geom)


def _load_crowns(crowns_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(crowns_path)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["geometry"] = gdf.geometry.apply(_force_2d)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[gdf.geometry.geom_type == "Polygon"].copy()
    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise RuntimeError(f"No valid crown polygons in {crowns_path}")
    return gdf


def _pixel_polygons(crowns_geo, window_transform, tile_w: int, tile_h: int):
    """Map geographic crown polygons to tile-local pixel coords and clip to tile.

    ``window_transform`` maps (col, row) -> (x, y); its inverse maps geographic
    coords to pixel coords with the origin at the tile's top-left, row increasing
    downward — the MillionTrees image-pixel convention.
    """
    inv = ~window_transform
    # shapely affine matrix [a, b, d, e, xoff, yoff] for (x, y) -> (col, row).
    matrix = [inv.a, inv.b, inv.d, inv.e, inv.c, inv.f]
    tile_box = box(0, 0, tile_w, tile_h)
    out = []
    for geom in crowns_geo:
        px = affine_transform(geom, matrix)
        clipped = px.intersection(tile_box)
        if clipped.is_empty:
            continue
        # A crown clipped at a tile edge may split into several pieces.
        parts = list(getattr(clipped, "geoms", [clipped]))
        for part in parts:
            if part.geom_type != "Polygon" or part.is_empty:
                continue
            if part.area < MIN_PIXEL_AREA:
                continue
            out.append(part)
    return out


def build_annotations(
    base_dir: Path,
    raster_name: str = DEFAULT_RASTER,
    crowns_name: str = DEFAULT_CROWNS,
    patch_size: int = 2000,
) -> pd.DataFrame:
    raster_path = base_dir / raster_name
    crowns_path = base_dir / crowns_name
    output_dir = base_dir / "crops"
    output_dir.mkdir(parents=True, exist_ok=True)

    crowns = _load_crowns(crowns_path)
    print(f"Loaded {len(crowns)} crown polygons from {crowns_path.name}")

    rows = []
    with rasterio.open(raster_path) as src:
        if crowns.crs is not None and src.crs is not None and crowns.crs != src.crs:
            crowns = crowns.to_crs(src.crs)
        sindex = crowns.sindex
        crown_geoms = crowns.geometry.values
        width, height = src.width, src.height
        n_tiles = 0
        for row0 in range(0, height, patch_size):
            tile_h = min(patch_size, height - row0)
            for col0 in range(0, width, patch_size):
                tile_w = min(patch_size, width - col0)
                window = Window(col0, row0, tile_w, tile_h)
                w_left, w_bottom, w_right, w_top = rasterio.windows.bounds(
                    window, src.transform)
                hits = list(sindex.intersection(
                    (w_left, w_bottom, w_right, w_top)))
                if not hits:
                    continue
                window_transform = src.window_transform(window)
                candidate = crown_geoms[hits]
                pixel_polys = _pixel_polygons(
                    candidate, window_transform, tile_w, tile_h)
                if not pixel_polys:
                    continue

                data = src.read((1, 2, 3), window=window)  # (3, H, W) RGB
                if not data.any():  # fully black / nodata window
                    continue
                tile_img = np.transpose(data, (1, 2, 0))  # (H, W, 3)

                tile_name = f"{PLOT_ID}_{col0}_{row0}.png"
                tile_path = output_dir / tile_name
                Image.fromarray(tile_img, mode="RGB").save(tile_path)

                for poly in pixel_polys:
                    rows.append({
                        "image_path": str(tile_path),
                        "geometry": poly.wkt,
                        "label": "Tree",
                        "plot_id": PLOT_ID,
                    })
                n_tiles += 1

    if not rows:
        raise RuntimeError("No annotated tiles produced.")

    annotations = pd.DataFrame(rows)
    annotations["source"] = SOURCE_NAME
    annotations["existing_split"] = "validation"
    annotations["complete"] = True

    polygons_path = output_dir / "annotations_polygons.csv"
    boxes_path = output_dir / "annotations_boxes.csv"
    points_path = output_dir / "annotations_points.csv"

    _write_geometry_csv(annotations, polygons_path, "polygon")
    _write_geometry_csv(annotations, boxes_path, "box")
    _write_geometry_csv(annotations, points_path, "point")

    print(
        f"Saved {len(annotations)} crowns across "
        f"{annotations['image_path'].nunique()} tiles")
    print(f"  polygons -> {polygons_path}")
    print(f"  boxes    -> {boxes_path}")
    print(f"  points   -> {points_path}")

    write_overlay(str(polygons_path), str(output_dir))
    return annotations


def _write_geometry_csv(df: pd.DataFrame, path: Path, geom_kind: str) -> None:
    out = df.copy()
    if geom_kind == "polygon":
        pass
    elif geom_kind == "box":
        geoms = gpd.GeoSeries(out["geometry"].apply(shapely.wkt.loads))
        bounds = geoms.bounds
        out["xmin"] = bounds["minx"]
        out["ymin"] = bounds["miny"]
        out["xmax"] = bounds["maxx"]
        out["ymax"] = bounds["maxy"]
    elif geom_kind == "point":
        geoms = gpd.GeoSeries(out["geometry"].apply(shapely.wkt.loads))
        centroids = geoms.centroid
        out["x"] = centroids.x
        out["y"] = centroids.y
        out["geometry"] = centroids.to_wkt()
    else:
        raise ValueError(f"Unknown geom_kind: {geom_kind}")
    out.to_csv(path, index=False)


def write_overlay(csv_path: str, images_dir: str, max_samples: int = 6) -> None:
    """Draw crown polygons on their tiles for visual alignment QC."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    for img_path in sorted(df["image_path"].unique())[:max_samples]:
        sub = df[df["image_path"] == img_path]
        img = np.array(Image.open(img_path).convert("RGB"))
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        for _, r in sub.iterrows():
            geom = shapely.wkt.loads(r["geometry"])
            xs, ys = geom.exterior.xy
            ax.plot(xs, ys, "c-", linewidth=1.2)
        ax.set_title(f"{os.path.basename(img_path)} — {len(sub)} crowns")
        ax.axis("off")
        out = os.path.join(
            images_dir,
            "overlay_" + os.path.splitext(os.path.basename(img_path))[0] + ".png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Overlay saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_dir", nargs="?", default=str(DEFAULT_BASE),
                        help=f"EcoSense TLS data root (default: {DEFAULT_BASE})")
    parser.add_argument("--raster", default=DEFAULT_RASTER)
    parser.add_argument("--crowns", default=DEFAULT_CROWNS)
    parser.add_argument("--patch-size", type=int, default=2000,
                        help="Tile size in pixels, no overlap (default: 2000)")
    args = parser.parse_args()
    build_annotations(Path(args.base_dir), raster_name=args.raster,
                      crowns_name=args.crowns, patch_size=args.patch_size)


if __name__ == "__main__":
    main()
