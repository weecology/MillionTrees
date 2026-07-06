"""
Prepare Allen et al. 2025 TLS-derived crown polygons for MillionTrees.

Terrestrial LiDAR (TLS) crown footprints aligned to drone orthomosaics from
Joensuu (Finland) and Alto Tajo (Spain). See:
https://arxiv.org/pdf/2503.14273

Supported layouts under ``base_dir``:

1. **COCO clipped plots** (release format)::

       FIN01_clipped.tif
       FIN01_annotations.json
       SPA01_clipped.tif
       SPA01_annotations.json
       ...

   JSON files follow COCO instance-segmentation conventions with polygon
   coordinates already in image pixel space. Allen's released clips extend
   ~500 px beyond the TLS-annotated crowns on every side, so each clip is
   cropped to the buffered bounding box of its annotations (``--envelope-buffer``,
   default 50 px) before being written to ``crops/``. This drops the ring of
   unannotated trees that would otherwise score as false positives at eval.

2. **Vector + orthomosaic** (optional fallback)::

       orthomosaics/FIN01.tif
       crowns/FIN01.shp

   Large orthomosaics are tiled with ``split_raster`` when wider or taller than
   ``patch_size``.

All rows are assigned ``existing_split="validation"`` — a held-out TLS
ground-truth set for post-hoc model evaluation, not hyperparameter tuning.

Outputs (under ``base_dir/crops/``):

    annotations_polygons.csv
    annotations_boxes.csv      # axis-aligned bounds around each polygon
    annotations_points.csv     # polygon centroids
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
import shapely.wkt
from deepforest.preprocess import split_raster
from deepforest.utilities import read_file
from rasterio.windows import Window
from shapely.affinity import translate
from shapely.geometry import Polygon

SOURCE_NAME = "Allen et al. 2025"
DEFAULT_BASE = Path("/orange/ewhite/DeepForest/Allen2025")
PLOT_RE = re.compile(r"(FIN\d+|SPA\d+)", re.IGNORECASE)


def _plot_id(path: Path) -> str | None:
    match = PLOT_RE.search(path.stem)
    return match.group(1).upper() if match else None


def _segmentation_to_polygon(flat_coords: list[float]) -> Polygon | None:
    pairs = list(zip(flat_coords[0::2], flat_coords[1::2]))
    if len(pairs) < 3:
        return None
    poly = Polygon(pairs)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.geom_type != "Polygon":
        return None
    return poly


def _discover_coco_clipped_pairs(base_dir: Path) -> dict[str, tuple[Path, Path]]:
    pairs: dict[str, tuple[Path, Path]] = {}
    for json_path in sorted(base_dir.glob("*_annotations.json")):
        plot = _plot_id(json_path)
        if plot is None:
            continue
        tif_path = base_dir / f"{plot}_clipped.tif"
        if tif_path.exists():
            pairs[plot] = (tif_path, json_path)
    return pairs


def _discover_orthomosaics(base_dir: Path) -> dict[str, Path]:
    candidates = []
    for pattern in ("orthomosaics/*.tif", "orthomosaics/*.tiff", "images/*.tif",
                      "*_ortho.tif", "*_orthomosaic.tif"):
        candidates.extend(base_dir.glob(pattern))
    orthos: dict[str, Path] = {}
    for path in sorted(set(candidates)):
        if path.parent.name in ("crops", "crowns", "labels"):
            continue
        plot = _plot_id(path)
        if plot:
            orthos[plot] = path
    return orthos


def _discover_crowns(base_dir: Path) -> dict[str, Path]:
    candidates = []
    for pattern in ("crowns/*.shp", "crowns/*.gpkg", "labels/*.shp",
                    "labels/*.gpkg", "*_crowns.shp", "*_crowns.gpkg"):
        candidates.extend(base_dir.glob(pattern))
    crowns: dict[str, Path] = {}
    for path in sorted(set(candidates)):
        plot = _plot_id(path)
        if plot:
            crowns[plot] = path
    return crowns


def _to_2d_polygon(geom) -> Polygon | None:
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type != "Polygon":
        return None
    if geom.has_z:
        coords = [(x, y) for x, y, *_ in geom.exterior.coords]
        return Polygon(coords)
    return geom


def _crop_image_to_envelope(
    plot_id: str,
    src_tif: Path,
    polys: list[Polygon],
    output_dir: Path,
    buffer_px: int,
) -> list[dict]:
    """Crop ``src_tif`` to the buffered envelope of ``polys`` and shift coords.

    Allen's released clips extend well beyond the TLS-annotated crowns on every
    side, leaving a ring of visible-but-unannotated trees. Cropping to the
    annotation bounding box (plus ``buffer_px``) removes that margin so those
    trees no longer count as false positives at eval. Polygon coordinates are
    translated into the cropped raster's pixel frame.
    """
    minx = min(p.bounds[0] for p in polys)
    miny = min(p.bounds[1] for p in polys)
    maxx = max(p.bounds[2] for p in polys)
    maxy = max(p.bounds[3] for p in polys)

    with rasterio.open(src_tif) as src:
        width, height = src.width, src.height
        x0 = max(0, int(math.floor(minx - buffer_px)))
        y0 = max(0, int(math.floor(miny - buffer_px)))
        x1 = min(width, int(math.ceil(maxx + buffer_px)))
        y1 = min(height, int(math.ceil(maxy + buffer_px)))
        window = Window(x0, y0, x1 - x0, y1 - y0)
        data = src.read(window=window)
        profile = src.profile.copy()
        profile.update(
            width=x1 - x0,
            height=y1 - y0,
            transform=src.window_transform(window),
        )

    dest_tif = output_dir / src_tif.name
    with rasterio.open(dest_tif, "w", **profile) as dst:
        dst.write(data)

    print(
        f"  {plot_id}: cropped {width}x{height} -> {x1 - x0}x{y1 - y0} "
        f"(envelope + {buffer_px}px), {len(polys)} crowns")

    rows = []
    for poly in polys:
        shifted = translate(poly, xoff=-x0, yoff=-y0)
        rows.append({
            "image_path": str(dest_tif),
            "geometry": shifted.wkt,
            "label": "tree",
            "plot_id": plot_id,
        })
    return rows


def _coco_plot_to_rows(
    plot_id: str,
    tif_path: Path,
    json_path: Path,
    output_dir: Path,
    buffer_px: int = 50,
) -> list[dict]:
    with open(json_path) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    # Group polygons by their source image so each clip is cropped to its own
    # annotation envelope (released clips carry a single image, but grouping
    # keeps the multi-image case correct).
    polys_by_image: dict[int, list[Polygon]] = {}
    for ann in coco["annotations"]:
        seg = ann.get("segmentation")
        if not seg or not seg[0]:
            continue
        poly = _segmentation_to_polygon(seg[0])
        if poly is None:
            continue
        polys_by_image.setdefault(ann["image_id"], []).append(poly)

    rows = []
    for image_id, polys in polys_by_image.items():
        file_name = images.get(image_id, {}).get("file_name", tif_path.name)
        src_tif = tif_path if file_name == tif_path.name else tif_path.parent / file_name
        if not src_tif.exists():
            print(f"  WARNING: source raster {src_tif} missing; skipping")
            continue
        rows.extend(
            _crop_image_to_envelope(plot_id, src_tif, polys, output_dir, buffer_px))
    return rows


def _vector_plot_to_rows(
    plot_id: str,
    ortho_path: Path,
    crown_path: Path,
    output_dir: Path,
    patch_size: int,
) -> pd.DataFrame:
    gdf = gpd.read_file(crown_path)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["geometry"] = gdf.geometry.apply(_to_2d_polygon)
    gdf = gdf[gdf.geometry.notna()].copy()
    if gdf.empty:
        return pd.DataFrame()

    gdf["image_path"] = ortho_path.name
    gdf["label"] = "tree"
    annotation = read_file(gdf, root_dir=ortho_path.parent)

    with rasterio.open(ortho_path) as src:
        needs_tiling = src.width > patch_size or src.height > patch_size

    if needs_tiling:
        tiles = split_raster(
            annotation,
            path_to_raster=str(ortho_path),
            patch_size=patch_size,
            allow_empty=False,
            base_dir=str(output_dir),
        )
        if tiles is None or len(tiles) == 0:
            return pd.DataFrame()
        tiles["plot_id"] = plot_id
        return tiles

    dest_tif = output_dir / ortho_path.name
    if not dest_tif.exists():
        shutil.copy(ortho_path, dest_tif)
    annotation["image_path"] = str(dest_tif)
    annotation["plot_id"] = plot_id
    return annotation


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


def build_annotations(
    base_dir: Path,
    patch_size: int = 4000,
    envelope_buffer: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = base_dir / "crops"
    output_dir.mkdir(parents=True, exist_ok=True)

    parts: list[pd.DataFrame | list[dict]] = []
    coco_pairs = _discover_coco_clipped_pairs(base_dir)
    if coco_pairs:
        print(f"Found {len(coco_pairs)} COCO clipped plot pairs")
        for plot_id, (tif_path, json_path) in sorted(coco_pairs.items()):
            print(f"Processing {plot_id}: {tif_path.name} + {json_path.name}")
            rows = _coco_plot_to_rows(
                plot_id, tif_path, json_path, output_dir, envelope_buffer)
            if rows:
                parts.append(pd.DataFrame(rows))
    else:
        orthos = _discover_orthomosaics(base_dir)
        crowns = _discover_crowns(base_dir)
        if not orthos:
            raise FileNotFoundError(
                f"No data found under {base_dir}. Expected "
                "{{PLOT}}_clipped.tif + {{PLOT}}_annotations.json or "
                "orthomosaics/ + crowns/ shapefiles.")
        if not crowns:
            raise FileNotFoundError(f"No crown vectors found under {base_dir}.")

        paired = sorted(set(orthos) & set(crowns))
        if not paired:
            raise FileNotFoundError(
                f"No matching plot IDs between orthomosaics ({len(orthos)}) "
                f"and crowns ({len(crowns)}) under {base_dir}")

        for plot_id in paired:
            print(f"Processing {plot_id}: {orthos[plot_id].name} + {crowns[plot_id].name}")
            tiles = _vector_plot_to_rows(
                plot_id,
                orthos[plot_id],
                crowns[plot_id],
                output_dir,
                patch_size,
            )
            if len(tiles) > 0:
                parts.append(tiles)

    if not parts:
        raise RuntimeError("No annotations produced.")

    annotations = pd.concat(parts, ignore_index=True)
    annotations["image_path"] = annotations["image_path"].apply(
        lambda x: str(output_dir / Path(x).name))
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
        f"Saved {len(annotations)} annotations across "
        f"{annotations['image_path'].nunique()} images")
    print(f"  polygons -> {polygons_path}")
    print(f"  boxes    -> {boxes_path}")
    print(f"  points   -> {points_path}")

    return (
        pd.read_csv(polygons_path),
        pd.read_csv(boxes_path),
        pd.read_csv(points_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "base_dir",
        nargs="?",
        default=str(DEFAULT_BASE),
        help=f"Allen2025 data root (default: {DEFAULT_BASE})",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=4000,
        help="Tile size for large vector+ortho inputs only (default: 4000)",
    )
    parser.add_argument(
        "--envelope-buffer",
        type=int,
        default=50,
        help="Pixels to buffer the annotation bounding box before cropping each "
        "COCO clip (default: 50)",
    )
    args = parser.parse_args()
    build_annotations(
        Path(args.base_dir),
        patch_size=args.patch_size,
        envelope_buffer=args.envelope_buffer,
    )


if __name__ == "__main__":
    main()
