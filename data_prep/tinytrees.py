"""Prepare the PlanetScope (3 m) + Gaofen-2 (0.8 m) tinytrees point datasets.

Hand-drawn point labels made through photointerpretation of satellite imagery (with help from
higher-resolution products overlaid, e.g. Google Earth). Two sensors are provided, each with:

- one or more georeferenced ``.tif`` rasters,
- a vector layer of point labels (``.shp``, ``.gpkg``, or ``.geojson``),
- a vector layer of *labeling rectangles* delimiting the areas where annotation is exhaustive.
  Trees outside those rectangles may be unlabeled, so we crop each raster to the rectangles
  before writing per-tile pixel annotations.

The HPC layout this script expects (under ``$TINYTREES_ROOT``, default
``/orange/ewhite/DeepForest/tinytrees``)::

    tinytrees/
      planetscope/
        imagery/*.tif
        points.{shp,gpkg,geojson}
        rectangles.{shp,gpkg,geojson}
      gaofen2/
        imagery/*.tif
        points.{shp,gpkg,geojson}
        rectangles.{shp,gpkg,geojson}

Sensor sub-folders, the rasters dir, and the vector basenames are all auto-discovered, so
minor naming differences (e.g. ``planet``/``PlanetScope`` or ``labels.shp``/``points.shp``)
are tolerated. The script emits one ``annotations.csv`` per sensor and a combined
``annotations.csv`` at the dataset root (registered in ``data_prep/annotation_csvs.cfg``).

This is a TreePoints source for the random task; it is intentionally **not** listed as a
zero-shot test source.
"""

from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import Point, Polygon, box

ROOT_DIR = Path(
    os.environ.get("TINYTREES_ROOT", "/orange/ewhite/DeepForest/tinytrees")).resolve()

SOURCE_NAME = "Gominski et al. 2025"
SENSOR_SUFFIX = {
    "planetscope": "PlanetScope",
    "gaofen2": "Gaofen-2",
}
SENSOR_ALIASES = {
    "planetscope": ["planetscope", "planet_scope", "planet", "ps", "ps3m"],
    "gaofen2": ["gaofen2", "gaofen_2", "gaofen-2", "gaofen", "gf2", "gf-2", "gf_2"],
}

VECTOR_EXTS = (".shp", ".gpkg", ".geojson", ".json")
POINT_BASENAMES = ("points", "point_labels", "labels", "trees", "tree_points")
RECT_BASENAMES = (
    "rectangles",
    "rectangle",
    "labeling_rectangles",
    "label_rectangles",
    "aois",
    "aoi",
    "tiles",
    "extents",
)
RASTER_DIR_NAMES = ("imagery", "images", "rasters", "tifs", "tif", "")


def _find_sensor_dir(root: Path, sensor: str) -> Path:
    """Return the sub-directory matching one of the known aliases for ``sensor``."""
    aliases = SENSOR_ALIASES[sensor]
    candidates = [p for p in root.iterdir() if p.is_dir()]
    for p in candidates:
        if p.name.lower() in aliases:
            return p
    for p in candidates:
        low = p.name.lower()
        if any(alias in low for alias in aliases):
            return p
    raise FileNotFoundError(
        f"No sub-directory for sensor {sensor!r} under {root} "
        f"(tried {aliases})")


def _find_vector(sensor_dir: Path, basenames: tuple[str, ...], label: str) -> Path:
    matches: list[Path] = []
    for base in basenames:
        for ext in VECTOR_EXTS:
            matches.extend(sensor_dir.rglob(f"{base}{ext}"))
            matches.extend(sensor_dir.rglob(f"{base}.*{ext}"))
    if not matches:
        for ext in VECTOR_EXTS:
            for path in sensor_dir.rglob(f"*{ext}"):
                low = path.stem.lower()
                if any(bn in low for bn in basenames):
                    matches.append(path)
    matches = [p for p in matches if not any(seg.startswith(".") for seg in p.parts)]
    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError(
            f"No {label} vector found in {sensor_dir} "
            f"(looked for {basenames} with extensions {VECTOR_EXTS})")
    if len(matches) > 1:
        print(f"  Note: multiple {label} candidates in {sensor_dir}, using {matches[0]}")
    return matches[0]


def _find_rasters(sensor_dir: Path) -> list[Path]:
    rasters: list[Path] = []
    for sub in RASTER_DIR_NAMES:
        d = sensor_dir / sub if sub else sensor_dir
        if not d.exists():
            continue
        rasters.extend(d.glob("*.tif"))
        rasters.extend(d.glob("*.tiff"))
    rasters = sorted({r.resolve() for r in rasters if r.is_file()})
    if not rasters:
        raise FileNotFoundError(f"No .tif rasters found under {sensor_dir}")
    return [Path(r) for r in rasters]


def _ensure_rect_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Normalize the rectangle layer to a GeoDataFrame of axis-aligned polygons.

    Points/lines are dropped; multi-polygons are exploded into their parts.
    """
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[gdf.geometry.geom_type.isin({"Polygon", "MultiPolygon"})]
    if gdf.empty:
        raise RuntimeError("Rectangle layer contains no polygon geometries")
    gdf = gdf.explode(index_parts=False, ignore_index=True)
    return gdf


def _rect_id(row_index: int, geom) -> str:
    return f"rect{row_index:04d}"


def _crop_rectangle(
    raster_path: Path,
    rect_geom: Polygon,
    out_dir: Path,
    rect_name: str,
) -> tuple[Path, rasterio.Affine, tuple[float, float, float, float]] | None:
    """Crop ``raster_path`` to the bounds of ``rect_geom`` and write a tile.

    Returns the cropped tile path, its affine transform, and (left, bottom, right, top) bounds
    in raster CRS. Returns None when the rectangle does not intersect the raster's footprint.
    """
    with rasterio.open(raster_path) as src:
        rl, rb, rr, rt = src.bounds
        gl, gb, gr, gt = rect_geom.bounds
        left, bottom = max(rl, gl), max(rb, gb)
        right, top = min(rr, gr), min(rt, gt)
        if right <= left or top <= bottom:
            return None

        window = from_bounds(left, bottom, right, top, transform=src.transform)
        window = window.round_offsets().round_lengths()
        if window.width <= 0 or window.height <= 0:
            return None

        data = src.read(window=window)
        if data.size == 0:
            return None

        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(
            height=int(window.height),
            width=int(window.width),
            transform=transform,
            driver="GTiff",
            count=data.shape[0],
            compress="lzw",
        )

    out_path = out_dir / f"{raster_path.stem}__{rect_name}.tif"
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)

    crop_left = transform.c
    crop_top = transform.f
    crop_right = crop_left + transform.a * data.shape[2]
    crop_bottom = crop_top + transform.e * data.shape[1]
    return out_path, transform, (crop_left, crop_bottom, crop_right, crop_top)


def _points_in_rect(points: gpd.GeoDataFrame, rect_geom: Polygon) -> gpd.GeoDataFrame:
    sindex = points.sindex
    candidates = list(sindex.intersection(rect_geom.bounds))
    if not candidates:
        return points.iloc[0:0]
    subset = points.iloc[candidates]
    return subset[subset.geometry.within(rect_geom)]


def _project_points_to_pixel(
    pts: gpd.GeoDataFrame,
    transform: rasterio.Affine,
) -> list[tuple[float, float]]:
    inv = ~transform
    rows = []
    for geom in pts.geometry:
        if not isinstance(geom, Point):
            continue
        col, row = inv * (geom.x, geom.y)
        rows.append((float(col), float(row)))
    return rows


def _process_sensor(sensor: str, sensor_dir: Path, out_root: Path) -> pd.DataFrame:
    """Build a points annotation table for one sensor (planetscope or gaofen2)."""
    crops_dir = out_root / sensor / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    rasters = _find_rasters(sensor_dir)
    points_path = _find_vector(sensor_dir, POINT_BASENAMES, "point labels")
    rects_path = _find_vector(sensor_dir, RECT_BASENAMES, "labeling rectangles")

    print(f"[{sensor}] {len(rasters)} raster(s); points={points_path.name}; "
          f"rectangles={rects_path.name}")

    points = gpd.read_file(points_path)
    points = points[points.geometry.notna()].copy()
    points = points[points.geometry.geom_type == "Point"]
    if points.empty:
        raise RuntimeError(f"[{sensor}] No point geometries in {points_path}")
    rects = _ensure_rect_polygons(gpd.read_file(rects_path))

    rows: list[dict] = []
    source_name = f"{SOURCE_NAME} {SENSOR_SUFFIX[sensor]}"

    for raster_path in rasters:
        with rasterio.open(raster_path) as src:
            if src.crs is None:
                raise RuntimeError(f"[{sensor}] Raster {raster_path} has no CRS")
            raster_crs = src.crs
            raster_bbox = box(*src.bounds)

        rects_r = rects.to_crs(raster_crs) if rects.crs != raster_crs else rects
        points_r = points.to_crs(raster_crs) if points.crs != raster_crs else points
        rects_r = rects_r[rects_r.geometry.intersects(raster_bbox)]
        if rects_r.empty:
            continue

        for idx, rect_row in rects_r.iterrows():
            rect_geom = rect_row.geometry
            if rect_geom is None or rect_geom.is_empty:
                continue
            rect_name = _rect_id(int(idx), rect_geom)
            cropped = _crop_rectangle(raster_path, rect_geom, crops_dir, rect_name)
            if cropped is None:
                continue
            crop_path, transform, _ = cropped

            rect_points = _points_in_rect(points_r, rect_geom)
            for col, row in _project_points_to_pixel(rect_points, transform):
                rows.append({
                    "image_path": str(crop_path.resolve()),
                    "geometry": f"POINT ({col} {row})",
                    "source": source_name,
                    "label": "Tree",
                })

    if not rows:
        raise RuntimeError(f"[{sensor}] No annotations produced -- check the rectangles "
                           f"intersect the rasters and the points layer.")

    df = pd.DataFrame(rows)
    out_csv = out_root / sensor / "annotations.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[{sensor}] wrote {len(df)} points across {df['image_path'].nunique()} crops "
          f"-> {out_csv}")
    return df


def build(root: Path = ROOT_DIR) -> Path:
    """Build per-sensor and combined ``annotations.csv`` under ``root``."""
    parts: list[pd.DataFrame] = []
    for sensor in ("planetscope", "gaofen2"):
        sensor_dir = _find_sensor_dir(root, sensor)
        parts.append(_process_sensor(sensor, sensor_dir, root))

    combined = pd.concat(parts, ignore_index=True)
    combined_csv = root / "annotations.csv"
    combined.to_csv(combined_csv, index=False)
    print(f"Combined annotations: {len(combined)} points -> {combined_csv}")
    return combined_csv


if __name__ == "__main__":
    build()
