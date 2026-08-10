"""
Process Zenodo 15591546 (Krkonoše Bílé Labe Valley) for MillionTrees TreeBoxes.

Dataset: UAV, aerial, and terrestrial 3D point clouds from the Krkonoše Mts. (Czechia)
treeline ecotone with tree-level reference ground-based measurements.
https://zenodo.org/records/15591546

Data and outputs live under /orange/ewhite/DeepForest/Zenodo_15591546/ by default.
To move an existing download: mv zenodo_15591546 /orange/ewhite/DeepForest/Zenodo_15591546
Then extract Bile_Labe_Valley.zip there so it contains LOW_*, MID_*, HIGH_*, README.txt.

Run from MillionTrees repo root (no data_prep in path):
  uv run python Krkonose_BileLabe.py /orange/ewhite/DeepForest/Zenodo_15591546 --visualize
Or from the Zenodo data dir (extract_dir defaults to the orange path above):
  uv run python /path/to/MillionTrees/Krkonose_BileLabe.py --visualize

Annotations are matched to airborne data by plot: each of the three plots (LOW/MID/HIGH)
has one orthoimage (plot_level/BL_*_orthoimagery.tif) and one bounding-box shapefile
(tree_level/BL_*_bb.shp) in the same CRS (EPSG:5514). Boxes are LiDAR-derived crown
bounding boxes; we convert polygon bounds to raster pixel coordinates via the ortho's
geotransform.
"""

import argparse
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


PLOTS = [
    ("LOW", "LOW_upper_forest_limit", "BL_low"),
    ("MID", "MID_intermediate_part", "BL_mid"),
    ("HIGH", "HIGH_upper_tree_limit", "BL_high"),
]
SOURCE_LABEL = "Šrollerů et al. 2025"
DEFAULT_BASE = Path("/orange/ewhite/DeepForest/Zenodo_15591546")


def _ensure_bb_extracted(extract_dir: Path, plot_subdir: str, prefix: str) -> Path:
    """Unzip BL_*_bb.zip into tree_level if needed; return path to .shp."""
    tree_level = extract_dir / plot_subdir / "tree_level"
    bb_zip = tree_level / f"{prefix}_bb.zip"
    shp = tree_level / f"{prefix}_bb.shp"
    if not shp.exists() and bb_zip.exists():
        with zipfile.ZipFile(bb_zip) as z:
            z.extractall(tree_level)
    return shp


def _valid_pixel_mask(data: np.ndarray, nodata) -> np.ndarray:
    """(H, W) True where the ortho holds real imagery rather than fill."""
    if nodata is None:
        return np.ones(data.shape[1:], dtype=bool)
    if data.shape[0] >= 4:
        return data[3] != nodata  # 4th band is the alpha/fill mask
    return ~(data[:3] == nodata).all(axis=0)


def write_rgb_uint8(src_path: Path, dst_path: Path) -> None:
    """Write a 3-band uint8 copy of an ortho, preserving the pixel grid.

    These orthos are 4-band uint16 with 65535 fill, but the real radiometry only
    occupies 0-255. That layout crashes the packaging tree-coverage mask step
    (PIL: "Cannot handle this data type: (1, 1, 3), <f4"), which in turn drops
    the source from the release, and cv2 reads it inconsistently at train time.
    Width/height/transform are unchanged, so the annotations' pixel coordinates
    stay valid.
    """
    if dst_path.exists():
        with rasterio.open(dst_path) as dst:
            if dst.count == 3 and dst.dtypes[0] == "uint8":
                return

    with rasterio.open(src_path) as src:
        data = src.read()
        profile = src.profile
        nodata = src.nodata

    valid = _valid_pixel_mask(data, nodata)
    rgb = data[:3]
    if rgb[:, valid].max() <= 255:
        # Native 8-bit values stored in a 16-bit container: cast, don't stretch.
        out = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        out = np.transpose(_raster_to_uint8_rgb(data, nodata=nodata), (2, 0, 1))
    out[:, ~valid] = 0

    profile.update(count=3, dtype="uint8", nodata=None, compress="deflate")
    profile.pop("photometric", None)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(out)


def process_plot(
    extract_dir: Path,
    images_dir: Path,
    plot_subdir: str,
    prefix: str,
) -> tuple[pd.DataFrame, Path | None]:
    """
    Load one plot's ortho and bb shapefile; convert boxes to pixel coords; return
    annotations DataFrame and path to the ortho in images_dir (for writing CSV).
    """
    ortho_in = extract_dir / plot_subdir / "plot_level" / f"{prefix}_orthoimagery.tif"
    if not ortho_in.exists():
        return pd.DataFrame(), None

    shp_path = _ensure_bb_extracted(extract_dir, plot_subdir, prefix)
    if not shp_path.exists():
        return pd.DataFrame(), None

    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        return pd.DataFrame(), None

    with rasterio.open(ortho_in) as src:
        transform = src.transform
        w, h = src.width, src.height

    rows = []
    for _, row in gdf.iterrows():
        minx, miny, maxx, maxy = row.geometry.bounds
        c1, r1 = ~transform * (minx, miny)
        c2, r2 = ~transform * (maxx, maxy)
        xmin = max(0, min(c1, c2))
        xmax = min(w, max(c1, c2))
        ymin = max(0, min(r1, r2))
        ymax = min(h, max(r1, r2))
        if xmax <= xmin or ymax <= ymin:
            continue
        rows.append({"xmin": round(xmin), "ymin": round(ymin), "xmax": round(xmax), "ymax": round(ymax)})

    if not rows:
        return pd.DataFrame(), None

    ortho_basename = f"{prefix}_orthoimagery.tif"
    ortho_out = images_dir / ortho_basename
    write_rgb_uint8(ortho_in, ortho_out)

    df = pd.DataFrame(rows)
    df["image_path"] = str(ortho_out)
    df["source"] = SOURCE_LABEL
    return df, ortho_out


def run(extract_dir: Path, output_dir: Path) -> Path:
    """
    Process all plots and write MillionTrees-format annotations CSV.
    Copies orthos to output_dir/images/ and writes output_dir/annotations.csv.
    Returns path to annotations.csv.
    """
    extract_dir = Path(extract_dir)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for _zone_name, plot_subdir, prefix in PLOTS:
        df, _ = process_plot(extract_dir, images_dir, plot_subdir, prefix)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No plot data found under {extract_dir}. "
            "Extract Bile_Labe_Valley.zip and pass the extraction directory."
        )

    out_df = pd.concat(dfs, ignore_index=True)
    # Normalize to output images_dir so paths are consistent
    out_df["image_path"] = out_df["image_path"].apply(lambda p: str(Path(p).resolve()))
    annotations_path = output_dir / "annotations.csv"
    out_df.to_csv(annotations_path, index=False)
    return annotations_path


def main():
    parser = argparse.ArgumentParser(
        description="Process Zenodo 15591546 (Krkonoše Bílé Labe) to MillionTrees TreeBoxes format."
    )
    parser.add_argument(
        "extract_dir",
        type=Path,
        nargs="?",
        default=DEFAULT_BASE,
        help="Path to extracted Bile_Labe_Valley.zip (contains LOW_*, MID_*, HIGH_*). Default: %(default)s",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for annotations.csv and images/. Default: extract_dir.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Write sample overlay images (boxes on orthos) to savedir.",
    )
    parser.add_argument(
        "--savedir",
        type=Path,
        default=None,
        help="Where to save overlay images (used with --visualize). Default: <output_dir>/overlays",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.extract_dir
    if args.savedir is None:
        args.savedir = output_dir / "overlays"
    path = run(args.extract_dir, output_dir)
    n = len(pd.read_csv(path))
    print(f"Wrote {path} with {n} boxes.")
    if args.visualize:
        visualize_overlays(path, args.savedir)
        print(f"Sample overlays saved to {args.savedir}")


def _raster_to_uint8_rgb(data: "np.ndarray", nodata=None) -> "np.ndarray":
    """Convert raster bands (e.g. uint16) to 8-bit RGB for display. Per-band stretch.

    nodata pixels are excluded from the percentile stretch and rendered black; if
    they are not excluded (as with these orthos' 65535 fill), they blow out the
    range and the real imagery collapses to near-black.
    """
    import numpy as np

    if data.shape[0] >= 3:
        rgb = data[:3].astype(np.float64)  # (3, H, W)
    else:
        rgb = np.stack([data[0].astype(np.float64)] * 3, axis=0)
    out = np.zeros((rgb.shape[1], rgb.shape[2], 3), dtype=np.uint8)
    for c in range(3):
        band = rgb[c].copy()
        valid = band > 0 if nodata is None else band != nodata
        if valid.any():
            p1, p99 = np.percentile(band[valid], (1, 99))
            if p99 > p1:
                band = (band - p1) / (p99 - p1)
            else:
                band = np.where(valid, 1.0, 0.0)
        out[..., c] = np.clip(band * 255, 0, 255).astype(np.uint8)
    if nodata is not None:
        out[(rgb == nodata).all(axis=0)] = 0
    return out


def visualize_overlays(annotations_path: Path, savedir: Path) -> None:
    """Draw bounding boxes on orthos and save sample images."""
    import cv2
    import numpy as np

    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(annotations_path)
    for image_path in df["image_path"].unique():
        group = df[df["image_path"] == image_path]
        path = Path(image_path)
        if not path.exists():
            continue
        im = cv2.imread(str(path))
        if im is None or path.suffix.lower() in (".tif", ".tiff"):
            with rasterio.open(path) as src:
                data = src.read()
                src_nodata = src.nodata
            rgb = _raster_to_uint8_rgb(data, nodata=src_nodata)
            im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if im is None:
            continue
        for _, row in group.iterrows():
            x1, y1 = int(row["xmin"]), int(row["ymin"])
            x2, y2 = int(row["xmax"]), int(row["ymax"])
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        out = savedir / f"{path.stem}_overlay.png"
        cv2.imwrite(str(out), im)


if __name__ == "__main__":
    main()
