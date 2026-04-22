"""
Download and prepare DeepTrees Halle (Saale), Germany for MillionTrees.

Issue: https://github.com/weecology/MillionTrees/issues/132
Zenodo record: https://zenodo.org/records/19695972
DOI: https://doi.org/10.5281/zenodo.19695972

This script:
1) Downloads ``deeptrees-halle.zip`` from Zenodo (if missing)
2) Extracts archive contents (if needed)
3) Converts per-tile polygon shapefiles to MillionTrees annotation format

Dataset specifics from Zenodo metadata:
- tiles: ``tiles/tile_<x>_<y>.tif``
- labels: ``labels/label_tile_<x>_<y>.shp``
- classes: 0=tree, 1=cluster of trees, 2=unsure

By default, only class ``0`` (tree) polygons are retained.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from deepforest.utilities import read_file

SOURCE_NAME = "Khan et al. 2026"
RECORD_ID = 19695972
DEFAULT_ROOT = Path("/orange/ewhite/DeepForest/Zenodo_19695972")
ZIP_NAME = "deeptrees-halle.zip"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"


def _archive_url_from_record(api_url: str = API_URL, archive_name: str = ZIP_NAME) -> str:
    """Fetch Zenodo record JSON and return the download URL for ``archive_name``."""
    response = requests.get(api_url, timeout=60)
    response.raise_for_status()
    payload = response.json()
    for f in payload.get("files", []):
        if f.get("key") == archive_name:
            return f["links"]["self"]
    raise FileNotFoundError(f"{archive_name} not found in Zenodo record {api_url}")


def download_archive(root: Path, force: bool = False) -> Path:
    """Download the dataset zip into ``root`` and return archive path."""
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / ZIP_NAME
    if archive_path.exists() and not force:
        return archive_path

    url = _archive_url_from_record()
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(archive_path, "wb") as out:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out.write(chunk)

    return archive_path


def extract_archive(archive_path: Path, extract_dir: Path, force: bool = False) -> Path:
    """Extract archive to ``extract_dir`` and return the extraction root."""
    tiles_dir = extract_dir / "tiles"
    labels_dir = extract_dir / "labels"
    if tiles_dir.exists() and labels_dir.exists() and not force:
        return extract_dir

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(extract_dir)

    return extract_dir


def _clean_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Explode multipolygons and keep polygon geometry rows only."""
    if gdf.empty:
        return gdf
    if "MultiPolygon" in gdf.geometry.geom_type.unique():
        try:
            gdf = gdf.explode(index_parts=False)
        except TypeError:
            gdf = gdf.explode().reset_index(drop=True)
    return gdf[gdf.geometry.geom_type == "Polygon"]


def prepare_annotations(
    extract_dir: Path,
    output_csv: Path,
    keep_classes: tuple[int, ...] = (0,),
) -> pd.DataFrame:
    """Convert DeepTrees labels to MillionTrees-format annotations CSV."""
    tiles_dir = extract_dir / "tiles"
    labels_dir = extract_dir / "labels"

    if not tiles_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"Expected {tiles_dir} and {labels_dir}; run extraction first."
        )

    label_files = sorted(labels_dir.glob("label_tile_*.shp"))
    if not label_files:
        raise FileNotFoundError(f"No label shapefiles found in {labels_dir}")

    all_annotations: list[pd.DataFrame] = []
    for shp_path in label_files:
        tile_name = shp_path.name.replace("label_", "").replace(".shp", ".tif")
        tif_path = tiles_dir / tile_name
        if not tif_path.exists():
            continue

        gdf = gpd.read_file(shp_path)
        if gdf.empty:
            continue

        if "class" in gdf.columns:
            classes = pd.to_numeric(gdf["class"], errors="coerce")
            gdf = gdf[classes.isin(keep_classes)]
            if gdf.empty:
                continue

        gdf = _clean_polygons(gdf)
        if gdf.empty:
            continue

        gdf = gdf.copy()
        gdf["image_path"] = tile_name
        gdf["label"] = "Tree"

        ann = read_file(gdf, root_dir=str(tiles_dir))
        ann["source"] = SOURCE_NAME
        all_annotations.append(ann[["image_path", "geometry", "source"]])

    if not all_annotations:
        raise ValueError("No annotations were generated.")

    annotations = pd.concat(all_annotations, ignore_index=True)
    annotations["image_path"] = annotations["image_path"].apply(
        lambda x: str((tiles_dir / Path(x).name).resolve())
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    annotations.to_csv(output_csv, index=False)
    return annotations


def run(root: Path, force_download: bool = False, force_extract: bool = False) -> Path:
    """Execute full download + prepare workflow and return annotations path."""
    archive_path = download_archive(root=root, force=force_download)
    extract_dir = extract_archive(
        archive_path=archive_path,
        extract_dir=root,
        force=force_extract,
    )
    output_csv = root / "annotations.csv"
    annotations = prepare_annotations(extract_dir=extract_dir, output_csv=output_csv)
    print(
        f"Wrote {len(annotations)} polygons over "
        f"{annotations['image_path'].nunique()} images to {output_csv}"
    )
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download and prepare DeepTrees Halle (Zenodo 19695972) "
            "to MillionTrees polygon CSV format."
        ))
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=(
            "Dataset root for zip, extracted files, and annotations.csv "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download deeptrees-halle.zip even if it exists.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract archive even if tiles/ and labels/ already exist.",
    )
    args = parser.parse_args()
    run(root=args.root, force_download=args.force_download, force_extract=args.force_extract)


if __name__ == "__main__":
    main()
