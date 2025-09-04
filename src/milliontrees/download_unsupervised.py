import os
import re
import shutil
import argparse
from glob import glob
from typing import Optional, Tuple, List

import pandas as pd

# New dependencies for tiling and parallelization
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from deepforest.preprocess import split_raster


def read_neon_token(token_path: str = "neon_token.txt") -> str:
    """Read NEON API token from a text file."""
    with open(token_path, "r") as file:
        token = file.read().strip()
    return token


def parse_tile_easting_northing(tile_name: str) -> Optional[Tuple[int, int]]:
    """Parse NEON tile easting/northing from a tile name.

    Supports patterns like '123000_456000' or 'E123000_N456000'. Returns integers.
    """
    # Try E######_N###### pattern
    m = re.search(r"E(\d{3,7}).*?N(\d{3,7})", tile_name)
    if m:
        return int(m.group(1)), int(m.group(2))
    # Try ######_###### pattern
    m = re.search(r"(\d{3,7})[_-](\d{3,7})", tile_name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def filter_unique_tiles(df: pd.DataFrame, site_col: str, tile_col: str,
                        max_tiles_per_site: Optional[int]) -> pd.DataFrame:
    """Optionally limit number of unique tiles per site."""
    if max_tiles_per_site is None:
        return df
    # Keep at most N unique tiles per site
    tile_first = (df.drop_duplicates(
        [site_col, tile_col]).groupby(site_col).head(max_tiles_per_site))
    keep_tiles = set(zip(tile_first[site_col], tile_first[tile_col]))
    mask = list(zip(df[site_col], df[tile_col]))
    return df[[pair in keep_tiles for pair in mask]]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_tile_rgb(site: str,
                      easting: int,
                      northing: int,
                      year: int,
                      savepath: str,
                      token: str,
                      data_product: str = "DP3.30010.001") -> None:
    try:
        import neonutilities as nu  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Optional dependency missing: 'neonutilities'. Install with `pip install milliontrees[unsupervised]` or `pip install neonutilities`."
        ) from exc
    nu.by_tile_aop(
        dpid=data_product,
        site=site,
        easting=int(easting),
        northing=int(northing),
        year=year,
        token=token,
        include_provisional=True,
        check_size=False,
        savepath=savepath,
        verbose=True,
    )


def copy_downloads_to_images(download_root: str, images_dir: str) -> None:
    """Copy downloaded image tiles (e.g., .tif) into the dataset images directory."""
    ensure_dir(images_dir)
    tif_files = glob(os.path.join(download_root, "**", "*.tif"), recursive=True)
    for src in tif_files:
        dst = os.path.join(images_dir, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)


def normalize_filenames_column(df: pd.DataFrame) -> pd.Series:
    if 'filename' in df.columns:
        return df['filename'].astype(str).apply(os.path.basename)
    if 'image_path' in df.columns:
        return df['image_path'].astype(str).apply(os.path.basename)
    raise ValueError(
        "Annotations must contain either 'filename' or 'image_path' column.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Download NEON tiles based on annotations and append to MillionTrees dataset."
    )
    parser.add_argument(
        '--data_dir',
        required=True,
        help='Path to dataset directory, e.g., /path/to/TreeBoxes_v0.2')
    parser.add_argument('--annotations_parquet',
                        required=True,
                        help='Parquet of box annotations (unsupervised)')
    parser.add_argument(
        '--max_tiles_per_site',
        type=int,
        default=None,
        help='Optional limit on number of unique tiles per site')
    parser.add_argument('--patch_size',
                        type=int,
                        default=400,
                        help='Patch size for tiling (pixels). Default 400')
    parser.add_argument('--allow_empty',
                        action='store_true',
                        help='Include empty crops during tiling')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Parallel workers for tiling (dask)')
    parser.add_argument('--token_path',
                        default='neon_token.txt',
                        help='Path to NEON API token file')
    parser.add_argument('--data_product',
                        default='DP3.30010.001',
                        help='NEON data product (default RGB)')
    parser.add_argument(
        '--download_dir',
        default='neon_downloads',
        help='Temporary directory to store NEON downloads before copying')
    return parser.parse_args()


def run(data_dir, annotations_parquet, max_tiles_per_site, patch_size,
        allow_empty, num_workers, token_path, data_product, download_dir):
    images_dir = os.path.join(data_dir, 'images')
    # Load annotations (unsupervised parquet)
    ann = pd.read_parquet(annotations_parquet)

    # Normalize filename for later appending
    ann['filename'] = normalize_filenames_column(ann)

    # Filter tiles per site if requested
    ann_filtered = filter_unique_tiles(ann, 'siteID', 'tile_name',
                                       max_tiles_per_site)

    # Download tiles
    token = read_neon_token(token_path)
    # Expect to be able to parse easting/northing from tile_name
    to_download = ann_filtered[["siteID", "tile_name"]].drop_duplicates()
    for _, row in to_download.iterrows():
        site = row["siteID"]
        tile_name = str(row["tile_name"])
        parsed = parse_tile_easting_northing(tile_name)
        if parsed is None:
            print(
                f"Could not parse easting/northing from tile '{tile_name}', skipping download."
            )
            continue
        easting, northing = parsed

        # Year is the first 4 digits of the tile name
        year = int(tile_name[:4])

        print(
            f"Downloading site={site}, tile={tile_name}, easting={easting}, northing={northing}, year={year}"
        )
        download_tile_rgb(site=site,
                          easting=easting,
                          northing=northing,
                          year=year,
                          savepath=download_dir,
                          token=token,
                          data_product=data_product)

    # Copy downloaded tiles into dataset images directory
    copy_downloads_to_images(download_dir, images_dir)

    # --- Tiling annotations per tile and saving as parquet collection ---
    tiled_output_dir = os.path.join(data_dir, 'unsupervised',
                                    'unsupervised_annotations_tiled')
    os.makedirs(tiled_output_dir, exist_ok=True)

    def split_tile(ann_tile: pd.DataFrame,
                   tile_image_path: str) -> Optional[str]:
        # Run DeepForest tiling
        # Rename the tile_name to the image_path
        ann_tile['image_path'] = os.path.basename(tile_image_path)
        tiled_df = split_raster(
            annotations_file=ann_tile,
            path_to_raster=tile_image_path,
            save_dir=tiled_output_dir,
            patch_size=patch_size,
            allow_empty=allow_empty,
            root_dir=images_dir,
        )

        # Normalize schema for MillionTrees TreeBoxes
        tiled_df['filename'] = tiled_df['image_path']
        tiled_df['source'] = 'Weinstein et al. 2018 unsupervised'
        tiled_df['split'] = 'train'

        keep_cols = [
            c for c in
            ['xmin', 'ymin', 'xmax', 'ymax', 'filename', 'source', 'split']
            if c in tiled_df.columns
        ]
        tiled_df = tiled_df[keep_cols]
        return tiled_df

    # Build tasks with dask
    crop_annotations = []
    for tile_name in to_download.tile_name.unique():
        full_path = os.path.join(images_dir, tile_name)
        ann_tile = ann[ann.tile_name == tile_name]
        tiled_df = split_tile(ann_tile, full_path)
        crop_annotations.append(tiled_df)

    crop_annotations = pd.concat(crop_annotations)
    crop_annotations.to_parquet(os.path.join(
        data_dir, 'unsupervised', 'unsupervised_annotations_tiled.parquet'),
                                index=False)

    # tasks = []
    # for _, row in to_download.iterrows():
    #     site = row["siteID"]
    #     tile_name = str(row["tile_name"])
    #     tasks.append(delayed(split_tile)(ann_tile, tile_image_path))

    # if len(tasks) > 0:
    #     with ProgressBar():
    #         compute(*tasks, scheduler='threads', num_workers=num_workers)
    #     print(f"Wrote tiled parquet annotations to {tiled_output_dir}")


if __name__ == '__main__':
    run(parse_args())
