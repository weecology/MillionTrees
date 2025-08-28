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
            "Optional dependency missing: 'neonutilities'. Install with `pip install milliontrees[neon]` or `pip install neonutilities`."
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
    # Post-process: move matching .tif to savepath root and remove nested dirs
    all_tifs = glob(os.path.join(savepath + "/DP3.30010.001", "**", "*.tif"),
                    recursive=True)
    chosen = sorted(
        [p for p in all_tifs if f"_{int(easting)}_{int(northing)} in p"],
        key=len)[0]
    dst = os.path.join(savepath, os.path.basename(chosen))
    shutil.move(chosen, dst)
    for d in next(os.walk(savepath + "/DP3.30010.001"))[1]:
        shutil.rmtree(os.path.join(savepath + "/DP3.30010.001", d),
                      ignore_errors=True)


def copy_downloads_to_images(download_root: str, images_dir: str) -> None:
    """Copy downloaded image tiles (e.g., .tif) into the dataset images directory."""
    ensure_dir(images_dir)
    tif_files = glob(os.path.join(download_root, "**", "*.tif"), recursive=True)
    for src in tif_files:
        dst = os.path.join(images_dir, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)


def infer_tile_column(df: pd.DataFrame, tile_col: Optional[str]) -> str:
    if tile_col and tile_col in df.columns:
        return tile_col
    # Try to derive from filename/image_path by dropping last underscore part
    candidate = None
    if 'tile_name' in df.columns:
        candidate = 'tile_name'
    elif 'filename' in df.columns:
        candidate = 'filename'
    elif 'image_path' in df.columns:
        candidate = 'image_path'
    if candidate is None:
        raise ValueError("Could not infer tile column. Provide --tile_column.")
    base = df[candidate].astype(str).apply(os.path.basename)
    df['__tile_name__'] = base.apply(lambda x: '_'.join(x.split('_')[:-1])
                                     if '_' in x else os.path.splitext(x)[0])
    return '__tile_name__'


def normalize_filenames_column(df: pd.DataFrame) -> pd.Series:
    if 'filename' in df.columns:
        return df['filename'].astype(str).apply(os.path.basename)
    if 'image_path' in df.columns:
        return df['image_path'].astype(str).apply(os.path.basename)
    raise ValueError(
        "Annotations must contain either 'filename' or 'image_path' column.")


def main():
    parser = argparse.ArgumentParser(
        description=
        "Download NEON tiles based on box annotations and append to MillionTrees dataset."
    )
    parser.add_argument(
        '--data_dir',
        required=True,
        help='Path to dataset directory, e.g., /path/to/TreeBoxes_v0.2')
    parser.add_argument('--annotations_parquet',
                        required=True,
                        help='Parquet of box annotations (unsupervised)')
    parser.add_argument('--site_column',
                        default='siteID',
                        help='Column name for NEON site ID')
    parser.add_argument(
        '--tile_column',
        default=None,
        help='Column name for tile identifier (if omitted, will be inferred)')
    parser.add_argument(
        '--year',
        type=int,
        default=None,
        help=
        'NEON year to request; if omitted, uses year parsed from filenames when possible'
    )
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
    args = parser.parse_args()

    data_dir = args.data_dir
    images_dir = os.path.join(data_dir, 'images')
    csv_path = os.path.join(data_dir, 'random.csv')
    ensure_dir(images_dir)
    ensure_dir(args.download_dir)

    # Load annotations (unsupervised parquet)
    ann = pd.read_parquet(args.annotations_parquet)
    if args.site_column not in ann.columns:
        raise ValueError(
            f"Annotations CSV must contain site column '{args.site_column}'.")

    # Normalize filename for later appending
    ann['filename'] = normalize_filenames_column(ann)

    # Infer tile column if needed
    tile_col = infer_tile_column(ann, args.tile_column)

    # Filter tiles per site if requested
    ann_filtered = filter_unique_tiles(ann, args.site_column, tile_col,
                                       args.max_tiles_per_site)

    # Download tiles
    token = read_neon_token(args.token_path)
    # Expect to be able to parse easting/northing from tile_name
    to_download = ann_filtered[[args.site_column, tile_col]].drop_duplicates()
    for _, row in to_download.iterrows():
        site = row[args.site_column]
        tile_name = str(row[tile_col])
        parsed = parse_tile_easting_northing(tile_name)
        if parsed is None:
            print(
                f"Could not parse easting/northing from tile '{tile_name}', skipping download."
            )
            continue
        easting, northing = parsed
        year = args.year if args.year is not None else None
        if year is None:
            # Try to find a 4-digit year in the tile name
            m = re.search(r"(20\d{2})", tile_name)
            if m:
                year = int(m.group(1))
        if year is None:
            print(
                f"No year specified and could not infer year from tile '{tile_name}', skipping."
            )
            continue
        print(
            f"Downloading site={site}, tile={tile_name}, easting={easting}, northing={northing}, year={year}"
        )
        download_tile_rgb(site=site,
                          easting=easting,
                          northing=northing,
                          year=year,
                          savepath=args.download_dir,
                          token=token,
                          data_product=args.data_product)

    # Copy downloaded tiles into dataset images directory
    copy_downloads_to_images(args.download_dir, images_dir)

    # Append annotations to train split
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected dataset CSV at {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = ['xmin', 'ymin', 'xmax', 'ymax']
    for col in required_cols:
        if col not in ann.columns:
            raise ValueError(f"Annotations CSV must contain column '{col}'.")

    to_append = ann.copy()
    to_append['filename'] = to_append['filename'].astype(str).apply(
        os.path.basename)
    to_append['source'] = 'Weinstein et al. 2018'
    to_append['split'] = 'train'

    # Keep only columns expected by dataset; preserve extras if present
    keep_cols = [
        c for c in
        ['xmin', 'ymin', 'xmax', 'ymax', 'filename', 'source', 'split']
        if c in to_append.columns
    ]
    to_append = to_append[keep_cols]

    updated = pd.concat([df, to_append], ignore_index=True)
    updated.to_csv(csv_path, index=False)
    print(f"Appended {len(to_append)} annotations to {csv_path}")

    # --- Tiling annotations per tile and saving as parquet collection ---
    tiled_output_dir = os.path.join(data_dir, 'unsupervised',
                                    'unsupervised_annotations_tiled')
    os.makedirs(tiled_output_dir, exist_ok=True)

    def tile_one_tile(site: str, tile_name: str) -> Optional[str]:
        # Expect downloaded image present in images_dir as .tif
        tif_candidates: List[str] = glob(os.path.join(images_dir, f"{tile_name}*.tif"))
        if len(tif_candidates) == 0:
            print(f"Warning: no downloaded image found for tile {tile_name}")
            return None
        tile_image_path = sorted(tif_candidates, key=len)[0]

        # Subset annotations for this tile (by image_path or filename base)
        tile_base = os.path.basename(tile_image_path)
        tile_stem = os.path.splitext(tile_base)[0]
        ann_tile = ann[ann['filename'].astype(str).str.contains(tile_stem)]
        if ann_tile.empty:
            print(f"Warning: no annotations found for tile {tile_name}")
            return None

        # Run DeepForest tiling
        save_dir = os.path.join(data_dir, 'unsupervised', 'images_tiled')
        os.makedirs(save_dir, exist_ok=True)
        try:
            tiled_df = split_raster(
                annotations_file=ann_tile,
                path_to_raster=tile_image_path,
                save_dir=save_dir,
                patch_size=args.patch_size,
                allow_empty=args.allow_empty,
            )
        except Exception as e:
            print(f"Error splitting raster {tile_name}: {e}")
            return None

        # Normalize schema for MillionTrees TreeBoxes
        # Ensure required columns
        if 'image_path' in tiled_df.columns:
            tiled_df = tiled_df.rename(columns={'image_path': 'filename'})
        tiled_df['source'] = 'Weinstein et al. 2018 unsupervised'
        tiled_df['split'] = 'train'

        keep_cols = [c for c in ['xmin', 'ymin', 'xmax', 'ymax', 'filename', 'source', 'split'] if c in tiled_df.columns]
        tiled_df = tiled_df[keep_cols]

        out_path = os.path.join(tiled_output_dir, f"{tile_stem}.parquet")
        tiled_df.to_parquet(out_path, index=False)
        return out_path

    # Build tasks with dask
    tasks = []
    for _, row in to_download.iterrows():
        site = row[args.site_column]
        tile_name = str(row[tile_col])
        tasks.append(delayed(tile_one_tile)(site, tile_name))

    if len(tasks) > 0:
        with ProgressBar():
            compute(*tasks, scheduler='threads', num_workers=args.num_workers)
        print(f"Wrote tiled parquet annotations to {tiled_output_dir}")


if __name__ == '__main__':
    main()
