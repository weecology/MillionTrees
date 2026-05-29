#!/usr/bin/env python3
"""Process OFO field-validated tree maps into a MillionTrees TreePoints dataset.

The Open Forest Observatory (OFO) ground-reference catalog contains stem maps where each tree is
manually surveyed in the field and then registered to a specific drone mission's photogrammetry
products. David Young provided a concatenated geopackage of 330 plot-drone pairs against the
``missions_03`` photogrammetry run on the public ``ofo-public`` bucket at
``https://js2.jetstream-cloud.org:8001``.

For each unique ``mission_id`` in the field-trees file we download the matching orthomosaic from
``drone/missions_03/{mission_id}/photogrammetry_03/full/{mission_id}_ortho-dsm-ptcloud.tif``, tile
the orthomosaic into uniform patches, and emit per-tile point annotations in image coordinates with
the standard MillionTrees columns. Trees marked ``withhold_from_training`` are routed to the test
split. By default we also drop trees that are not flagged ``predicted_overstory`` since canopy-only
field stems are the only ones expected to be visible from above.
"""

import argparse
import os
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from deepforest import preprocess
from deepforest.utilities import read_file

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

OFO_ENDPOINT = 'https://js2.jetstream-cloud.org:8001'
OFO_BUCKET = 'ofo-public'
MISSION_PREFIX = 'drone/missions_03'


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _require_requests():
    import requests
    return requests


def _stream_download(url: str,
                     dest_path: str,
                     chunk_size: int = 1024 * 1024) -> None:
    requests = _require_requests()
    ensure_dir(os.path.dirname(dest_path))
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        tmp_path = dest_path + ".part"
        with open(tmp_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        os.replace(tmp_path, dest_path)


def download_mission_orthomosaic(
    mission_id: str,
    out_root: str,
    endpoint: str = OFO_ENDPOINT,
    bucket: str = OFO_BUCKET,
) -> Optional[str]:
    """Download a missions_03 orthomosaic for ``mission_id``.

    Returns the local path to ``orthomosaic.tif`` on success, or None when no remote orthomosaic was
    available. Trees registered against ``missions_03`` only point to the ``ortho-dsm-ptcloud.tif``
    product in the ``photogrammetry_03/full/`` subfolder.
    """
    ortho_filename = f"{mission_id}_ortho-dsm-ptcloud.tif"
    remote_url = (f"{endpoint.rstrip('/')}/swift/v1/{bucket}/"
                  f"{MISSION_PREFIX}/{mission_id}/photogrammetry_03/full/"
                  f"{ortho_filename}")
    local_ortho = os.path.join(out_root, mission_id, 'photogrammetry_03',
                               'full', 'orthomosaic.tif')

    if os.path.exists(local_ortho):
        return local_ortho

    requests = _require_requests()
    head = requests.head(remote_url, timeout=60, allow_redirects=True)
    if head.status_code != 200:
        print(f"  Warning: no ortho for {mission_id} ({head.status_code})")
        return None

    print(f"  Downloading orthomosaic for mission {mission_id}")
    _stream_download(remote_url, local_ortho)
    return local_ortho


def load_field_trees(path: str,
                     only_overstory: bool = True) -> gpd.GeoDataFrame:
    """Read the concatenated OFO field-trees file and apply standard filtering.

    Args:
        path: Path to a geopackage / shapefile containing rows for each tree x mission pairing. The
            schema is the ofo-public ground-reference schema augmented with ``mission_id``,
            ``withhold_from_training`` and ``predicted_overstory`` per David Young's note.
        only_overstory: When True, keep only trees flagged ``predicted_overstory`` (or where
            ``ohvis`` is explicitly True). These are the trees expected to be detectable in nadir
            drone imagery.

    Returns:
        A GeoDataFrame whose ``mission_id`` column is normalized to a 6-digit zero-padded string and
        that has been filtered to overstory-visible trees by default.
    """
    gdf = gpd.read_file(path)

    if 'mission_id' not in gdf.columns:
        raise ValueError(
            "Field trees file is missing the 'mission_id' column. "
            "Per David Young's note, trees overlapping multiple missions should be duplicated with"
            " a 'mission_id' attribute identifying the drone pairing.")

    gdf['mission_id'] = (gdf['mission_id'].astype(str).str.split(
        '-').str[0].str.zfill(6))

    if only_overstory:
        if 'predicted_overstory' in gdf.columns:
            mask = gdf['predicted_overstory'].astype('boolean').fillna(False)
            if 'ohvis' in gdf.columns:
                # ohvis is a numeric code (1 == overhead-visible); other codes
                # (e.g. stray 7.0) and NaN are not "explicitly visible".
                ohvis_visible = (gdf['ohvis'] == 1).fillna(False)
                mask = mask | ohvis_visible
            gdf = gdf[mask].copy()
        else:
            print("Warning: 'predicted_overstory' missing; skipping overstory filter")

    return gdf


def tile_mission(
    mission_id: str,
    orthomosaic_path: str,
    field_trees: gpd.GeoDataFrame,
    images_dir: str,
    patch_size: int = 800,
) -> Optional[pd.DataFrame]:
    """Tile a single mission's orthomosaic and project field trees into image coordinates.

    Reuses ``deepforest.preprocess.split_raster`` after rewriting the orthomosaic as a clean
    3-band uint8 raster (Jetstream2 orthos are 4-band float with NaNs, which split_raster cannot
    consume directly).
    """
    mission_trees = field_trees[field_trees['mission_id'] == mission_id].copy()
    if mission_trees.empty:
        return None

    with rio.open(orthomosaic_path) as src:
        arr = src.read()
        profile = src.profile.copy()
        raster_crs = src.crs

    arr[np.isnan(arr)] = 0
    arr3 = arr[:3, :, :].astype(np.uint8)
    # BIGTIFF=IF_SAFER lets large orthos (>4 GB uncompressed) write as BigTIFF
    # while keeping small ones as classic TIFF.
    profile.update(count=3, dtype=rio.uint8, nodata=0, BIGTIFF='IF_SAFER')

    out_ortho = os.path.join(os.path.dirname(orthomosaic_path),
                             f"{mission_id}_ortho.tif")
    with rio.open(out_ortho, 'w', **profile) as dst:
        dst.write(arr3)

    if mission_trees.crs is not None and raster_crs is not None and mission_trees.crs != raster_crs:
        mission_trees = mission_trees.to_crs(raster_crs)

    mission_trees['image_path'] = os.path.basename(out_ortho)
    mission_trees['label'] = 'Tree'
    keep_cols = ['image_path', 'label', 'geometry']
    if 'withhold_from_training' in mission_trees.columns:
        mission_trees['withhold_from_training'] = mission_trees[
            'withhold_from_training'].astype('boolean').fillna(False).astype(bool)
        keep_cols.append('withhold_from_training')
    df_for_split = mission_trees[keep_cols].copy()
    df_for_split = read_file(df_for_split, root_dir=os.path.dirname(out_ortho))

    height, width = arr3.shape[1], arr3.shape[2]
    if height < patch_size or width < patch_size:
        print(f"  {mission_id}: ortho {width}x{height} smaller than patch_size"
              f" {patch_size}, skipping")
        return None

    # read_file projects points into ortho-pixel coordinates; drop any field
    # trees that fall outside the downloaded orthomosaic footprint. They cannot
    # appear on any tile and would otherwise trip split_raster's bounds check.
    px = df_for_split.geometry.x
    py = df_for_split.geometry.y
    in_bounds = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    n_out = int((~in_bounds).sum())
    if n_out:
        print(f"  {mission_id}: dropping {n_out}/{len(df_for_split)} trees"
              " outside ortho extent")
    df_for_split = df_for_split[in_bounds].copy()
    if df_for_split.empty:
        print(f"  {mission_id}: no trees within ortho extent, skipping")
        return None

    points_tiled = preprocess.split_raster(
        df_for_split,
        path_to_raster=out_ortho,
        root_dir=os.path.dirname(out_ortho),
        save_dir=images_dir,
        patch_size=patch_size,
        patch_overlap=0,
        allow_empty=False,
    )

    points_tiled = points_tiled.rename(columns={'image_path': 'filename'})
    points_tiled['mission_id'] = str(mission_id)
    points_tiled['source'] = 'OFO field 2025'

    if 'withhold_from_training' in points_tiled.columns:
        points_tiled['split'] = np.where(
            points_tiled['withhold_from_training'].astype(bool), 'test',
            'train')
    else:
        points_tiled['split'] = 'train'

    return points_tiled


def write_sample_overlays(images_dir: str,
                          annotations_csv: str,
                          savedir: str,
                          max_samples: int = 8) -> None:
    """Render a handful of patches with overlaid points for quick QC.

    Trains/tests are colored differently so reviewers can see which trees are routed to the test
    split via ``withhold_from_training``.
    """
    from matplotlib import pyplot as plt

    ensure_dir(savedir)
    df = pd.read_csv(annotations_csv)
    df['basename'] = df['image_path'].apply(os.path.basename)
    for basename in sorted(df['basename'].unique())[:max_samples]:
        sub = df[df['basename'] == basename]
        img = np.array(Image.open(os.path.join(images_dir, basename)))
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        train = sub[sub['split'] == 'train']
        test = sub[sub['split'] == 'test']
        if not train.empty:
            ax.scatter(train['x'], train['y'], c='cyan', s=80, marker='+',
                       linewidths=2.0, label='train')
        if not test.empty:
            ax.scatter(test['x'], test['y'], c='red', s=80, marker='x',
                       linewidths=2.0, label='test (withheld)')
        ax.set_title(f"{basename} — {len(sub)} field trees")
        ax.legend(loc='upper right')
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(savedir, f"sample_points_{basename}"),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)
    print(f"Sample overlay plots written to {savedir}")


def run(
    field_trees_path: str,
    output_dir: str,
    ofo_root: str,
    patch_size: int = 800,
    num_missions: Optional[int] = None,
    only_overstory: bool = True,
    write_overlays: bool = False,
    overlays_dir: Optional[str] = None,
) -> str:
    """End-to-end pipeline: download orthos, tile, write TreePoints_OFO_field.csv.

    Returns the path of the written CSV.
    """
    ensure_dir(ofo_root)
    images_dir = os.path.join(output_dir, 'images')
    ensure_dir(images_dir)

    field_trees = load_field_trees(field_trees_path,
                                   only_overstory=only_overstory)
    mission_ids = sorted(field_trees['mission_id'].unique())
    if num_missions is not None:
        mission_ids = mission_ids[:num_missions]
    print(f"Processing {len(mission_ids)} missions: {mission_ids}")

    tiled_records: List[pd.DataFrame] = []
    failed_missions: List[str] = []
    for mid in mission_ids:
        ortho_path = download_mission_orthomosaic(mid, ofo_root)
        if ortho_path is None:
            continue
        try:
            tiled = tile_mission(mid, ortho_path, field_trees, images_dir,
                                 patch_size=patch_size)
        except Exception as e:  # noqa: BLE001 - keep going so one bad mission
            print(f"  ERROR tiling mission {mid}: {e}")
            failed_missions.append(mid)
            continue
        if tiled is not None:
            tiled_records.append(tiled)

    if failed_missions:
        print(f"Skipped {len(failed_missions)} missions due to errors:"
              f" {failed_missions}")

    if not tiled_records:
        raise RuntimeError(
            "No tiled annotations produced. Check field-trees mission_id values and orthomosaic"
            " availability on Jetstream2.")

    combined = pd.concat(tiled_records, ignore_index=True)
    combined['image_path'] = combined['filename'].apply(
        lambda fn: os.path.join(images_dir, fn))
    combined = combined.drop(columns=['filename'])

    out_csv = os.path.join(output_dir, 'TreePoints_OFO_field.csv')
    combined.to_csv(out_csv, index=False)
    print(f"Wrote {len(combined)} annotations across {combined['image_path'].nunique()} tiles to"
          f" {out_csv}")

    if write_overlays:
        write_sample_overlays(images_dir, out_csv,
                              overlays_dir or os.path.join(output_dir, 'sample_plots'))

    return out_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build TreePoints_OFO_field dataset from OFO field-validated trees')
    parser.add_argument('--field_trees', required=True,
                        help='Geopackage of concatenated field trees with mission_id')
    parser.add_argument('--output_dir', required=True,
                        help='MillionTrees-style output directory (will contain images/)')
    parser.add_argument('--ofo_root', required=True,
                        help='Local cache dir for downloaded orthomosaics')
    parser.add_argument('--patch_size', type=int, default=800)
    parser.add_argument('--num_missions', type=int, default=None,
                        help='Limit number of missions (handy for local tests)')
    parser.add_argument('--include_understory', action='store_true',
                        help='Keep trees regardless of predicted_overstory flag')
    parser.add_argument('--sample_plots', action='store_true',
                        help='Write per-tile QC overlay PNGs alongside the CSV')
    parser.add_argument('--sample_plots_dir', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run(
        field_trees_path=args.field_trees,
        output_dir=args.output_dir,
        ofo_root=args.ofo_root,
        patch_size=args.patch_size,
        num_missions=args.num_missions,
        only_overstory=not args.include_understory,
        write_overlays=args.sample_plots,
        overlays_dir=args.sample_plots_dir,
    )


if __name__ == '__main__':
    main()
