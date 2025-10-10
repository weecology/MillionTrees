import os
import argparse
from glob import glob
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from urllib.parse import urljoin


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_geopandas(path: str):
    try:
        import geopandas as gpd  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Optional dependency missing: 'geopandas'. Install with `pip install milliontrees[unsupervised]` or `pip install geopandas`."
        ) from exc
    return gpd.read_file(path)


def _open_raster(path: str):
    try:
        import rasterio  # type: ignore
        from rasterio.windows import Window  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Optional dependency missing: 'rasterio'. Install with `pip install milliontrees[unsupervised]` or `pip install rasterio`."
        ) from exc
    return __import__('rasterio').open(path)


def _rowcol(transform, xs, ys):
    from rasterio.transform import rowcol  # type: ignore
    return rowcol(transform, xs, ys)


def _save_png(img_arr: np.ndarray, out_path: str) -> None:
    from PIL import Image

    # img_arr expected shape (bands, rows, cols)
    if img_arr.ndim == 3:
        img_arr = np.transpose(img_arr, (1, 2, 0))
    elif img_arr.ndim == 2:
        img_arr = img_arr[:, :, None]

    # Select first 3 channels if more
    if img_arr.shape[2] > 3:
        img_arr = img_arr[:, :, :3]
    # If less than 3, tile
    if img_arr.shape[2] == 1:
        img_arr = np.repeat(img_arr, 3, axis=2)

    # Normalize to uint8 if needed
    if img_arr.dtype != np.uint8:
        arr = img_arr.astype(np.float32)
        if arr.max() > 255.0:
            arr = arr / 256.0
        elif arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = img_arr

    Image.fromarray(arr).save(out_path)


# ------------------------
# Jetstream2 (Swift) helpers
# ------------------------

def _require_requests():
    try:
        import requests  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Optional dependency missing: 'requests'. Install with `pip install milliontrees[unsupervised]` or `pip install requests`."
        ) from exc
    return __import__('requests')


def _swift_list(endpoint: str, bucket: str, prefix: str, delimiter: str = '/') -> List[dict]:
    requests = _require_requests()
    base = endpoint.rstrip('/') + f"/swift/v1/{bucket}"
    resp = requests.get(base, params={
        'format': 'json',
        'prefix': prefix,
        'delimiter': delimiter,
    }, timeout=60)
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        # If not JSON (truncated HTML listing), return empty to fail gracefully
        return []


def _stream_download(url: str, dest_path: str, chunk_size: int = 1024 * 1024) -> None:
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


def list_public_ofo_missions(endpoint: str = 'https://js2.jetstream-cloud.org:8001',
                             bucket: str = 'ofo-public',
                             limit: Optional[int] = None) -> List[str]:
    """List mission IDs available in the public OFO bucket via Swift API."""
    prefix = 'drone/missions_01/'
    entries = _swift_list(endpoint, bucket, prefix=prefix, delimiter='/')
    mission_ids = []
    for item in entries:
        # Swift JSON returns objects with key 'subdir' when using delimiter
        subdir = item.get('subdir')
        if subdir and subdir.startswith(prefix):
            mission_id = os.path.basename(subdir.rstrip('/'))
            if mission_id:
                mission_ids.append(mission_id)
    mission_ids.sort()
    if limit is not None:
        mission_ids = mission_ids[:limit]
    return mission_ids


def download_ofo_public(
    out_root: str,
    endpoint: str = 'https://js2.jetstream-cloud.org:8001',
    bucket: str = 'ofo-public',
    mission_ids: Optional[List[str]] = None,
    num_missions: int = 5,
) -> List[str]:
    """Download OFO public missions (orthomosaic and ITD results) to a local root.

    Returns list of mission IDs that were attempted.
    """
    ensure_dir(out_root)
    if mission_ids is None or len(mission_ids) == 0:
        mission_ids = list_public_ofo_missions(endpoint=endpoint, bucket=bucket, limit=num_missions)
    else:
        mission_ids = mission_ids[:num_missions]

    for mission_id in mission_ids:
        print(f"Processing mission {mission_id}")

        # 1) Orthomosaic COG (save locally as orthomosaic.tif for downstream compatibility)
        ortho_http = endpoint.rstrip('/') + f"/ofo-public/drone/missions_01/{mission_id}/processed_02/full/{mission_id}_ortho-dsm-ptcloud.tif"
        local_ortho = os.path.join(out_root, mission_id, 'processed_02', 'full', 'orthomosaic.tif')
        try:
            if not os.path.exists(local_ortho):
                print(f"  Downloading orthomosaic: {ortho_http}")
                _stream_download(ortho_http, local_ortho)
        except Exception as e:
            print(f"  Warning: failed to download orthomosaic for {mission_id}: {e}")

        # 2) ITD results (treetops and crowns) under processed_02/itd-0001/
        itd_prefix = f"drone/missions_01/{mission_id}/processed_02/itd-0001/"
        items = _swift_list(endpoint, bucket, prefix=itd_prefix, delimiter='\n')
        # When no delimiter is provided, Swift returns a flat list of objects with 'name'
        if not items:
            items = _swift_list(endpoint, bucket, prefix=itd_prefix, delimiter='')

        files = []
        for obj in items:
            name = obj.get('name')
            if not name:
                continue
            # Keep vector and metadata files commonly used for treetops/crowns
            ext = os.path.splitext(name)[1].lower()
            if ext in {'.gpkg', '.geojson', '.json', '.shp', '.shx', '.dbf', '.prj', '.cpg', '.xml', '.csv', '.tif'}:
                files.append(name)

        for remote_name in files:
            rel = os.path.relpath(remote_name, start=f"drone/missions_01/{mission_id}")
            local_path = os.path.join(out_root, mission_id, rel)
            url = endpoint.rstrip('/') + f"/swift/v1/{bucket}/" + remote_name
            try:
                if not os.path.exists(local_path):
                    print(f"  Downloading ITD file: {url}")
                    _stream_download(url, local_path)
            except Exception as e:
                print(f"  Warning: failed to download {remote_name}: {e}")

    return mission_ids


def _collect_missions(ofo_root: str, mission_ids: Optional[List[str]]) -> List[str]:
    if mission_ids is not None and len(mission_ids) > 0:
        return [os.path.join(ofo_root, mid) for mid in mission_ids]
    # otherwise list directories under root
    return [
        os.path.join(ofo_root, d) for d in os.listdir(ofo_root)
        if os.path.isdir(os.path.join(ofo_root, d))
    ]


def run(
    data_dir: str,
    ofo_root: str,
    patch_size: int = 400,
    allow_empty: bool = False,
    mission_ids: Optional[List[str]] = None,
    photogrammetry_glob: str = 'processed_*',
    output_parquet_name: str = 'TreePoints_OFO_unsupervised.parquet',
    split: str = 'train',
):
    """
    Build an unsupervised OFO points parquet by tiling orthomosaics and mapping treetops points.

    Args:
        data_dir: MillionTrees dataset directory (contains 'images').
        ofo_root: Local path to 'Community Data/ofo/public/missions'.
        patch_size: Patch tile size in pixels.
        allow_empty: If True, include empty patches with no points.
        mission_ids: Optional list of mission_id to process; default all.
        photogrammetry_glob: Glob under mission to find processed dir.
        output_parquet_name: Filename for parquet under unsupervised/.
        split: Split name to set for produced annotations.
    """
    images_dir = os.path.join(data_dir, 'images')
    ensure_dir(images_dir)
    out_dir = os.path.join(data_dir, 'unsupervised')
    ensure_dir(out_dir)

    missions = _collect_missions(ofo_root, mission_ids)

    records = []

    for mission_path in missions:
        mission_id = os.path.basename(mission_path.rstrip(os.sep))

        # Find photogrammetry folder and inputs
        proc_dirs = glob(os.path.join(mission_path, photogrammetry_glob))
        if len(proc_dirs) == 0:
            print(f"No processed_* folder found for mission {mission_id}, skipping")
            continue
        proc_dir = proc_dirs[0]
        orthomosaic = os.path.join(proc_dir, 'full', 'orthomosaic.tif')
        # ITD outputs live under the processed folder in public structure
        itd_dir = os.path.join(proc_dir, 'itd-0001')
        treetops = None
        # Prefer .gpkg then .geojson
        candidates = [
            os.path.join(itd_dir, 'treetops.gpkg'),
            os.path.join(itd_dir, 'treetops.geojson'),
        ]
        # Fallback: any file containing 'treetop'
        if not any(os.path.exists(p) for p in candidates):
            if os.path.isdir(itd_dir):
                for ext in ('*.gpkg', '*.geojson'):
                    for p in glob(os.path.join(itd_dir, ext)):
                        base = os.path.basename(p).lower()
                        if 'treetop' in base or 'tree_top' in base:
                            candidates.append(p)
        # Choose first existing
        for p in candidates:
            if os.path.exists(p):
                treetops = p
                break

        if not os.path.exists(orthomosaic) or treetops is None or not os.path.exists(treetops):
            print(f"Missing inputs for {mission_id}, skipping: {orthomosaic}, {treetops}")
            continue

        # Open data
        gdf = _read_geopandas(treetops)
        ras = _open_raster(orthomosaic)
        try:
            raster_crs = ras.crs
            if hasattr(gdf, 'crs') and gdf.crs is not None and raster_crs is not None and gdf.crs != raster_crs:
                gdf = gdf.to_crs(raster_crs)

            xs = gdf.geometry.x.values
            ys = gdf.geometry.y.values
            rows, cols = _rowcol(ras.transform, xs, ys)
            # col->x, row->y pixel coords
            pts = pd.DataFrame({'x': cols.astype(int), 'y': rows.astype(int)})

            height = ras.height
            width = ras.width

            # Tiling
            for y0 in range(0, height, patch_size):
                for x0 in range(0, width, patch_size):
                    x1 = min(x0 + patch_size, width)
                    y1 = min(y0 + patch_size, height)

                    # Points within tile bounds
                    mask = (pts['x'] >= x0) & (pts['x'] < x1) & (pts['y'] >= y0) & (pts['y'] < y1)
                    pts_tile = pts[mask]
                    if not allow_empty and len(pts_tile) == 0:
                        continue

                    # Read window and write PNG
                    from rasterio.windows import Window  # type: ignore
                    window = Window(col_off=x0, row_off=y0, width=x1 - x0, height=y1 - y0)
                    img = ras.read(window=window)  # (bands, rows, cols)

                    out_name = f"{mission_id}_{x0}_{y0}.png"
                    out_path = os.path.join(images_dir, out_name)
                    _save_png(img, out_path)

                    # Add point rows adjusted to local tile coords
                    if len(pts_tile) > 0:
                        df_tile = pd.DataFrame({
                            'filename': out_name,
                            'x': (pts_tile['x'] - x0).astype(int).values,
                            'y': (pts_tile['y'] - y0).astype(int).values,
                            'source': 'OFO treetops unsupervised',
                            'split': split,
                        })
                        records.append(df_tile)
                    elif allow_empty:
                        # Add empty row to mark tile as present (optional)
                        records.append(pd.DataFrame({
                            'filename': [out_name],
                            'x': [np.nan],
                            'y': [np.nan],
                            'source': ['OFO treetops unsupervised'],
                            'split': [split],
                        }))
        finally:
            ras.close()

    if len(records) == 0:
        print("No OFO tiles produced. Nothing to write.")
        return

    df = pd.concat(records, ignore_index=True)
    # Remove NaN rows if any
    df = df.dropna(subset=['x', 'y'])
    out_path = os.path.join(out_dir, output_parquet_name)
    df.to_parquet(out_path, index=False)
    print(f"Wrote OFO unsupervised points to {out_path}")



def parse_args():
    parser = argparse.ArgumentParser(description='Build OFO unsupervised points dataset')
    parser.add_argument('--data_dir', required=True, help='MillionTrees dataset directory')
    parser.add_argument('--ofo_root', required=True, help='Path to OFO missions root (local)')
    parser.add_argument('--patch_size', type=int, default=400)
    parser.add_argument('--allow_empty', action='store_true')
    parser.add_argument('--mission_ids', type=str, default=None, help='Comma-separated mission IDs to include')
    parser.add_argument('--photogrammetry_glob', type=str, default='processed_*')
    parser.add_argument('--output_parquet_name', type=str, default='TreePoints_OFO_unsupervised.parquet')
    parser.add_argument('--split', type=str, default='train')
    args = parser.parse_args()
    mission_ids = None
    if args.mission_ids:
        mission_ids = [s.strip() for s in args.mission_ids.split(',') if s.strip()]
    run(
        data_dir=args.data_dir,
        ofo_root=args.ofo_root,
        patch_size=args.patch_size,
        allow_empty=args.allow_empty,
        mission_ids=mission_ids,
        photogrammetry_glob=args.photogrammetry_glob,
        output_parquet_name=args.output_parquet_name,
        split=args.split,
    )


if __name__ == '__main__':
    parse_args()
