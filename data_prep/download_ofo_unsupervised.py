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
            "Optional dependency missing: 'geopandas'. Install with `pip install milliontrees[weak_supervised]` or `pip install geopandas`."
        ) from exc
    return gpd.read_file(path)


def _open_raster(path: str):
    try:
        import rasterio  # type: ignore
        from rasterio.windows import Window  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Optional dependency missing: 'rasterio'. Install with `pip install milliontrees[weak_supervised]` or `pip install rasterio`."
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
            "Optional dependency missing: 'requests'. Install with `pip install milliontrees[weak_supervised]` or `pip install requests`."
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
        # If not JSON, try parsing plain text response
        lines = resp.text.strip().split('\n')
        items = []
        
        if delimiter == '/':
            # For directory listing, extract unique mission IDs
            mission_ids = set()
            for line in lines:
                if line.strip() and line.startswith(prefix):
                    # Extract mission ID from path like "drone/missions_01/000029/..."
                    parts = line.replace(prefix, '').split('/')
                    if len(parts) > 0 and parts[0]:
                        mission_ids.add(parts[0])
            
            # Convert to expected format
            for mission_id in sorted(mission_ids):
                items.append({'subdir': f"{prefix}{mission_id}/"})
        else:
            # For file listing
            for line in lines:
                if line.strip():
                    items.append({'name': line.strip()})
        
        return items


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
            # IMPORTANT: Filter to only files that actually belong to this specific mission
            # Since Swift prefix matching can return files from other missions (e.g., 000029 matches 000030, 000031, etc.)
            if not name.startswith(f"drone/missions_01/{mission_id}/"):
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
    parquet_path: str,
    milliontrees_image_dir: str,
    patch_size: int = 800,
    num_missions: int = 5,
    allow_empty: bool = False,
):
    """
    Build a weak supervised OFO points parquet by tiling orthomosaics and mapping treetops points.

    Args:
        parquet_path: Input parquet file to load annotations and get missions to download
        milliontrees_image_dir: MillionTrees dataset directory (contains 'images').
        patch_size: Patch tile size in pixels.
        num_missions: Number of missions to process; default all.
        allow_empty: If True, include empty patches with no points.
    """


    images_dir = os.path.join(milliontrees_image_dir, 'images')
    ensure_dir(images_dir)
    out_dir = os.path.join(milliontrees_image_dir, 'weak_supervised')
    ensure_dir(out_dir)

    # Read annotations parquet to get mission IDs

    ann_df = pd.read_parquet(parquet_path)
    mission_ids = ann_df['mission_id'].unique().tolist()
    download_ofo_public(mission_ids=mission_ids, num_missions=num_missions, out_root=out_dir)
    missions = _collect_missions(out_dir, mission_ids=mission_ids)

    records = []

    for mission_path in missions:
        mission_id = os.path.basename(mission_path.rstrip(os.sep))

        # Find photogrammetry folder and inputs
        proc_dirs = glob(os.path.join(mission_path, 'processed_*'))
        if len(proc_dirs) == 0:
            print(f"No processed_* folder found for mission {mission_id}, skipping")
            continue
        
        # Find the processed directory that contains orthomosaic.tif
        proc_dir = None
        for candidate_dir in proc_dirs:
            test_orthomosaic = os.path.join(candidate_dir, 'full', 'orthomosaic.tif')
            if os.path.exists(test_orthomosaic):
                proc_dir = candidate_dir
                break
        
        if proc_dir is None:
            # Fallback to first directory if no orthomosaic found
            proc_dir = proc_dirs[0]
        
        orthomosaic = os.path.join(proc_dir, 'full', 'orthomosaic.tif')
        # ITD outputs live under the processed folder in public structure
        # Note: Swift API uses itd_0001 (underscore) not itd-0001 (hyphen)
        itd_dir = os.path.join(proc_dir, 'itd_0001')
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
                            'source': 'OFO treetops weak supervised',
                            'split': 'train',
                        })
                        records.append(df_tile)
                    elif allow_empty:
                        # Add empty row to mark tile as present (optional)
                        records.append(pd.DataFrame({
                            'filename': [out_name],
                            'x': [np.nan],
                            'y': [np.nan],
                            'source': ['OFO treetops weak supervised'],
                            'split': ['train'],
                        }))
        finally:
            ras.close()

    if len(records) == 0:
        print("No OFO tiles produced. Nothing to write.")
        return

    df = pd.concat(records, ignore_index=True)
    # Remove NaN rows if any
    df = df.dropna(subset=['x', 'y'])
    out_path = os.path.join(out_dir, "TreePoints_OFO_weak_supervised.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Wrote OFO weak supervised points to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Build OFO weak supervised points dataset')
    parser.add_argument('--data_dir', required=True, help='MillionTrees dataset directory')
    parser.add_argument('--patch_size', type=int, default=800)
    parser.add_argument('--allow_empty', action='store_true')
    parser.add_argument('--num_missions', type=int, default=None, help='Number of missions to include')
    args = parser.parse_args()
    run(
        data_dir=args.data_dir,
        patch_size=args.patch_size,
        allow_empty=args.allow_empty,
        num_missions=args.num_missions,
    )


if __name__ == '__main__':
    parse_args()
