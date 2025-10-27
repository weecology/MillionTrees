#!/usr/bin/env python3
"""
Combined script to download OFO annotations and images, tile them, and prepare datasets.
This script:
1. Downloads OFO mission annotations (treetops and crowns) from Jetstream2 Swift storage
2. Downloads corresponding orthomosaic imagery
3. Tiles the orthomosaics and creates patch-level annotations
4. Formats data for TreeBoxes and TreePoints datasets
"""

import os
import argparse
import glob
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from deepforest import preprocess
from deepforest.utilities import read_file

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def download_ofo_data(
    out_root: str,
    endpoint: str = 'https://js2.jetstream-cloud.org:8001',
    bucket: str = 'ofo-public',
    mission_ids: Optional[List[str]] = None,
    num_missions: Optional[int] = None,
    download_orthomosaics: bool = True,
) -> List[str]:
    """Download OFO public missions (annotations and optionally orthomosaics) to a local root.

    Returns list of mission IDs that were attempted.
    """
    ensure_dir(out_root)
    if mission_ids is None or len(mission_ids) == 0:
        mission_ids = list_public_ofo_missions(endpoint=endpoint, bucket=bucket, limit=num_missions)
    else:
        if num_missions is not None:
            mission_ids = mission_ids[:num_missions]

    for i, mission_id in enumerate(mission_ids):
        print(f"[{i+1}/{len(mission_ids)}] Processing mission {mission_id}")

        # Download ITD results (treetops and crowns) under processed_02/itd_0001/
        itd_prefix = f"drone/missions_01/{mission_id}/processed_02/itd_0001/"
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
            # Keep only vector files for annotations (treetops and crowns)
            ext = os.path.splitext(name)[1].lower()
            if ext in {'.gpkg', '.geojson', '.json', '.shp', '.shx', '.dbf', '.prj', '.cpg', '.xml', '.csv', '.tif'}:
                files.append(name)

        for remote_name in files:
            rel = os.path.relpath(remote_name, start=f"drone/missions_01/{mission_id}")
            local_path = os.path.join(out_root, mission_id, rel)
            url = endpoint.rstrip('/') + f"/swift/v1/{bucket}/" + remote_name
            try:
                if not os.path.exists(local_path):
                    print(f"  Downloading annotation file: {url}")
                    _stream_download(url, local_path)
            except Exception as e:
                print(f"  Warning: failed to download {remote_name}: {e}")

        # Download orthomosaic if requested
        if download_orthomosaics:
            ortho_http = endpoint.rstrip('/') + f"/ofo-public/drone/missions_01/{mission_id}/processed_02/full/orthomosaic.tif"
            local_ortho = os.path.join(out_root, mission_id, 'processed_02', 'full', 'orthomosaic.tif')
            if not os.path.exists(local_ortho):
                print(f"  Downloading orthomosaic: {ortho_http}")
                _stream_download(ortho_http, local_ortho)

    return mission_ids


def _collect_missions(ofo_root: str, mission_ids: Optional[List[str]]) -> List[str]:
    if mission_ids is not None and len(mission_ids) > 0:
        return [os.path.join(ofo_root, mid) for mid in mission_ids]
    # otherwise list directories under root
    return [
        os.path.join(ofo_root, d) for d in os.listdir(ofo_root)
        if os.path.isdir(os.path.join(ofo_root, d))
    ]


def process_ofo_annotations(
    data_dir: str,
    ofo_root: str,
    mission_ids: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """
    Process OFO annotations and save treetops as points and crowns as boxes.

    Args:
        data_dir: Output directory (will create 'points' and 'boxes' subdirs).
        ofo_root: Local path to downloaded OFO mission annotations.
        mission_ids: Optional list of mission_id to process; default all.
        
    Returns:
        Tuple of (points_output_path, boxes_output_path)
    """
    print("=== PROCESSING OFO ANNOTATIONS ===")
    
    points_dir = os.path.join(data_dir, 'points')
    boxes_dir = os.path.join(data_dir, 'boxes')
    ensure_dir(points_dir)
    ensure_dir(boxes_dir)

    missions = _collect_missions(ofo_root, mission_ids)

    treetops_records = []
    crowns_records = []

    for mission_path in missions:
        mission_id = os.path.basename(mission_path.rstrip(os.sep))

        # Look for ITD annotation files
        itd_dir = os.path.join(mission_path, 'processed_02', 'itd_0001')
        
        if not os.path.isdir(itd_dir):
            print(f"No ITD directory found for mission {mission_id}, skipping")
            continue

        # Find treetops file
        treetops_candidates = [
            os.path.join(itd_dir, 'treetops.gpkg'),
            os.path.join(itd_dir, 'treetops.geojson'),
        ]
        # Fallback: any file containing 'treetop'
        if not any(os.path.exists(p) for p in treetops_candidates):
            for ext in ('*.gpkg', '*.geojson'):
                for p in glob.glob(os.path.join(itd_dir, ext)):
                    base = os.path.basename(p).lower()
                    if 'treetop' in base or 'tree_top' in base:
                        treetops_candidates.append(p)

        treetops_file = None
        for p in treetops_candidates:
            if os.path.exists(p):
                treetops_file = p
                break

        # Find crowns silva file
        crowns_silva_candidates = [
            os.path.join(itd_dir, 'crowns-silva.gpkg'),
            os.path.join(itd_dir, 'crowns_silva.gpkg'),
        ]
        # Fallback: any file containing 'crowns' and 'silva'
        if not any(os.path.exists(p) for p in crowns_silva_candidates):
            for ext in ('*.gpkg', '*.geojson'):
                for p in glob.glob(os.path.join(itd_dir, ext)):
                    base = os.path.basename(p).lower()
                    if 'crown' in base and 'silva' in base:
                        crowns_silva_candidates.append(p)

        crowns_silva_file = None
        for p in crowns_silva_candidates:
            if os.path.exists(p):
                crowns_silva_file = p
                break

        # Process treetops (points)
        if treetops_file and os.path.exists(treetops_file):
            print(f"Processing treetops for mission {mission_id}: {treetops_file}")
            try:
                gdf = gpd.read_file(treetops_file)
                if len(gdf) > 0:
                    # Extract point coordinates
                    points_df = pd.DataFrame({
                        'mission_id': mission_id,
                        'x': gdf.geometry.x,
                        'y': gdf.geometry.y,
                        'source': 'Young et al. 2025 unsupervised',
                        'split': 'train',
                        'geometry': gdf.geometry.apply(lambda g: g.wkt)
                    })
                    treetops_records.append(points_df)
                    print(f"  Found {len(gdf)} treetops")
            except Exception as e:
                print(f"  Warning: failed to process treetops file {treetops_file}: {e}")
        else:
            print(f"No treetops file found for mission {mission_id}")

        # Process crowns silva (boxes)
        if crowns_silva_file and os.path.exists(crowns_silva_file):
            print(f"Processing crowns silva for mission {mission_id}: {crowns_silva_file}")
            try:
                gdf = gpd.read_file(crowns_silva_file)
                if len(gdf) > 0:
                    # Extract bounding boxes
                    bounds = gdf.bounds
                    boxes_df = pd.DataFrame({
                        'mission_id': mission_id,
                        'xmin': bounds['minx'],
                        'ymin': bounds['miny'],
                        'xmax': bounds['maxx'],
                        'ymax': bounds['maxy'],
                        'source': 'Young et al. 2025 unsupervised',
                        'geometry': gdf.geometry.apply(lambda g: g.wkt)
                    })
                    crowns_records.append(boxes_df)
                    print(f"  Found {len(gdf)} crown polygons")
            except Exception as e:
                print(f"  Warning: failed to process crowns silva file {crowns_silva_file}: {e}")
        else:
            print(f"No crowns silva file found for mission {mission_id}")

    # Save treetops as points
    points_output = None
    if treetops_records:
        treetops_df = pd.concat(treetops_records, ignore_index=True)
        points_output = os.path.join(points_dir, 'TreePoints_OFO.csv')
        treetops_df.to_csv(points_output, index=False)
        print(f"Saved {len(treetops_df)} treetops to {points_output}")

    # Save crowns as boxes
    boxes_output = None
    if crowns_records:
        crowns_df = pd.concat(crowns_records, ignore_index=True)
        boxes_output = os.path.join(boxes_dir, 'TreeBoxes_OFO.csv')
        crowns_df.to_csv(boxes_output, index=False)
        print(f"Saved {len(crowns_df)} crown boxes to {boxes_output}")

    return points_output, boxes_output


def tile_ofo_orthomosaics(
    data_dir: str,
    ofo_root: str,
    patch_size: int = 800,
    allow_empty: bool = False,
    mission_ids: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Tile OFO orthomosaics and create point/box annotations for tiled images.

    Args:
        data_dir: MillionTrees dataset directory (contains 'images').
        ofo_root: Local path to downloaded OFO missions.
        patch_size: Patch tile size in pixels.
        allow_empty: If True, include empty patches with no points/boxes.
        mission_ids: Optional list of mission IDs to process.
        
    Returns:
        Tuple of (points_csv_path, boxes_csv_path)
    """
    print("\n=== TILING OFO ORTHOMOSAICS ===")
    
    images_dir = os.path.join(data_dir, 'images')
    ensure_dir(images_dir)
    out_dir = os.path.join(data_dir, 'unsupervised')
    ensure_dir(out_dir)

    missions = _collect_missions(ofo_root, mission_ids)
    points_records = []
    boxes_records = []

    for mission_path in missions:
        mission_id = os.path.basename(mission_path.rstrip(os.sep))
        print(f"Tiling mission {mission_id}")

        proc_dirs = glob.glob(os.path.join(mission_path, 'processed_*'))
        if not proc_dirs:
            print(f"No processed_* folder found for mission {mission_id}, skipping")
            continue
        
        # Find directory with orthomosaic
        proc_dir = None
        for candidate_dir in proc_dirs:
            test_orthomosaic = os.path.join(candidate_dir, 'full', 'orthomosaic.tif')
            if os.path.exists(test_orthomosaic):
                proc_dir = candidate_dir
                break
        
        if not proc_dir:
            proc_dir = proc_dirs[0]
        
        orthomosaic = os.path.join(proc_dir, 'full', 'orthomosaic.tif')
        itd_dir = os.path.join(proc_dir, 'itd_0001')
        
        crowns_file = os.path.join(itd_dir, mission_id + '_crowns-silva.gpkg')
        treetops_file = os.path.join(itd_dir, mission_id + '_treetops.gpkg')

        # Process points if treetops exist
        gdf = gpd.read_file(treetops_file)
        # Tile annotations  
        tiled_df = preprocess.split_raster(
            gdf,
            path_to_raster=orthomosaic,
            save_dir=images_dir,
            patch_size=patch_size,
            patch_overlap=0,
            allow_empty=allow_empty
        )
            
        tiled_df['source'] = 'Young et al. 2025 unsupervised'
        tiled_df = tiled_df.rename(columns={'image_path': 'filename'})
        points_records.append(tiled_df)
        print(f"  Created {len(tiled_df)} point annotations for {mission_id}")

        gdf = gpd.read_file(crowns_file)
        # Tile annotations
        tiled_df = preprocess.split_raster(
            gdf,
            path_to_raster=orthomosaic,
            base_dir=images_dir, 
            patch_size=patch_size,
            patch_overlap=0,
            allow_empty=allow_empty
        )
        
        tiled_df['source'] = 'Young et al. 2025 unsupervised'
        tiled_df = tiled_df.rename(columns={'image_path': 'filename'})
        boxes_records.append(tiled_df)
        print(f"  Created {len(tiled_df)} box annotations for {mission_id}")
    
    # Save points annotations
    points_df = pd.concat(points_records, ignore_index=True)
    points_output = os.path.join(out_dir, "TreePoints_OFO_unsupervised.csv")
    points_df.to_csv(points_output, index=False)
    print(f"Wrote {len(points_df)} OFO unsupervised points to {points_output}")

    # Save boxes annotations  
    boxes_df = pd.concat(boxes_records, ignore_index=True)
    boxes_output = os.path.join(out_dir, "TreeBoxes_OFO_unsupervised.csv")
    boxes_df.to_csv(boxes_output, index=False)
    print(f"Wrote {len(boxes_df)} OFO unsupervised boxes to {boxes_output}")

    return points_output, boxes_output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process OFO data: download annotations and images, tile orthomosaics, and create datasets."
    )
    
    # Main arguments
    parser.add_argument(
        '--data_dir',
        required=True,
        help='Path to dataset directory, e.g., /path/to/TreeBoxes_v0.5'
    )
    parser.add_argument(
        '--ofo_root',
        default='ofo_downloads',
        help='Local path where OFO missions will be downloaded (default: ofo_downloads)'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Output directory for processed annotations'
    )
    
    # Mission selection
    parser.add_argument(
        '--mission_ids',
        type=str,
        default=None,
        help='Comma-separated mission IDs to process (default: all available)'
    )
    parser.add_argument(
        '--num_missions',
        type=int,
        default=None,
        help='Maximum number of missions to process (default: 5)'
    )
    
    # Processing options
    parser.add_argument(
        '--patch_size',
        type=int,
        default=800,
        help='Patch size for tiling (pixels). Default 800'
    )
    parser.add_argument(
        '--allow_empty',
        action='store_true',
        help='Include empty patches during tiling'
    )
    
    # Control flags
    parser.add_argument(
        '--skip_download',
        action='store_true',
        help='Skip download step and use existing data'
    )
    parser.add_argument(
        '--skip_annotations',
        action='store_true',
        help='Skip annotation processing step'
    )
    parser.add_argument(
        '--skip_tiling',
        action='store_true',
        help='Skip orthomosaic tiling step'
    )
    parser.add_argument(
        '--download_orthomosaics',
        action='store_true',
        default=True,
        help='Download orthomosaics for tiling (default: True)'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Parse mission IDs
    mission_ids = None
    if args.mission_ids:
        mission_ids = [s.strip() for s in args.mission_ids.split(',') if s.strip()]
    
    ensure_dir(args.ofo_root)
    ensure_dir(args.output_dir)
    
    # Step 1: Download OFO data (unless skipped)
    if not args.skip_download:
        print("=== DOWNLOADING OFO DATA ===")
        downloaded_missions = download_ofo_data(
            out_root=args.ofo_root,
            mission_ids=mission_ids,
            num_missions=args.num_missions,
            download_orthomosaics=args.download_orthomosaics
        )
        print(f"Downloaded data for {len(downloaded_missions)} missions")
        
        # Use downloaded missions for processing if no specific mission_ids were provided
        if mission_ids is None:
            mission_ids = downloaded_missions
    else:
        print("Skipping download step - using existing data")
    
    # Step 2: Process annotations (unless skipped)
    if not args.skip_annotations:
        points_output, boxes_output = process_ofo_annotations(
            data_dir=args.output_dir,
            ofo_root=args.ofo_root,
            mission_ids=mission_ids
        )
        if points_output:
            print(f"Created points annotations: {points_output}")
        if boxes_output:
            print(f"Created boxes annotations: {boxes_output}")
    else:
        print("Skipping annotation processing step")
    
    # Step 3: Tile orthomosaics (unless skipped)
    if not args.skip_tiling and args.download_orthomosaics:
        tiled_output = tile_ofo_orthomosaics(
            data_dir=args.data_dir,
            ofo_root=args.ofo_root,
            patch_size=args.patch_size,
            allow_empty=args.allow_empty,
            mission_ids=mission_ids
        )
        if tiled_output:
            print(f"Created tiled annotations: {tiled_output}")
    else:
        if args.skip_tiling:
            print("Skipping tiling step")
        else:
            print("Skipping tiling step - orthomosaics not downloaded")
    
    print("\n=== OFO PROCESSING COMPLETE ===")


if __name__ == '__main__':
    main()