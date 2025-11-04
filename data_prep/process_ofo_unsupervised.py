#!/usr/bin/env python3
"""
Combined script to download OFO annotations and images, tile them, and prepare datasets.
Downloads OFO mission data, processes annotations, and creates tiled datasets.
"""

import os
import argparse
import glob
from typing import List, Optional, Tuple
import rasterio as rio

import numpy as np
import pandas as pd
import geopandas as gpd
from deepforest import preprocess
from deepforest.utilities import read_file

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _require_requests():
    import requests
    return requests

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
        lines = resp.text.strip().split('\n')
        items = []
        
        if delimiter == '/':
            mission_ids = set()
            for line in lines:
                if line.strip() and line.startswith(prefix):
                    parts = line.replace(prefix, '').split('/')
                    if len(parts) > 0 and parts[0]:
                        mission_ids.add(parts[0])
            
            for mission_id in sorted(mission_ids):
                items.append({'subdir': f"{prefix}{mission_id}/"})
        else:
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

def list_missions(endpoint: str = 'https://js2.jetstream-cloud.org:8001',
                  bucket: str = 'ofo-public',
                  limit: Optional[int] = None) -> List[str]:
    """List available OFO mission IDs."""
    prefix = 'drone/missions_01/'
    entries = _swift_list(endpoint, bucket, prefix=prefix, delimiter='/')
    mission_ids = []
    
    for item in entries:
        subdir = item.get('subdir')
        if subdir and subdir.startswith(prefix):
            mission_id = os.path.basename(subdir.rstrip('/'))
            if mission_id:
                mission_ids.append(mission_id)
    
    mission_ids.sort()
    return mission_ids[:limit] if limit else mission_ids

def download_mission_data(mission_id: str, out_root: str,
                         endpoint: str = 'https://js2.jetstream-cloud.org:8001',
                         bucket: str = 'ofo-public') -> bool:
    """Download single mission's annotations and orthomosaic. Returns True if orthomosaic exists."""
    print(f"Downloading mission {mission_id}")
    
    # Download annotations
    itd_prefix = f"drone/missions_01/{mission_id}/processed_02/itd_0001/"
    items = _swift_list(endpoint, bucket, prefix=itd_prefix, delimiter='')
    
    for obj in items:
        name = obj.get('name')
        if not name or not name.startswith(f"drone/missions_01/{mission_id}/"):
            continue
        
        ext = os.path.splitext(name)[1].lower()
        if ext in {'.gpkg', '.geojson', '.json', '.shp', '.shx', '.dbf', '.prj', '.cpg', '.xml', '.csv', '.tif'}:
            rel = os.path.relpath(name, start=f"drone/missions_01/{mission_id}")
            local_path = os.path.join(out_root, mission_id, rel)
            
            if not os.path.exists(local_path):
                url = endpoint.rstrip('/') + f"/swift/v1/{bucket}/" + name
                _stream_download(url, local_path)
    
    # Download orthomosaic
    ortho_candidates = [
        f"{mission_id}_ortho-dsm-ptcloud.tif",
        f"{mission_id}_ortho-dsm-mesh.tif"
    ]
    
    for ortho_filename in ortho_candidates:
        ortho_url = f"{endpoint.rstrip('/')}/swift/v1/{bucket}/drone/missions_01/{mission_id}/processed_02/full/{ortho_filename}"
        local_ortho = os.path.join(out_root, mission_id, 'processed_02', 'full', 'orthomosaic.tif')
        
        if os.path.exists(local_ortho):
            return True
        
        requests = _require_requests()
        resp = requests.head(ortho_url, timeout=30)
        if resp.status_code == 200:
            print(f"  Downloading orthomosaic: {ortho_url}")
            _stream_download(ortho_url, local_ortho)
            return True
    
    print(f"  Warning: No orthomosaic found for {mission_id}")
    return False

def process_mission_annotations(mission_path: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Process single mission's annotations into points and boxes DataFrames."""
    mission_id = os.path.basename(mission_path.rstrip(os.sep))
    itd_dir = os.path.join(mission_path, 'processed_02', 'itd_0001')
    
    if not os.path.isdir(itd_dir):
        return None, None
    
    points_df = None
    boxes_df = None
    
    # Process treetops (points)
    treetops_file = os.path.join(itd_dir, 'treetops.gpkg')
    if os.path.exists(treetops_file):
        gdf = gpd.read_file(treetops_file)
        points_df = pd.DataFrame({
            'mission_id': mission_id,
            'x': gdf.geometry.x,
            'y': gdf.geometry.y,
            'source': 'Young et al. 2025',
            'split': 'train',
            'geometry': gdf.geometry.apply(lambda g: g.wkt)
        })
    
    # Process crowns (boxes)
    crowns_file = os.path.join(itd_dir, 'crowns-silva.gpkg')
    if os.path.exists(crowns_file):
        gdf = gpd.read_file(crowns_file)
        bounds = gdf.bounds
        boxes_df = pd.DataFrame({
            'mission_id': mission_id,
            'xmin': bounds['minx'],
            'ymin': bounds['miny'],
            'xmax': bounds['maxx'],
            'ymax': bounds['maxy'],
            'source': 'Young et al. 2025',
            'geometry': gdf.geometry.apply(lambda g: g.wkt)
        })
    
    return points_df, boxes_df

def tile_mission_orthomosaic(mission_path: str, images_dir: str, patch_size: int = 800) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Tile single mission's orthomosaic and return tiled annotations."""
    mission_id = os.path.basename(mission_path.rstrip(os.sep))
    
    proc_dirs = glob.glob(os.path.join(mission_path, 'processed_*'))
    if not proc_dirs:
        return None, None
    
    # Find orthomosaic
    orthomosaic = None
    for proc_dir in proc_dirs:
        test_path = os.path.join(proc_dir, 'full', 'orthomosaic.tif')
        if os.path.exists(test_path):
            orthomosaic = test_path
            break
    
    if not orthomosaic:
        return None, None
    
    itd_dir = os.path.join(os.path.dirname(os.path.dirname(orthomosaic)), 'itd_0001')
    
    # Prepare 3-band orthomosaic for tiling
    with rio.open(orthomosaic) as src:
        arr = src.read()
        profile = src.profile.copy()
    
    arr[np.isnan(arr)] = 0
    arr3 = arr[:3, :, :].astype(np.uint8)
    
    profile.update(count=3, dtype=rio.uint8, nodata=0)
    
    out_ortho = os.path.join(os.path.dirname(orthomosaic), f"{mission_id}_ortho.tif")
    with rio.open(out_ortho, 'w', **profile) as dst:
        dst.write(arr3)
    
    points_tiled = None
    boxes_tiled = None
    
    # Tile points
    treetops_file = os.path.join(itd_dir,f'{mission_id}_treetops.gpkg')
    gdf = gpd.read_file(treetops_file)
    gdf["image_path"] = os.path.basename(out_ortho)
    gdf["label"] = "Tree"
    gdf = read_file(gdf, root_dir=os.path.dirname(out_ortho))
    
    points_tiled = preprocess.split_raster(
        gdf,
        path_to_raster=out_ortho,
        root_dir=os.path.dirname(out_ortho),
        save_dir=images_dir,
        patch_size=patch_size,
        patch_overlap=0,
        allow_empty=False
    )
    
    if len(points_tiled) > 0:
        points_tiled['source'] = 'Young et al. 2025 unsupervised'
        points_tiled['split'] = 'train'
        points_tiled = points_tiled.rename(columns={'image_path': 'filename'})

    # Tile boxes
    crowns_file = os.path.join(itd_dir, f'{mission_id}_crowns-silva.gpkg')
    gdf = gpd.read_file(crowns_file)
    gdf["image_path"] = os.path.basename(out_ortho)
    gdf["label"] = "Tree"
    gdf = read_file(gdf, root_dir=os.path.dirname(out_ortho))
    
    boxes_tiled = preprocess.split_raster(
        gdf,
        path_to_raster=out_ortho,
        save_dir=images_dir,
        patch_size=patch_size,
        patch_overlap=0,
        allow_empty=False
    )
    
    boxes_tiled['source'] = 'Young et al. 2025 unsupervised'
    boxes_tiled = boxes_tiled.rename(columns={'image_path': 'filename'})

    return points_tiled, boxes_tiled

def parse_args():
    parser = argparse.ArgumentParser(description="Process OFO data")
    parser.add_argument('--data_dir', required=True, help='Dataset directory')
    parser.add_argument('--ofo_root', default='ofo_downloads', help='OFO download directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--mission_ids', help='Comma-separated mission IDs')
    parser.add_argument('--num_missions', type=int, default=3, help='Max missions to process')
    parser.add_argument('--patch_size', type=int, default=800, help='Tile size in pixels')
    return parser.parse_args()

def main():
    args = parse_args()
    
    mission_ids = None
    if args.mission_ids:
        mission_ids = [s.strip() for s in args.mission_ids.split(',') if s.strip()]
    
    ensure_dir(args.ofo_root)
    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.data_dir, 'images'))
    
    # Get missions to process
    if not mission_ids:
        mission_ids = list_missions(limit=args.num_missions)
    
    print(f"Processing {len(mission_ids)} missions")
    
    # Download all mission data
    valid_missions = []
    for mission_id in mission_ids:
        has_orthomosaic = download_mission_data(mission_id, args.ofo_root)
        if has_orthomosaic:
            valid_missions.append(os.path.join(args.ofo_root, mission_id))
    
    print(f"Found orthomosaics for {len(valid_missions)} missions")
    
    # Process annotations
    points_records = []
    boxes_records = []
    
    for mission_path in valid_missions:
        points_df, boxes_df = process_mission_annotations(mission_path)
        if points_df is not None:
            points_records.append(points_df)
        if boxes_df is not None:
            boxes_records.append(boxes_df)
    
    # Save annotation CSVs
    points_dir = os.path.join(args.output_dir, 'points')
    boxes_dir = os.path.join(args.output_dir, 'boxes')
    ensure_dir(points_dir)
    ensure_dir(boxes_dir)
    
    if points_records:
        points_df = pd.concat(points_records, ignore_index=True)
        points_output = os.path.join(points_dir, 'TreePoints_OFO.csv')
        points_df.to_csv(points_output, index=False)
        print(f"Saved {len(points_df)} points to {points_output}")
    
    if boxes_records:
        boxes_df = pd.concat(boxes_records, ignore_index=True)
        boxes_output = os.path.join(boxes_dir, 'TreeBoxes_OFO.csv')
        boxes_df.to_csv(boxes_output, index=False)
        print(f"Saved {len(boxes_df)} boxes to {boxes_output}")
    
    # Tile orthomosaics
    images_dir = os.path.join(args.data_dir, 'images')
    tiled_points = []
    tiled_boxes = []
    
    for mission_path in valid_missions:
        points_tiled, boxes_tiled = tile_mission_orthomosaic(mission_path, images_dir, args.patch_size)
        tiled_points.append(points_tiled)
        tiled_boxes.append(boxes_tiled)
    
    tiled_boxes = pd.concat(tiled_boxes)
    
    tiled_boxes["image_path"] = tiled_boxes["filename"].apply(lambda x: os.path.join(images_dir,x))
    tiled_boxes = read_file(tiled_boxes, root_dir=images_dir)

    tiled_boxes["image_path"] = tiled_boxes["image_path"].apply(lambda x: os.path.join(images_dir,x))
    tiled_boxes = tiled_boxes.drop(columns=['filename'])
    tiled_boxes.to_csv(os.path.join(images_dir, 'TreeBoxes_OFO_unsupervised.csv'), index=False)

    tiled_points = pd.concat(tiled_points)

    tiled_points["image_path"] = tiled_points["filename"].apply(lambda x: os.path.join(images_dir,x))
    tiled_points = read_file(tiled_points, root_dir=images_dir)

    tiled_points["image_path"] = tiled_points["image_path"].apply(lambda x: os.path.join(images_dir,x))
    
    # remove filename column to avoid confusion
    tiled_points = tiled_points.drop(columns=['filename'])
    tiled_points.to_csv(os.path.join(images_dir, 'TreePoints_OFO_unsupervised.csv'), index=False)

if __name__ == '__main__':
    main()