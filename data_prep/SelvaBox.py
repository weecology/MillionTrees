import pandas as pd
import numpy as np
import os
import requests
from tqdm import tqdm
from PIL import Image
import json
from deepforest.utilities import read_file

def download_selvabox(force_download=False):
    """Download and process the SelvaBox dataset from HuggingFace
    
    Args:
        force_download (bool): If True, re-download parquet files even if cached
    """
    
    # Create output directory (using standard MillionTrees path structure)
    output_dir = "/orange/ewhite/DeepForest/SelvaBox"
    images_dir = os.path.join(output_dir, "images")
    cache_dir = os.path.join(output_dir, "cache")
    annotations_csv = os.path.join(output_dir, "annotations.csv")
    
    # Check if dataset already exists locally
    if not force_download and os.path.exists(annotations_csv) and os.path.exists(images_dir):
        print(f"Dataset already exists at {output_dir}")
        print(f"Found {len(os.listdir(images_dir))} images and annotations at {annotations_csv}")
        print("Use force_download=True to re-download the dataset")
        return annotations_csv
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    print("Downloading SelvaBox dataset from HuggingFace...")
    
    # Get the parquet file URLs from the HuggingFace dataset viewer API
    dataset_name = "CanopyRS/SelvaBox"
    api_url = f"https://datasets-server.huggingface.co/parquet?dataset={dataset_name}"
    
    response = requests.get(api_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch dataset info: {response.status_code}")
    
    parquet_info = response.json()
    
    # Process each split (train, validation, test)
    all_annotations = []
    
    for file_info in parquet_info['parquet_files']:
        split = file_info['split'] 
        parquet_url = file_info['url']
        
        # Cache parquet files locally
        parquet_filename = os.path.basename(parquet_url.split('?')[0])  # Remove query params
        cached_parquet_path = os.path.join(cache_dir, f"{split}_{parquet_filename}")
        
        # Download parquet file if not cached or if force_download is True
        if force_download or not os.path.exists(cached_parquet_path):
            print(f"Downloading {split} split parquet file...")
            parquet_response = requests.get(parquet_url, stream=True)
            parquet_response.raise_for_status()
            
            with open(cached_parquet_path, 'wb') as f:
                for chunk in parquet_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Cached {split} split to {cached_parquet_path}")
        else:
            print(f"Using cached {split} split from {cached_parquet_path}")
        
        # Read from cached file
        df = pd.read_parquet(cached_parquet_path)
        
        print(f"Loaded {len(df)} rows from {split} split")
        
        # Process each row
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
            try:
                # Extract image data
                if 'image' in row and row['image'] is not None:
                    image_data = row['image']
                    
                    # Get image filename from tile_name or create one
                    if 'tile_name' in row and row['tile_name'] is not None:
                        tile_name = row['tile_name']
                        # Use the original tile name as filename
                        image_filename = tile_name.replace('.tif', '.png')
                    else:
                        image_filename = f"{split}_{idx}.png"
                    
                    image_path = os.path.join(images_dir, image_filename)
                    
                    # Skip saving if image already exists (unless force_download is True)
                    if not force_download and os.path.exists(image_path):
                        # Image exists, skip saving but continue to annotations
                        pass
                    # Save image from bytes
                    elif isinstance(image_data, dict) and 'bytes' in image_data:
                        try:
                            image_bytes = image_data['bytes']
                            
                            # Save as temporary tif first, then convert to PNG
                            temp_tif_path = image_path.replace('.png', '_temp.tif')
                            with open(temp_tif_path, 'wb') as f:
                                f.write(image_bytes)
                            
                            # Convert to PNG and verify dimensions
                            with Image.open(temp_tif_path) as img:
                                # Convert to RGB if necessary
                                if img.mode != 'RGB':
                                    img = img.convert('RGB')
                                img.save(image_path, 'PNG')
                            
                            # Remove temporary tif file
                            os.remove(temp_tif_path)
                            
                        except Exception as e:
                            print(f"Failed to save or convert image for row {idx}: {e}")
                            continue
                    else:
                        print(f"Unexpected image data format for row {idx}")
                        continue
                
                # Extract annotations
                if 'annotations' in row and row['annotations'] is not None:
                    annotations = row['annotations']
                    
                    # Handle the bbox annotations
                    if isinstance(annotations, dict) and 'bbox' in annotations:
                        bboxes = annotations['bbox']
                        
                        # Process each bounding box
                        for i, bbox in enumerate(bboxes):
                            if len(bbox) == 4:
                                # SelvaBox format appears to be [x, y, width, height]
                                x, y, width, height = bbox
                                xmin = x
                                ymin = y
                                xmax = x + width
                                ymax = y + height
                                
                                # Validate bounding box coordinates
                                if xmin < xmax and ymin < ymax and xmin >= 0 and ymin >= 0:
                                    annotation_row = {
                                        'image_path': image_path,
                                        'xmin': float(xmin),
                                        'ymin': float(ymin), 
                                        'xmax': float(xmax),
                                        'ymax': float(ymax),
                                        'label': 'tree',  # All annotations are trees
                                        'source': 'SelvaBox'
                                    }
                                    
                                    all_annotations.append(annotation_row)
                                else:
                                    print(f"Invalid bounding box for row {idx}, bbox {i}: {bbox}")
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
    
    # Convert to DataFrame
    annotations_df = pd.DataFrame(all_annotations)
    annotations_df = read_file(annotations_df)
    # full path
    annotations_df["image_path"] = annotations_df["image_path"].apply(lambda x: os.path.join(images_dir, x))
    
    # Infer existing split from filename patterns (e.g., train_*, val_*, validation_*, test_*)
    def infer_split_from_filename(p: str):
        name = os.path.basename(p).lower()
        if name.startswith("train_") or "_train_" in name or name.endswith("_train.png"):
            return "train"
        if name.startswith("test_") or "_test_" in name or name.endswith("_test.png"):
            return "test"
        if name.startswith("validation_") or "_validation_" in name or name.endswith("_validation.png"):
            return "validation"
        if name.startswith("val_") or "_val_" in name or name.endswith("_val.png"):
            return "validation"
        return None
    
    annotations_df["existing_split"] = annotations_df["image_path"].apply(infer_split_from_filename)
    
    if len(annotations_df) == 0:
        print("No annotations found!")
        return
    
    print(f"Processed {len(annotations_df)} annotations")
    print(f"Unique images: {annotations_df['image_path'].nunique()}")
    print(f"Annotation bounds - xmin: [{annotations_df['xmin'].min():.2f}, {annotations_df['xmin'].max():.2f}]")
    print(f"Annotation bounds - ymin: [{annotations_df['ymin'].min():.2f}, {annotations_df['ymin'].max():.2f}]") 
    print(f"Annotation bounds - xmax: [{annotations_df['xmax'].min():.2f}, {annotations_df['xmax'].max():.2f}]")
    print(f"Annotation bounds - ymax: [{annotations_df['ymax'].min():.2f}, {annotations_df['ymax'].max():.2f}]")
    
    # Save annotations
    annotations_df.to_csv(annotations_csv, index=False)
    print(f"Annotations saved to {annotations_csv}")
    
    # Show sample of the data
    print("\nSample annotations:")
    print(annotations_df.head())
    
    return annotations_csv

if __name__ == "__main__":
    download_selvabox()