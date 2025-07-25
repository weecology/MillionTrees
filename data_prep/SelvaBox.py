import pandas as pd
import numpy as np
import os
import requests
from tqdm import tqdm
from PIL import Image
import json
import ast
from deepforest.utilities import read_file

def download_selvabox():
    """Download and process the SelvaBox dataset from HuggingFace"""
    
    # Create output directory (using standard MillionTrees path structure)
    output_dir = "/orange/ewhite/DeepForest/SelvaBox"
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
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
        
        print(f"Processing {split} split from {parquet_url}")
        
        # Read the parquet file directly from HuggingFace
        df = pd.read_parquet(parquet_url)
        
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
                    
                    # Save image from bytes
                    if isinstance(image_data, dict) and 'bytes' in image_data:
                        try:
                            image_bytes = image_data['bytes']
                            
                            # Save as temporary tif first, then convert to PNG
                            temp_tif_path = image_path.replace('.png', '_temp.tif')
                            with open(temp_tif_path, 'wb') as f:
                                f.write(image_bytes)
                            
                            # Convert to PNG and verify dimensions
                            with Image.open(temp_tif_path) as img:
                                img_width, img_height = img.size
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
    output_csv = os.path.join(output_dir, "annotations.csv")
    annotations_df.to_csv(output_csv, index=False)
    print(f"Annotations saved to {output_csv}")
    
    # Show sample of the data
    print("\nSample annotations:")
    print(annotations_df.head())
    
    return output_csv

if __name__ == "__main__":
    download_selvabox()