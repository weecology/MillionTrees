import pandas as pd
import numpy as np
import os
import requests
from tqdm import tqdm
from PIL import Image
import json
from deepforest.utilities import read_file
from deepforest.visualize import plot_results

def download_oam_tcd(force_download=False):
    """Download and process the OAM-TCD dataset from HuggingFace
    
    Args:
        force_download (bool): If True, re-download the dataset even if it exists locally
    """
    
    # Create output directory (using standard MillionTrees path structure)
    output_dir = "/orange/ewhite/DeepForest/OAM_TCD"
    images_dir = os.path.join(output_dir, "images")
    annotations_csv = os.path.join(output_dir, "annotations.csv")
    
    # Check if dataset already exists locally
    if not force_download and os.path.exists(annotations_csv) and os.path.exists(images_dir):
        print(f"Dataset already exists at {output_dir}")
        print(f"Found {len(os.listdir(images_dir))} images and annotations at {annotations_csv}")
        print("Use force_download=True to re-download the dataset")
        return annotations_csv
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    print("Downloading OAM-TCD dataset from HuggingFace...")
    
    # Get the parquet file URLs from the HuggingFace dataset viewer API
    dataset_name = "restor/tcd"
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
            # Extract image data
                if 'image' in row and row['image'] is not None:
                    image_data = row['image']
                    
                    image_filename = f"{split}_{row['image_id']}.png"
                    image_path = os.path.join(images_dir, image_filename)
                    
                    #Skip if image already exists (unless force_download is True)
                    if not force_download and os.path.exists(image_path):
                       continue
                    
                    # Save image
                    if isinstance(image_data, dict) and 'bytes' in image_data:
                        image_bytes = image_data['bytes']
                            
                        # Save as temporary file first, then convert to PNG
                        temp_path = image_path.replace('.png', '_temp')
                        with open(temp_path, 'wb') as f:
                            f.write(image_bytes)
                        
                        # Convert to PNG and verify dimensions
                        with Image.open(temp_path) as img:
                            img_width, img_height = img.size
                            # Convert to RGB if necessary
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            img.save(image_path, 'PNG')
                            
                        # Remove temporary file
                        os.remove(temp_path)
                    elif hasattr(image_data, 'save'):
                        # PIL Image object
                        img_width, img_height = image_data.size
                        if image_data.mode != 'RGB':
                            image_data = image_data.convert('RGB')
                        image_data.save(image_path, 'PNG')
                    else:
                        print(f"Unexpected image data format for row {idx}")
                        continue
                
                # Extract annotations
                if 'coco_annotations' in row and row['coco_annotations'] is not None:
                    # Parse JSON string to get coco_annotations
                    coco_annotations = json.loads(row['coco_annotations'])
                    # Process each annotation in the COCO format
                    for annotation in coco_annotations:
                        # Skip if annotation is not a dict
                        if not isinstance(annotation, dict):
                            print(f"Unexpected annotation format: {annotation}")
                            continue
                            
                        # Get bbox and category
                        bbox = annotation.get('bbox', [])
                        category_id = annotation.get('category_id', 0)
                        
                        # Validate bbox format
                        if len(bbox) != 4:
                            print(f"Invalid bbox format: {bbox}")
                            continue
                        
                        # COCO format is typically [x, y, width, height]
                        x, y, width, height = bbox
                        xmin = x
                        ymin = y
                        xmax = x + width
                        ymax = y + height
                        
                        # Filter out canopy label annotations
                        if category_id in [1]: 
                            continue
                    
                        # Validate bounding box coordinates
                        if xmin >= xmax or ymin >= ymax or xmin < 0 or ymin < 0:
                            print(f"Invalid bounding box coordinates: {bbox}")
                            continue
                        
                        # Ensure coordinates are within image bounds
                        xmin = max(0, min(xmin, img_width))
                        ymin = max(0, min(ymin, img_height))
                        xmax = max(xmin, min(xmax, img_width))
                        ymax = max(ymin, min(ymax, img_height))
                        
                        annotation_row = {
                            'image_path': image_path,
                            'xmin': float(xmin),
                            'ymin': float(ymin), 
                            'xmax': float(xmax),
                            'ymax': float(ymax),
                            'label': 'tree',  # All non-canopy annotations are trees
                            'source': 'Veitch-Michaelis et al. 2024'
                        }
                        
                        all_annotations.append(annotation_row)
    
    # Convert to DataFrame
    annotations_df = pd.DataFrame(all_annotations)
    
    if len(annotations_df) == 0:
        print("No annotations found!")
        return
    
    # Use deepforest utilities to process annotations
    annotations_df = read_file(annotations_df)
    
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
    import matplotlib.pyplot as plt
    annotations_csv = download_oam_tcd(force_download=True)

    # Plot the results
    df = pd.read_csv(annotations_csv)
    print(df.head())
    print(df.columns)
    print(df['image_path'].nunique())
    print(df['label'].value_counts())
    print(df['source'].value_counts())
    print(df['xmin'].min(), df['xmin'].max())
    print(df['ymin'].min(), df['ymin'].max())
    print(df['xmax'].min(), df['xmax'].max())
    print(df['ymax'].min(), df['ymax'].max())
    sample_image = df['image_path'].iloc[2000]
    sample_annotations = df[df['image_path'] == sample_image]
    sample_annotations = read_file(sample_annotations)
    sample_annotations.root_dir = "/orange/ewhite/DeepForest/OAM_TCD/images"
    plot_results(sample_annotations)
    plt.savefig("current.png")
