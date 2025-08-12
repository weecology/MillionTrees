#!/usr/bin/env python3
"""
Visualization script for MillionTrees datasets.

This script loops through all images in all datasets, uses plot_results to overlay 
annotations on images, and saves PNG files with the filename as source + image_name.
It also creates miniature and full dataset splits for reviewers.
"""

import pandas as pd
import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import shutil
from deepforest.visualize import plot_results
from deepforest.utilities import read_file
import warnings
warnings.filterwarnings('ignore')

# Dataset paths from count_annotations.py
DATASET_PATHS = {
    'TreeBoxes': [
        "/orange/ewhite/DeepForest/Ryoungseob_2023/train_datasets/images/train.csv",
        "/orange/ewhite/DeepForest/Velasquez_urban_trees/tree_canopies/nueva_carpeta/annotations.csv",
        '/orange/ewhite/DeepForest/individual_urban_tree_crown_detection/annotations.csv',
        '/orange/ewhite/DeepForest/Radogoshi_Sweden/annotations.csv',
        "/orange/ewhite/DeepForest/WRI/WRI-labels-opensource/annotations.csv",
        "/orange/ewhite/DeepForest/Guangzhou2022/annotations.csv",
        "/orange/ewhite/DeepForest/NEON_benchmark/NeonTreeEvaluation_annotations.csv",
        "/orange/ewhite/DeepForest/NEON_benchmark/University_of_Florida.csv",
        '/orange/ewhite/DeepForest/ReForestTree/images/train.csv',
        "/orange/ewhite/DeepForest/Santos2019/annotations.csv",
        "/orange/ewhite/DeepForest/Zenodo_15155081/parsed_annotations.csv",
        "/orange/ewhite/DeepForest/SelvaBox/annotations.csv"
    ],
    'TreePoints': [
        "/orange/ewhite/DeepForest/TreeFormer/all_images/annotations.csv",
        "/orange/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/annotations.csv",
        "/orange/ewhite/MillionTrees/NEON_points/annotations.csv",
        "/orange/ewhite/DeepForest/Tonga/annotations.csv",
        '/orange/ewhite/DeepForest/BohlmanBCI/crops/annotations_points.csv',
        "/orange/ewhite/DeepForest/AutoArborist/downloaded_imagery/AutoArborist_combined_annotations.csv"
    ],
    'TreePolygons': [
        "/orange/ewhite/DeepForest/Jansen_2023/pngs/annotations.csv",
        "/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/annotations/annotations.csv",
        "/orange/ewhite/DeepForest/Cloutier2023/images/annotations.csv",
        "/orange/ewhite/DeepForest/Firoze2023/annotations.csv",
        "/orange/ewhite/DeepForest/Wagner_Australia/annotations.csv",
        "/orange/ewhite/DeepForest/Alejandro_Chile/alejandro/annotations.csv",
        "/orange/ewhite/DeepForest/UrbanLondon/annotations.csv",
        "/orange/ewhite/DeepForest/OliveTrees_spain/Dataset_RGB/annotations.csv",
        "/orange/ewhite/DeepForest/Araujo_2020/annotations.csv",
        "/orange/ewhite/DeepForest/justdiggit-drone/label_sample/annotations.csv",
        "/orange/ewhite/DeepForest/BCI/BCI_50ha_2020_08_01_crownmap_raw/annotations.csv",
        "/orange/ewhite/DeepForest/BCI/BCI_50ha_2022_09_29_crownmap_raw/annotations.csv",
        "/orange/ewhite/DeepForest/Harz_Mountains/ML_TreeDetection_Harz/annotations.csv",
        "/orange/ewhite/DeepForest/SPREAD/annotations.csv",
        "/orange/ewhite/DeepForest/KagglePalm/Palm-Counting-349images/annotations.csv",
        "/orange/ewhite/DeepForest/Kattenborn/uav_newzealand_waititu/crops/annotations.csv",
        "/orange/ewhite/DeepForest/Quebec_Lefebvre/Dataset/Crops/annotations.csv",
        "/orange/ewhite/DeepForest/BohlmanBCI/crops/annotations_crowns.csv",
        "/orange/ewhite/DeepForest/TreeCountSegHeight/extracted_data_2aux_v4_cleaned_centroid_raw 2/annotations.csv",
        "/orange/ewhite/DeepForest/takeshige2025/crops/annotations.csv"
    ]
}

def load_datasets():
    """Load all datasets and combine them with dataset type information."""
    all_data = []
    for dataset_type, paths in DATASET_PATHS.items():
        for path in paths:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    df['dataset_type'] = dataset_type
                    df['csv_path'] = path
                    # Standardize column names
                    if 'image_path' in df.columns:
                        df = df.rename(columns={'image_path': 'filename'})
                    all_data.append(df)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def get_image_root_dir(csv_path):
    """Determine the root directory for images based on CSV path."""
    return os.path.dirname(csv_path)

def visualize_image_annotations(df_subset, output_dir):
    """Visualize annotations for a subset of images and save as PNG."""
    os.makedirs(output_dir, exist_ok=True)
    
    for (source, filename), group in df_subset.groupby(['source', 'filename']):
        try:
            # Get image root directory
            csv_path = group['csv_path'].iloc[0]
            root_dir = get_image_root_dir(csv_path)
            
            # Prepare data for plotting
            plot_data = group.copy()
            plot_data['image_path'] = plot_data['filename']
            plot_data = read_file(plot_data, root_dir=root_dir)
            plot_data.root_dir = root_dir
            
            # Create safe filename
            safe_source = source.replace(" ", "_").replace("/", "_").replace(".", "_")
            safe_filename = os.path.splitext(os.path.basename(filename))[0]
            output_filename = f"{safe_source}_{safe_filename}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # Get image dimensions for polygons
            dataset_type = group['dataset_type'].iloc[0]
            full_image_path = os.path.join(root_dir, filename)
            
            if os.path.exists(full_image_path):
                if dataset_type == "TreePolygons":
                    height, width, _ = cv2.imread(full_image_path).shape
                    plot_results(plot_data, savedir=output_dir, basename=f"{safe_source}_{safe_filename}", 
                               height=height, width=width)
                else:
                    plot_results(plot_data, savedir=output_dir, basename=f"{safe_source}_{safe_filename}")
                    
                print(f"Saved visualization: {output_filename}")
            else:
                print(f"Image not found: {full_image_path}")
                
        except Exception as e:
            print(f"Error visualizing {source} - {filename}: {e}")

def create_mini_dataset(df, images_per_source=10):
    """Create a miniature dataset with specified number of images per source."""
    mini_data = []
    for source in df['source'].unique():
        source_data = df[df['source'] == source]
        unique_images = source_data['filename'].unique()
        
        # Select images with most annotations
        image_counts = source_data.groupby('filename').size()
        top_images = image_counts.nlargest(min(images_per_source, len(unique_images))).index
        
        source_mini = source_data[source_data['filename'].isin(top_images)]
        mini_data.append(source_mini)
    
    return pd.concat(mini_data, ignore_index=True) if mini_data else pd.DataFrame()

def split_for_reviewers(df, output_dir, name_prefix, num_splits=4):
    """Split dataset into equal parts for reviewers."""
    unique_images = df['filename'].unique()
    np.random.shuffle(unique_images)
    
    splits = np.array_split(unique_images, num_splits)
    
    split_info = []
    for i, split_images in enumerate(splits):
        split_data = df[df['filename'].isin(split_images)]
        split_name = f"{name_prefix}_reviewer_{i+1}"
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        # Save split data
        split_data.to_csv(os.path.join(split_dir, "annotations.csv"), index=False)
        
        # Create visualizations for this split
        visualize_image_annotations(split_data, os.path.join(split_dir, "visualizations"))
        
        split_info.append({
            'split_name': split_name,
            'num_images': len(split_images),
            'num_annotations': len(split_data),
            'sources': split_data['source'].unique().tolist()
        })
        
        print(f"Created {split_name}: {len(split_images)} images, {len(split_data)} annotations")
    
    return split_info

def main():
    """Main execution function."""
    print("Starting MillionTrees dataset visualization...")
    
    # Create output directory
    output_base = "/tmp/milliontrees_visualization"
    os.makedirs(output_base, exist_ok=True)
    
    # Load all datasets
    print("Loading datasets...")
    all_data = load_datasets()
    
    if all_data.empty:
        print("No data loaded. Check dataset paths.")
        return
    
    print(f"Loaded {len(all_data)} annotations from {all_data['source'].nunique()} sources")
    
    # Create mini dataset (10 images per source)
    print("Creating mini dataset...")
    mini_data = create_mini_dataset(all_data, images_per_source=10)
    print(f"Mini dataset: {len(mini_data)} annotations from {mini_data['source'].nunique()} sources")
    
    # Split mini dataset for reviewers
    print("Creating mini dataset splits for reviewers...")
    mini_output_dir = os.path.join(output_base, "mini_dataset")
    mini_splits = split_for_reviewers(mini_data, mini_output_dir, "mini_dataset")
    
    # Split full dataset for reviewers
    print("Creating full dataset splits for reviewers...")
    full_output_dir = os.path.join(output_base, "full_dataset")
    full_splits = split_for_reviewers(all_data, full_output_dir, "full_dataset")
    
    # Generate summary report
    summary = {
        'total_annotations': len(all_data),
        'total_images': all_data['filename'].nunique(),
        'total_sources': all_data['source'].nunique(),
        'dataset_types': all_data['dataset_type'].value_counts().to_dict(),
        'mini_dataset_info': {
            'total_annotations': len(mini_data),
            'total_images': mini_data['filename'].nunique(),
            'splits': mini_splits
        },
        'full_dataset_info': {
            'total_annotations': len(all_data),
            'total_images': all_data['filename'].nunique(),
            'splits': full_splits
        }
    }
    
    # Save summary
    import json
    with open(os.path.join(output_base, "visualization_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nVisualization complete! Output saved to: {output_base}")
    print(f"Summary: {summary['total_annotations']} annotations, {summary['total_images']} images, {summary['total_sources']} sources")

if __name__ == "__main__":
    main()