#!/usr/bin/env python3
"""
Concise visualization script for MillionTrees datasets.
Loops through all datasets, overlays annotations, and creates reviewer splits.
"""

import pandas as pd
import os
import numpy as np
import cv2
from deepforest.visualize import plot_results
from deepforest.utilities import read_file
import json

# Dataset paths
DATASETS = {
    'TreeBoxes': ["/orange/ewhite/DeepForest/Ryoungseob_2023/train_datasets/images/train.csv",
                  "/orange/ewhite/DeepForest/Velasquez_urban_trees/tree_canopies/nueva_carpeta/annotations.csv",
                  "/orange/ewhite/DeepForest/individual_urban_tree_crown_detection/annotations.csv",
                  "/orange/ewhite/DeepForest/Radogoshi_Sweden/annotations.csv",
                  "/orange/ewhite/DeepForest/WRI/WRI-labels-opensource/annotations.csv",
                  "/orange/ewhite/DeepForest/Guangzhou2022/annotations.csv",
                  "/orange/ewhite/DeepForest/NEON_benchmark/NeonTreeEvaluation_annotations.csv",
                  "/orange/ewhite/DeepForest/NEON_benchmark/University_of_Florida.csv",
                  "/orange/ewhite/DeepForest/ReForestTree/images/train.csv",
                  "/orange/ewhite/DeepForest/Santos2019/annotations.csv",
                  "/orange/ewhite/DeepForest/Zenodo_15155081/parsed_annotations.csv",
                  "/orange/ewhite/DeepForest/SelvaBox/annotations.csv"],
    'TreePoints': ["/orange/ewhite/DeepForest/TreeFormer/all_images/annotations.csv",
                   "/orange/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/annotations.csv",
                   "/orange/ewhite/MillionTrees/NEON_points/annotations.csv",
                   "/orange/ewhite/DeepForest/Tonga/annotations.csv",
                   "/orange/ewhite/DeepForest/BohlmanBCI/crops/annotations_points.csv",
                   "/orange/ewhite/DeepForest/AutoArborist/downloaded_imagery/AutoArborist_combined_annotations.csv"],
    'TreePolygons': ["/orange/ewhite/DeepForest/Jansen_2023/pngs/annotations.csv",
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
                     "/orange/ewhite/DeepForest/takeshige2025/crops/annotations.csv"]
}

def load_all_data():
    """Load and combine all datasets."""
    return pd.concat([
        pd.read_csv(path).assign(dataset_type=dtype, csv_path=path).rename(columns={'image_path': 'filename'})
        for dtype, paths in DATASETS.items() for path in paths if os.path.exists(path)
    ], ignore_index=True)

def visualize_annotations(df, output_dir):
    """Generate visualizations for all images."""
    os.makedirs(output_dir, exist_ok=True)
    
    for (source, filename), group in df.groupby(['source', 'filename']):
        try:
            root_dir = os.path.dirname(group['csv_path'].iloc[0])
            plot_data = read_file(group.assign(image_path=group['filename']), root_dir=root_dir)
            plot_data.root_dir = root_dir
            
            safe_name = f"{source.replace(' ', '_').replace('/', '_')}_{os.path.splitext(os.path.basename(filename))[0]}"
            
            # Use image dimensions for polygons
            image_path = os.path.join(root_dir, filename)
            if os.path.exists(image_path) and group['dataset_type'].iloc[0] == "TreePolygons":
                h, w, _ = cv2.imread(image_path).shape
                plot_results(plot_data, savedir=output_dir, basename=safe_name, height=h, width=w)
            elif os.path.exists(image_path):
                plot_results(plot_data, savedir=output_dir, basename=safe_name)
                
        except Exception as e:
            print(f"Error: {source} - {filename}: {e}")

def create_splits(df, output_dir, prefix, num_splits=4):
    """Create equal dataset splits for reviewers."""
    images = df['filename'].unique()
    np.random.shuffle(images)
    
    for i, split_images in enumerate(np.array_split(images, num_splits)):
        split_data = df[df['filename'].isin(split_images)]
        split_dir = os.path.join(output_dir, f"{prefix}_reviewer_{i+1}")
        os.makedirs(split_dir, exist_ok=True)
        
        split_data.to_csv(os.path.join(split_dir, "annotations.csv"), index=False)
        visualize_annotations(split_data, os.path.join(split_dir, "visualizations"))

def main():
    """Main execution: load data, create mini/full splits, generate visualizations."""
    output_base = "/tmp/milliontrees_viz"
    os.makedirs(output_base, exist_ok=True)
    
    # Load all data
    all_data = load_all_data()
    print(f"Loaded {len(all_data)} annotations from {all_data['source'].nunique()} sources")
    
    # Create mini dataset (10 best images per source)
    mini_data = pd.concat([
        source_data[source_data['filename'].isin(
            source_data.groupby('filename').size().nlargest(10).index
        )] for _, source_data in all_data.groupby('source')
    ])
    
    # Create splits
    create_splits(mini_data, os.path.join(output_base, "mini_dataset"), "mini")
    create_splits(all_data, os.path.join(output_base, "full_dataset"), "full")
    
    # Save summary
    summary = {
        'total_annotations': len(all_data),
        'total_images': all_data['filename'].nunique(),
        'total_sources': all_data['source'].nunique(),
        'dataset_types': all_data['dataset_type'].value_counts().to_dict()
    }
    
    with open(os.path.join(output_base, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Complete! Output: {output_base}")
    print(f"{summary['total_annotations']} annotations, {summary['total_images']} images, {summary['total_sources']} sources")

if __name__ == "__main__":
    main()