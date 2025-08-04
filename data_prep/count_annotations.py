import pandas as pd
import os
from pathlib import Path

def analyze_dataset(file_path, dataset_type):
    """Analyze a single dataset and return statistics."""
    try:
        if not os.path.exists(file_path):
            return {
                'file_path': file_path,
                'dataset_type': dataset_type,
                'status': 'File not found',
                'num_annotations': 0,
                'unique_images': 0,
                'source': 'N/A'
            }
        
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Get basic statistics
        num_annotations = len(df)
        
        # Count unique images (assuming there's an image_path column)
        # Try different possible column names for image paths
        image_col = None
        for col in ['image_path', 'image', 'filename', 'file_name', 'img_path']:
            if col in df.columns:
                image_col = col
                break
        
        unique_images = len(df[image_col].unique()) if image_col else 0
        
        # Extract source from file path
        source = Path(file_path).parent.name
        
        return {
            'file_path': file_path,
            'dataset_type': dataset_type,
            'status': 'Loaded successfully',
            'num_annotations': num_annotations,
            'unique_images': unique_images,
            'source': source,
            'columns': list(df.columns)
        }
        
    except Exception as e:
        return {
            'file_path': file_path,
            'dataset_type': dataset_type,
            'status': f'Error: {str(e)}',
            'num_annotations': 0,
            'unique_images': 0,
            'source': 'N/A'
        }

def main():
    # Dataset lists
    TreeBoxes = [
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
        "/orange/ewhite/DeepForest/Zenodo_15155081/parsed_annotations.csv",
        "/orange/ewhite/DeepForest/SelvaBox/annotations.csv"
    ]

    TreePoints = [
        "/orange/ewhite/DeepForest/TreeFormer/all_images/annotations.csv",
        "/orange/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/annotations.csv",
        "/orange/ewhite/MillionTrees/NEON_points/annotations.csv",
        "/orange/ewhite/DeepForest/Tonga/annotations.csv",
        '/orange/ewhite/DeepForest/BohlmanBCI/crops/annotations_points.csv',
        "/orange/ewhite/DeepForest/AutoArborist/downloaded_imagery/AutoArborist_combined_annotations.csv"
    ]

    TreePolygons = [
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

    # Combine all datasets
    all_datasets = []
    all_datasets.extend([(path, 'TreeBoxes') for path in TreeBoxes])
    all_datasets.extend([(path, 'TreePoints') for path in TreePoints])
    all_datasets.extend([(path, 'TreePolygons') for path in TreePolygons])

    print("=" * 80)
    print("DATASET ANALYSIS REPORT")
    print("=" * 80)
    
    total_annotations = 0
    total_unique_images = 0
    successful_datasets = 0
    
    # Analyze each dataset
    for file_path, dataset_type in all_datasets:
        print(f"\nAnalyzing: {file_path}")
        print(f"Type: {dataset_type}")
        
        result = analyze_dataset(file_path, dataset_type)
        
        print(f"Status: {result['status']}")
        if result['status'] == 'Loaded successfully':
            print(f"Number of annotations: {result['num_annotations']:,}")
            print(f"Unique images: {result['unique_images']:,}")
            print(f"Source: {result['source']}")
            print(f"Columns: {result['columns']}")
            
            total_annotations += result['num_annotations']
            total_unique_images += result['unique_images']
            successful_datasets += 1
        else:
            print(f"Error: {result['status']}")
        
        print("-" * 60)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total datasets processed: {len(all_datasets)}")
    print(f"Successfully loaded: {successful_datasets}")
    print(f"Failed to load: {len(all_datasets) - successful_datasets}")
    print(f"Total annotations across all datasets: {total_annotations:,}")
    print(f"Total unique images across all datasets: {total_unique_images:,}")
    
    # Breakdown by type
    print("\nBreakdown by dataset type:")
    for dataset_type in ['TreeBoxes', 'TreePoints', 'TreePolygons']:
        type_datasets = [ds for ds in all_datasets if ds[1] == dataset_type]
        type_results = [analyze_dataset(path, dtype) for path, dtype in type_datasets]
        type_annotations = sum(r['num_annotations'] for r in type_results if r['status'] == 'Loaded successfully')
        type_images = sum(r['unique_images'] for r in type_results if r['status'] == 'Loaded successfully')
        print(f"  {dataset_type}: {type_annotations:,} annotations, {type_images:,} unique images")

if __name__ == "__main__":
    main()