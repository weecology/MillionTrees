import os
import pandas as pd
import geopandas as gpd
import rasterio
import requests
from shapely.geometry import Point
from deepforest.utilities import read_file
from deepforest.visualize import plot_results
import matplotlib.pyplot as plt
from rasterio.windows import from_bounds
from rasterio.transform import xy
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

def download_metadata():
    """Download OFO metadata files from Google Drive"""
    # Create data directory
    os.makedirs("data_prep/OpenForestObservatory", exist_ok=True)
    
    # Note: These would need to be actual Google Drive download links
    # For now, we'll use placeholder URLs that would need to be updated
    metadata_urls = {
        "ground_reference_plots": "https://drive.google.com/uc?export=download&id=PLOT_METADATA_ID",
        "ground_reference_trees": "https://drive.google.com/uc?export=download&id=TREE_METADATA_ID", 
        "drone_missions_metadata": "https://drive.google.com/uc?export=download&id=DRONE_METADATA_ID"
    }
    
    # Download metadata files (this would need actual implementation)
    print("Note: Metadata download URLs need to be configured with actual Google Drive file IDs")
    return True

def load_ofo_data():
    """Load OFO ground reference and drone mission data"""
    base_dir = "data_prep/OpenForestObservatory"
    
    # For testing, create mock data structures
    # In reality, these would be loaded from the downloaded .gpkg files
    
    # Mock plot data - would be loaded from ofo_ground-reference_plots.gpkg
    plots_data = {
        'plot_id': ['plot_001', 'plot_002', 'plot_003'],
        'geometry': [Point(-120.5, 37.5), Point(-121.0, 38.0), Point(-119.8, 36.9)],
        'survey_date': ['2023-06-15', '2023-07-20', '2023-08-10'],
        'license': ['CC-BY-4.0', 'CC-BY-4.0', 'CC-BY-4.0']
    }
    plots_gdf = gpd.GeoDataFrame(plots_data, crs='EPSG:4326')
    
    # Mock tree data - would be loaded from ofo_ground-reference_trees.gpkg  
    trees_data = {
        'plot_id': ['plot_001', 'plot_001', 'plot_002', 'plot_002', 'plot_003'],
        'tree_id': ['tree_001_01', 'tree_001_02', 'tree_002_01', 'tree_002_02', 'tree_003_01'],
        'geometry': [Point(-120.499, 37.501), Point(-120.501, 37.499), 
                    Point(-121.001, 38.001), Point(-120.999, 37.999),
                    Point(-119.799, 36.901)],
        'species': ['Quercus douglasii', 'Pinus ponderosa', 'Abies concolor', 'Pinus lambertiana', 'Sequoiadendron giganteum'],
        'height': [15.2, 18.7, 22.1, 25.3, 28.9],
        'dbh': [0.35, 0.42, 0.58, 0.72, 0.89]
    }
    trees_gdf = gpd.GeoDataFrame(trees_data, crs='EPSG:4326')
    
    # Mock drone mission data - would be loaded from ofo_drone-missions_metadata.gpkg
    missions_data = {
        'mission_id': ['mission_001', 'mission_002', 'mission_003'],
        'plot_id': ['plot_001', 'plot_002', 'plot_003'],
        'geometry': [Point(-120.5, 37.5), Point(-121.0, 38.0), Point(-119.8, 36.9)],
        'flight_date': ['2023-06-16', '2023-07-21', '2023-08-11'],
        'flight_altitude': [100, 120, 110],
        'license': ['CC-BY-4.0', 'CC-BY-4.0', 'CC-BY-4.0']
    }
    missions_gdf = gpd.GeoDataFrame(missions_data, crs='EPSG:4326')
    
    return plots_gdf, trees_gdf, missions_gdf

def download_orthomosaic(mission_id, output_dir):
    """Download orthomosaic from Jetstream2 Object Store"""
    url = f"https://js2.jetstream-cloud.org:8001/ofo-public/drone/missions_01/{mission_id}/processed_01/full/{mission_id}_01_ortho-dsm-ptcloud.tif"
    output_path = os.path.join(output_dir, f"{mission_id}_orthomosaic.tif")
    
    # Check if file already exists
    if os.path.exists(output_path):
        print(f"Orthomosaic for {mission_id} already exists")
        return output_path
    
    try:
        print(f"Downloading orthomosaic for {mission_id}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded {mission_id} orthomosaic to {output_path}")
        return output_path
    
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {mission_id}: {e}")
        return None

def match_trees_to_orthomosaic(trees_gdf, mission_row, orthomosaic_path):
    """Match tree locations to pixel coordinates in orthomosaic"""
    if not os.path.exists(orthomosaic_path):
        return None
    
    plot_id = mission_row['plot_id']
    plot_trees = trees_gdf[trees_gdf['plot_id'] == plot_id].copy()
    
    if plot_trees.empty:
        return None
    
    try:
        with rasterio.open(orthomosaic_path) as src:
            # Convert geographic coordinates to pixel coordinates
            plot_trees['x_pixel'] = 0
            plot_trees['y_pixel'] = 0
            
            for idx, tree in plot_trees.iterrows():
                # Convert geographic coordinates to pixel coordinates
                lon, lat = tree.geometry.x, tree.geometry.y
                row, col = rasterio.transform.rowcol(src.transform, lon, lat)
                plot_trees.at[idx, 'x_pixel'] = int(col)
                plot_trees.at[idx, 'y_pixel'] = int(row)
            
            # Filter trees that fall within the image bounds
            height, width = src.height, src.width
            plot_trees = plot_trees[
                (plot_trees['x_pixel'] >= 0) & 
                (plot_trees['x_pixel'] < width) &
                (plot_trees['y_pixel'] >= 0) & 
                (plot_trees['y_pixel'] < height)
            ]
            
            return plot_trees
    
    except Exception as e:
        print(f"Error processing {orthomosaic_path}: {e}")
        return None

def create_annotations(matched_trees, orthomosaic_path, mission_id):
    """Create MillionTrees-compatible annotations"""
    if matched_trees is None or matched_trees.empty:
        return None
    
    annotations = []
    for _, tree in matched_trees.iterrows():
        annotation = {
            'image_path': os.path.basename(orthomosaic_path),
            'x': tree['x_pixel'],
            'y': tree['y_pixel'],
            'label': 'Tree',
            'source': 'Open Forest Observatory',
            'species': tree.get('species', 'Unknown'),
            'height': tree.get('height', 0),
            'dbh': tree.get('dbh', 0),
            'tree_id': tree.get('tree_id', ''),
            'plot_id': tree.get('plot_id', ''),
            'mission_id': mission_id
        }
        annotations.append(annotation)
    
    annotations_df = pd.DataFrame(annotations)
    # Create geometry column for compatibility
    annotations_df['geometry'] = annotations_df.apply(
        lambda row: f"POINT ({row['x']} {row['y']})", axis=1
    )
    
    return annotations_df

def create_test_visualization(annotations_df, orthomosaic_path, output_dir):
    """Create visualization showing tree points overlaid on orthomosaic"""
    if annotations_df is None or annotations_df.empty:
        print("No annotations to visualize")
        return None
    
    try:
        # Read the orthomosaic
        with rasterio.open(orthomosaic_path) as src:
            # Read a smaller window for visualization (center crop)
            height, width = src.height, src.width
            window_size = min(2000, height, width)  # Max 2000x2000 for visualization
            
            x_start = max(0, (width - window_size) // 2)
            y_start = max(0, (height - window_size) // 2)
            
            window = rasterio.windows.Window(x_start, y_start, window_size, window_size)
            image_data = src.read(window=window)
            
            # Convert to RGB if needed
            if image_data.shape[0] >= 3:
                image_rgb = image_data[:3].transpose(1, 2, 0)
                # Normalize to 0-255 range
                image_rgb = ((image_rgb - image_rgb.min()) / 
                           (image_rgb.max() - image_rgb.min()) * 255).astype(np.uint8)
            else:
                print("Image does not have RGB channels")
                return None
        
        # Filter annotations to the visualization window
        viz_annotations = annotations_df[
            (annotations_df['x'] >= x_start) & 
            (annotations_df['x'] < x_start + window_size) &
            (annotations_df['y'] >= y_start) & 
            (annotations_df['y'] < y_start + window_size)
        ].copy()
        
        if viz_annotations.empty:
            print("No trees in visualization window")
            return None
        
        # Adjust coordinates relative to the window
        viz_annotations['x'] = viz_annotations['x'] - x_start
        viz_annotations['y'] = viz_annotations['y'] - y_start
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(image_rgb)
        
        # Plot tree points
        ax.scatter(viz_annotations['x'], viz_annotations['y'], 
                  c='red', s=50, alpha=0.8, marker='o', edgecolors='white', linewidth=1)
        
        # Add title and labels
        ax.set_title(f'Open Forest Observatory Trees\n{os.path.basename(orthomosaic_path)}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X Pixel Coordinate')
        ax.set_ylabel('Y Pixel Coordinate')
        
        # Add text annotation with count
        ax.text(0.02, 0.98, f'Trees: {len(viz_annotations)}', 
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               verticalalignment='top')
        
        # Save visualization
        viz_path = os.path.join(output_dir, f"OFO_visualization_{os.path.splitext(os.path.basename(orthomosaic_path))[0]}.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {viz_path}")
        return viz_path
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return None

def process_single_mission(mission_id="mission_001", test_mode=True):
    """Process a single mission for testing"""
    output_dir = "data_prep/OpenForestObservatory"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    plots_gdf, trees_gdf, missions_gdf = load_ofo_data()
    
    # Get mission info
    mission_row = missions_gdf[missions_gdf['mission_id'] == mission_id].iloc[0]
    
    if test_mode:
        # For testing, create a mock orthomosaic
        mock_path = os.path.join(output_dir, f"{mission_id}_orthomosaic.tif")
        create_mock_orthomosaic(mock_path)
        orthomosaic_path = mock_path
    else:
        # Download real orthomosaic
        orthomosaic_path = download_orthomosaic(mission_id, output_dir)
        if orthomosaic_path is None:
            return None
    
    # Match trees to orthomosaic
    matched_trees = match_trees_to_orthomosaic(trees_gdf, mission_row, orthomosaic_path)
    
    # Create annotations
    annotations_df = create_annotations(matched_trees, orthomosaic_path, mission_id)
    
    if annotations_df is not None:
        # Save annotations
        annotations_path = os.path.join(output_dir, f"{mission_id}_annotations.csv")
        annotations_df.to_csv(annotations_path, index=False)
        print(f"Saved {len(annotations_df)} annotations to {annotations_path}")
        
        # Create visualization
        viz_path = create_test_visualization(annotations_df, orthomosaic_path, output_dir)
        
        return annotations_df, viz_path
    
    return None, None

def create_mock_orthomosaic(output_path, size=(2000, 2000)):
    """Create a mock orthomosaic for testing purposes"""
    if os.path.exists(output_path):
        return output_path
        
    # Create synthetic forest-like image
    np.random.seed(42)
    
    # Generate base green forest texture
    forest_base = np.random.randint(40, 120, (size[0], size[1], 3), dtype=np.uint8)
    forest_base[:, :, 1] += 20  # More green
    
    # Add some brown patches for soil/roads
    for _ in range(10):
        x, y = np.random.randint(0, size[0]-100), np.random.randint(0, size[1]-100)
        w, h = np.random.randint(20, 80), np.random.randint(20, 80)
        forest_base[x:x+w, y:y+h] = [101, 67, 33]  # Brown
    
    # Add texture variation
    noise = np.random.randint(-20, 20, (size[0], size[1], 3))
    forest_base = np.clip(forest_base.astype(int) + noise, 0, 255).astype(np.uint8)
    
    # Save as GeoTIFF with mock geospatial info
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    
    transform = from_bounds(-120.51, 37.49, -120.49, 37.51, size[1], size[0])
    
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=size[0],
        width=size[1], 
        count=3,
        dtype=rasterio.uint8,
        crs=CRS.from_epsg(4326),
        transform=transform
    ) as dst:
        dst.write(forest_base[:, :, 0], 1)  # Red
        dst.write(forest_base[:, :, 1], 2)  # Green  
        dst.write(forest_base[:, :, 2], 3)  # Blue
    
    print(f"Created mock orthomosaic: {output_path}")
    return output_path

def run_full_processing():
    """Process all available missions"""
    output_dir = "data_prep/OpenForestObservatory"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    plots_gdf, trees_gdf, missions_gdf = load_ofo_data()
    
    all_annotations = []
    
    for _, mission_row in missions_gdf.iterrows():
        mission_id = mission_row['mission_id']
        print(f"\nProcessing mission: {mission_id}")
        
        # Download orthomosaic
        orthomosaic_path = download_orthomosaic(mission_id, output_dir)
        if orthomosaic_path is None:
            continue
        
        # Match trees to orthomosaic
        matched_trees = match_trees_to_orthomosaic(trees_gdf, mission_row, orthomosaic_path)
        
        # Create annotations
        annotations_df = create_annotations(matched_trees, orthomosaic_path, mission_id)
        
        if annotations_df is not None:
            all_annotations.append(annotations_df)
    
    if all_annotations:
        # Combine all annotations
        combined_annotations = pd.concat(all_annotations, ignore_index=True)
        
        # Save combined annotations
        output_path = os.path.join(output_dir, "annotations.csv")
        combined_annotations.to_csv(output_path, index=False)
        
        print(f"\nSaved {len(combined_annotations)} total annotations to {output_path}")
        print(f"Annotations from {len(all_annotations)} missions")
        print(f"Images: {combined_annotations['image_path'].nunique()}")
        
        return output_path
    
    return None

def main():
    """Main execution function"""
    print("Open Forest Observatory Data Processing")
    print("=" * 50)
    
    # Test with a single mission first
    print("Testing with single mission...")
    annotations_df, viz_path = process_single_mission(test_mode=True)
    
    if annotations_df is not None:
        print(f"✓ Successfully processed test mission")
        print(f"✓ Created {len(annotations_df)} annotations") 
        if viz_path:
            print(f"✓ Created visualization: {viz_path}")
        
        # Uncomment to run full processing
        # print("\nRunning full processing...")
        # full_output = run_full_processing()
        # if full_output:
        #     print(f"✓ Full processing complete: {full_output}")
    else:
        print("✗ Test processing failed")

if __name__ == "__main__":
    main()