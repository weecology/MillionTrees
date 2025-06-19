#!/usr/bin/env python3
"""
Test Image Sources and Visualize Results

This script tests a single image for each available data source, downloads imagery 
from various image servers, runs DeepForest predictions, and overlays the results 
with ground truth annotations to help users verify that data sources match.

Usage:
    python test_image_sources.py [--output_dir results] [--max_sources 5]

Dependencies:
    - deepforest
    - requests  
    - rasterio
    - geopandas
    - pandas
    - matplotlib
    - pillow
"""

import os
import sys
import argparse
import requests
import json
import logging
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# DeepForest imports
try:
    from deepforest import main
    from deepforest.visualize import plot_results
    from deepforest.utilities import read_file
except ImportError as e:
    print(f"Error importing DeepForest: {e}")
    print("Please install DeepForest: pip install deepforest")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Image server configurations
IMAGE_SERVERS = {
    "calgary": {
        "name": "Calgary, Canada",
        "url": "https://gis.calgary.ca/arcgis/rest/services/pub_Orthophotos/CurrentOrthophoto/ImageServer/exportImage",
        "type": "arcgis_imageserver",
        "crs": "EPSG:3776",  # 3TM 114W NAD83
        "test_bbox": [-113.1, 51.0, -113.0, 51.1],  # rough Calgary bbox in WGS84
        "resolution": 0.1  # 10cm
    },
    "charlottesville": {
        "name": "Charlottesville, Virginia, USA",
        "url": "https://gismaps.vdem.virginia.gov/arcgis/rest/services/VBMP_Imagery/MostRecentImagery_WGS_Tile_Index/MapServer/export",
        "type": "arcgis_mapserver", 
        "crs": "EPSG:3857",  # Web Mercator
        "test_bbox": [-78.5, 38.0, -78.4, 38.1],  # rough Charlottesville bbox
        "resolution": 0.3  # 1 foot
    },
    "new_york": {
        "name": "New York State, USA",
        "url": "https://orthos.its.ny.gov/arcgis/rest/services/wms/Latest/MapServer/export",
        "type": "arcgis_mapserver",
        "crs": "EPSG:3857",  # Web Mercator
        "test_bbox": [-74.0, 40.7, -73.9, 40.8],  # rough NYC bbox
        "resolution": 0.36  # 12 inch
    },
    "washington_dc": {
        "name": "Washington DC, USA", 
        "url": "https://imagery.dcgis.dc.gov/dcgis/rest/services/Ortho/Ortho_2023/ImageServer/exportImage",
        "type": "arcgis_imageserver",
        "crs": "EPSG:3857",  # Web Mercator
        "test_bbox": [-77.1, 38.8, -77.0, 38.9],  # rough DC bbox
        "resolution": 0.15  # 6 inch
    }
}

# Tree location data sources (CSV files with coordinates)
TREE_DATA_SOURCES = {
    "calgary": "data_prep/tree_locations/CalgaryTrees.csv",
    "charlottesville": "data_prep/tree_locations/CharlottesvilleTrees.csv", 
    "new_york": "data_prep/tree_locations/NewYorkTrees.csv",
    "washington_dc": "data_prep/tree_locations/WashingtonDcTrees.csv",
    "vancouver": "data_prep/tree_locations/VancouverTrees.csv",
    "seattle": "data_prep/tree_locations/SeattleTrees.csv",
    "montreal": "data_prep/tree_locations/MontrealTrees.csv",
    "denver": "data_prep/tree_locations/DenverTrees.csv"
}

def setup_output_directory(output_dir):
    """Create output directory structure."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    subdirs = ['images', 'annotations', 'visualizations', 'reports']
    for subdir in subdirs:
        (output_path / subdir).mkdir(exist_ok=True)
        
    return output_path

def test_image_server(server_name, server_config, timeout=30):
    """Test if an image server is accessible and functional."""
    logger.info(f"Testing image server: {server_name}")
    
    try:
        # Construct test URL
        bbox = server_config["test_bbox"]
        url = server_config["url"]
        
        if server_config["type"] == "arcgis_imageserver":
            params = {
                "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "bboxSR": "4326",  # WGS84
                "size": "256,256",
                "format": "png",
                "f": "image"
            }
        elif server_config["type"] == "arcgis_mapserver":
            params = {
                "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "bboxSR": "4326",  # WGS84
                "size": "256,256",
                "format": "png",
                "f": "image"
            }
        
        response = requests.get(url, params=params, timeout=timeout)
        
        if response.status_code == 200:
            # Check if response is actually an image
            if response.headers.get('content-type', '').startswith('image'):
                logger.info(f"✓ {server_name} is functional")
                return {"status": "functional", "response_size": len(response.content)}
            else:
                logger.warning(f"✗ {server_name} returned non-image content")
                return {"status": "error", "message": "Non-image response"}
        else:
            logger.warning(f"✗ {server_name} returned status {response.status_code}")
            return {"status": "error", "message": f"HTTP {response.status_code}"}
            
    except Exception as e:
        logger.error(f"✗ {server_name} failed: {str(e)}")
        return {"status": "error", "message": str(e)}

def download_image_from_server(server_name, server_config, bbox, image_size=(512, 512), output_path=None):
    """Download an image from an image server for a specific bounding box."""
    logger.info(f"Downloading image from {server_name}")
    
    try:
        url = server_config["url"]
        
        if server_config["type"] == "arcgis_imageserver":
            params = {
                "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "bboxSR": "4326",  # WGS84  
                "size": f"{image_size[0]},{image_size[1]}",
                "format": "png",
                "f": "image"
            }
        elif server_config["type"] == "arcgis_mapserver":
            params = {
                "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "bboxSR": "4326",  # WGS84
                "size": f"{image_size[0]},{image_size[1]}",
                "format": "png", 
                "f": "image"
            }
            
        response = requests.get(url, params=params, timeout=60)
        
        if response.status_code == 200 and response.headers.get('content-type', '').startswith('image'):
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Image saved to {output_path}")
            return response.content
        else:
            logger.error(f"Failed to download image from {server_name}")
            return None
            
    except Exception as e:
        logger.error(f"Error downloading from {server_name}: {str(e)}")
        return None

def load_tree_data(source_file, max_trees=50):
    """Load tree location data from CSV file."""
    try:
        if not os.path.exists(source_file):
            logger.warning(f"Tree data file not found: {source_file}")
            return None
            
        df = pd.read_csv(source_file)
        
        # Check for required columns
        if 'SHAPE_LNG' not in df.columns or 'SHAPE_LAT' not in df.columns:
            logger.warning(f"Missing coordinate columns in {source_file}")
            return None
            
        # Sample random trees if too many
        if len(df) > max_trees:
            df = df.sample(n=max_trees)
            
        logger.info(f"Loaded {len(df)} tree records from {source_file}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading tree data from {source_file}: {str(e)}")
        return None

def create_test_bbox_from_trees(tree_df, buffer_deg=0.01):
    """Create a bounding box around a cluster of trees."""
    if tree_df is None or len(tree_df) == 0:
        return None
        
    # Find a cluster of trees (take first 10 and get bounds)
    sample_trees = tree_df.head(10)
    
    min_lng = sample_trees['SHAPE_LNG'].min() - buffer_deg
    max_lng = sample_trees['SHAPE_LNG'].max() + buffer_deg  
    min_lat = sample_trees['SHAPE_LAT'].min() - buffer_deg
    max_lat = sample_trees['SHAPE_LAT'].max() + buffer_deg
    
    return [min_lng, min_lat, max_lng, max_lat]

def run_deepforest_prediction(image_path):
    """Run DeepForest tree detection on an image."""
    try:
        logger.info("Running DeepForest prediction")
        
        # Load DeepForest model
        model = main.deepforest()
        model.use_release()
        
        # Load and predict on image
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # Run prediction
        predictions = model.predict_image(image_array, return_plot=False)
        
        if predictions is not None and len(predictions) > 0:
            logger.info(f"DeepForest detected {len(predictions)} trees")
            return predictions
        else:
            logger.info("DeepForest found no trees")
            return None
            
    except Exception as e:
        logger.error(f"Error running DeepForest: {str(e)}")
        return None

def create_ground_truth_annotations(tree_df, bbox, image_size):
    """Convert tree coordinates to image pixel coordinates for visualization."""
    if tree_df is None:
        return None
        
    try:
        # Filter trees within bbox
        trees_in_bbox = tree_df[
            (tree_df['SHAPE_LNG'] >= bbox[0]) & 
            (tree_df['SHAPE_LNG'] <= bbox[2]) &
            (tree_df['SHAPE_LAT'] >= bbox[1]) & 
            (tree_df['SHAPE_LAT'] <= bbox[3])
        ].copy()
        
        if len(trees_in_bbox) == 0:
            return None
            
        # Convert to pixel coordinates
        lng_range = bbox[2] - bbox[0] 
        lat_range = bbox[3] - bbox[1]
        
        trees_in_bbox['pixel_x'] = ((trees_in_bbox['SHAPE_LNG'] - bbox[0]) / lng_range * image_size[0]).astype(int)
        trees_in_bbox['pixel_y'] = ((bbox[3] - trees_in_bbox['SHAPE_LAT']) / lat_range * image_size[1]).astype(int)  # Flip Y
        
        # Create bounding boxes (assume 10 pixel tree crowns)
        crown_size = 10
        trees_in_bbox['xmin'] = trees_in_bbox['pixel_x'] - crown_size//2
        trees_in_bbox['ymin'] = trees_in_bbox['pixel_y'] - crown_size//2  
        trees_in_bbox['xmax'] = trees_in_bbox['pixel_x'] + crown_size//2
        trees_in_bbox['ymax'] = trees_in_bbox['pixel_y'] + crown_size//2
        
        # Ensure bounds are within image
        trees_in_bbox['xmin'] = trees_in_bbox['xmin'].clip(0, image_size[0])
        trees_in_bbox['ymin'] = trees_in_bbox['ymin'].clip(0, image_size[1])
        trees_in_bbox['xmax'] = trees_in_bbox['xmax'].clip(0, image_size[0])
        trees_in_bbox['ymax'] = trees_in_bbox['ymax'].clip(0, image_size[1])
        
        logger.info(f"Created annotations for {len(trees_in_bbox)} trees in image bounds")
        return trees_in_bbox[['xmin', 'ymin', 'xmax', 'ymax']]
        
    except Exception as e:
        logger.error(f"Error creating ground truth annotations: {str(e)}")
        return None

def create_visualization(image_path, predictions, ground_truth, output_path):
    """Create a visualization showing DeepForest predictions vs ground truth."""
    try:
        logger.info("Creating visualization")
        
        # Load image
        image = Image.open(image_path)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Image with DeepForest predictions
        axes[1].imshow(image)
        if predictions is not None and len(predictions) > 0:
            for _, pred in predictions.iterrows():
                rect = Rectangle(
                    (pred['xmin'], pred['ymin']),
                    pred['xmax'] - pred['xmin'],
                    pred['ymax'] - pred['ymin'],
                    linewidth=2, edgecolor='red', facecolor='none', alpha=0.8
                )
                axes[1].add_patch(rect)
        axes[1].set_title(f"DeepForest Predictions ({len(predictions) if predictions is not None else 0} trees)")
        axes[1].axis('off')
        
        # Image with ground truth
        axes[2].imshow(image)
        if ground_truth is not None and len(ground_truth) > 0:
            for _, gt in ground_truth.iterrows():
                rect = Rectangle(
                    (gt['xmin'], gt['ymin']),
                    gt['xmax'] - gt['xmin'], 
                    gt['ymax'] - gt['ymin'],
                    linewidth=2, edgecolor='blue', facecolor='none', alpha=0.8
                )
                axes[2].add_patch(rect)
        axes[2].set_title(f"Ground Truth ({len(ground_truth) if ground_truth is not None else 0} trees)")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        return False

def test_single_data_source(source_name, output_dir, image_servers, tree_data_sources):
    """Test a single data source with image download and tree detection."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Testing data source: {source_name}")
    logger.info(f"{'='*50}")
    
    results = {
        "source": source_name,
        "server_test": None,
        "image_download": False,
        "tree_data_loaded": False,
        "deepforest_predictions": 0,
        "ground_truth_trees": 0,
        "visualization_created": False,
        "recommendations": []
    }
    
    # Test image server if available
    if source_name in image_servers:
        server_config = image_servers[source_name]
        results["server_test"] = test_image_server(source_name, server_config)
        
        # Load tree data if available
        tree_df = None
        if source_name in tree_data_sources:
            tree_df = load_tree_data(tree_data_sources[source_name])
            if tree_df is not None:
                results["tree_data_loaded"] = True
                results["ground_truth_trees"] = len(tree_df)
                
        # Create test bbox - use tree locations if available, otherwise default
        if tree_df is not None:
            bbox = create_test_bbox_from_trees(tree_df)
        else:
            bbox = server_config["test_bbox"]
            
        if bbox and results["server_test"]["status"] == "functional":
            # Download test image
            image_path = output_dir / "images" / f"{source_name}_test.png"
            image_data = download_image_from_server(source_name, server_config, bbox, output_path=image_path)
            
            if image_data:
                results["image_download"] = True
                
                # Run DeepForest prediction
                predictions = run_deepforest_prediction(image_path)
                if predictions is not None:
                    results["deepforest_predictions"] = len(predictions)
                    
                # Create ground truth annotations
                ground_truth = create_ground_truth_annotations(tree_df, bbox, (512, 512))
                
                # Create visualization
                viz_path = output_dir / "visualizations" / f"{source_name}_comparison.png"
                viz_success = create_visualization(image_path, predictions, ground_truth, viz_path)
                results["visualization_created"] = viz_success
                
    # Generate recommendations
    if results["server_test"] and results["server_test"]["status"] != "functional":
        results["recommendations"].append("Image server appears to be down or misconfigured")
        
    if not results["tree_data_loaded"]:
        results["recommendations"].append("No ground truth tree data available for validation")
        
    if results["deepforest_predictions"] == 0:
        results["recommendations"].append("DeepForest detected no trees - image may lack tree coverage or have poor quality")
        
    return results

def generate_summary_report(all_results, output_dir):
    """Generate a comprehensive summary report."""
    logger.info("Generating summary report")
    
    report_path = output_dir / "reports" / "summary_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# MillionTrees Image Sources Test Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        
        total_sources = len(all_results)
        functional_servers = sum(1 for r in all_results if r.get("server_test", {}).get("status") == "functional")
        successful_downloads = sum(1 for r in all_results if r.get("image_download", False))
        
        f.write(f"- **Total data sources tested**: {total_sources}\n")
        f.write(f"- **Functional image servers**: {functional_servers}/{total_sources}\n")
        f.write(f"- **Successful image downloads**: {successful_downloads}/{total_sources}\n\n")
        
        f.write("## Server Status\n\n")
        f.write("| Source | Server Status | Image Download | Trees Detected | Ground Truth Trees |\n")
        f.write("|--------|---------------|----------------|----------------|--------------------|\n")
        
        for result in all_results:
            server_status = result.get("server_test", {}).get("status", "not_tested")
            download_status = "✓" if result.get("image_download", False) else "✗"
            predictions = result.get("deepforest_predictions", 0)
            ground_truth = result.get("ground_truth_trees", 0)
            
            f.write(f"| {result['source']} | {server_status} | {download_status} | {predictions} | {ground_truth} |\n")
            
        f.write("\n## Detailed Results\n\n")
        
        for result in all_results:
            f.write(f"### {result['source']}\n\n")
            
            if result.get("server_test"):
                f.write(f"**Server Status**: {result['server_test']['status']}\n\n")
                
            if result.get("recommendations"):
                f.write("**Recommendations**:\n")
                for rec in result["recommendations"]:
                    f.write(f"- {rec}\n")
                f.write("\n")
                
        f.write("## Recommendations for Improving Flexibility\n\n")
        f.write("1. **Parameterize server configurations**: Store server URLs, CRS, and parameters in a JSON config file\n")
        f.write("2. **Add fallback servers**: Configure backup image servers for each region\n") 
        f.write("3. **Implement caching**: Cache downloaded imagery to reduce server load\n")
        f.write("4. **Add server health monitoring**: Implement periodic health checks for all servers\n")
        f.write("5. **Support multiple image formats**: Add support for WMTS, WCS, and other OGC services\n")
        f.write("6. **Flexible coordinate systems**: Add automatic CRS transformation capabilities\n")
        f.write("7. **Dynamic bounding box generation**: Automatically generate test areas based on tree density\n")
        
    logger.info(f"Summary report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Test image sources and visualize tree detection results")
    parser.add_argument("--output_dir", default="test_results", help="Output directory for results")
    parser.add_argument("--max_sources", type=int, default=10, help="Maximum number of sources to test")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout for server requests (seconds)")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = setup_output_directory(args.output_dir)
    logger.info(f"Results will be saved to: {output_dir}")
    
    # Test all available sources
    all_results = []
    
    sources_to_test = list(IMAGE_SERVERS.keys())[:args.max_sources]
    
    for source_name in sources_to_test:
        try:
            result = test_single_data_source(source_name, output_dir, IMAGE_SERVERS, TREE_DATA_SOURCES)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed to test {source_name}: {str(e)}")
            all_results.append({
                "source": source_name,
                "error": str(e),
                "recommendations": ["Failed during testing - check logs for details"]
            })
    
    # Generate summary report
    generate_summary_report(all_results, output_dir)
    
    # Print summary to console
    logger.info(f"\n{'='*50}")
    logger.info("TESTING COMPLETE")
    logger.info(f"{'='*50}")
    logger.info(f"Tested {len(all_results)} data sources")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("Check the summary report for detailed recommendations")

if __name__ == "__main__":
    main()