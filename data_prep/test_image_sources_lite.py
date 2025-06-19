#!/usr/bin/env python3
"""
Test Image Sources (Lite Version)

This script tests image sources without requiring DeepForest for basic server connectivity testing.
For full functionality including tree detection, use test_image_sources.py with DeepForest installed.

Usage:
    python test_image_sources_lite.py [--output_dir results] [--max_sources 5]
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
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration from JSON file
def load_config():
    config_file = Path(__file__).parent / "image_server_config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        # Fallback to hardcoded config
        return {
            "image_servers": {
                "calgary": {
                    "name": "Calgary, Canada",
                    "url": "https://gis.calgary.ca/arcgis/rest/services/pub_Orthophotos/CurrentOrthophoto/ImageServer/exportImage",
                    "type": "arcgis_imageserver",
                    "crs": "EPSG:3776",
                    "test_bbox": [-113.1, 51.0, -113.0, 51.1],
                    "resolution": 0.1
                }
            },
            "testing_config": {
                "default_image_size": [512, 512],
                "max_trees_to_load": 50,
                "default_timeout": 30,
                "bbox_buffer_degrees": 0.01,
                "crown_size_pixels": 10
            }
        }

# Tree location data sources
TREE_DATA_SOURCES = {
    "calgary": "tree_locations/CalgaryTrees.csv",
    "charlottesville": "tree_locations/CharlottesvilleTrees.csv", 
    "new_york": "tree_locations/NewYorkTrees.csv",
    "washington_dc": "tree_locations/WashingtonDcTrees.csv",
    "vancouver": "tree_locations/VancouverTrees.csv",
    "seattle": "tree_locations/SeattleTrees.csv",
    "montreal": "tree_locations/MontrealTrees.csv",
    "denver": "tree_locations/DenverTrees.csv"
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
        start_time = datetime.now()
        
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
        elif server_config["type"] == "wms":
            params = {
                "service": "WMS",
                "version": "1.3.0",
                "request": "GetMap",
                "bbox": f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}",  # lat,lon order for WMS 1.3.0
                "crs": "EPSG:4326",
                "width": "256",
                "height": "256",
                "format": "image/png"
            }
        
        response = requests.get(url, params=params, timeout=timeout)
        response_time = (datetime.now() - start_time).total_seconds()
        
        if response.status_code == 200:
            # Check if response is actually an image
            content_type = response.headers.get('content-type', '')
            if content_type.startswith('image'):
                logger.info(f"✓ {server_name} is functional (Response time: {response_time:.2f}s)")
                return {
                    "status": "functional", 
                    "response_size": len(response.content),
                    "response_time": response_time,
                    "content_type": content_type
                }
            else:
                logger.warning(f"✗ {server_name} returned non-image content: {content_type}")
                return {"status": "error", "message": f"Non-image response: {content_type}"}
        else:
            logger.warning(f"✗ {server_name} returned status {response.status_code}")
            return {"status": "error", "message": f"HTTP {response.status_code}"}
            
    except requests.Timeout:
        logger.error(f"✗ {server_name} timed out after {timeout}s")
        return {"status": "error", "message": "Request timeout"}
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
        elif server_config["type"] == "wms":
            params = {
                "service": "WMS",
                "version": "1.3.0",
                "request": "GetMap",
                "bbox": f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}",  # lat,lon order
                "crs": "EPSG:4326",
                "width": str(image_size[0]),
                "height": str(image_size[1]),
                "format": "image/png"
            }
             
        response = requests.get(url, params=params, timeout=60)
        
        if response.status_code == 200 and response.headers.get('content-type', '').startswith('image'):
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Image saved to {output_path}")
            return response.content
        else:
            logger.error(f"Failed to download image from {server_name}: {response.status_code}")
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

def create_basic_visualization(image_path, ground_truth, output_path):
    """Create a basic visualization showing the image and ground truth locations."""
    try:
        logger.info("Creating basic visualization")
        
        # Load image
        image = Image.open(image_path)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Downloaded Image")
        axes[0].axis('off')
        
        # Image with ground truth
        axes[1].imshow(image)
        if ground_truth is not None and len(ground_truth) > 0:
            for _, gt in ground_truth.iterrows():
                rect = Rectangle(
                    (gt['xmin'], gt['ymin']),
                    gt['xmax'] - gt['xmin'], 
                    gt['ymax'] - gt['ymin'],
                    linewidth=2, edgecolor='blue', facecolor='none', alpha=0.8
                )
                axes[1].add_patch(rect)
        axes[1].set_title(f"Ground Truth Trees ({len(ground_truth) if ground_truth is not None else 0} trees)")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        return False

def test_single_data_source(source_name, output_dir, image_servers, tree_data_sources, config):
    """Test a single data source with image download."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Testing data source: {source_name}")
    logger.info(f"{'='*50}")
    
    results = {
        "source": source_name,
        "server_test": None,
        "image_download": False,
        "tree_data_loaded": False,
        "ground_truth_trees": 0,
        "visualization_created": False,
        "recommendations": []
    }
    
    # Test image server if available
    if source_name in image_servers:
        server_config = image_servers[source_name]
        results["server_test"] = test_image_server(source_name, server_config, config["testing_config"]["default_timeout"])
        
        # Load tree data if available
        tree_df = None
        if source_name in tree_data_sources:
            tree_df = load_tree_data(tree_data_sources[source_name], config["testing_config"]["max_trees_to_load"])
            if tree_df is not None:
                results["tree_data_loaded"] = True
                results["ground_truth_trees"] = len(tree_df)
                
        # Create test bbox - use tree locations if available, otherwise default
        if tree_df is not None:
            bbox = create_test_bbox_from_trees(tree_df, config["testing_config"]["bbox_buffer_degrees"])
        else:
            bbox = server_config["test_bbox"]
            
        if bbox and results["server_test"]["status"] == "functional":
            # Download test image
            image_path = output_dir / "images" / f"{source_name}_test.png"
            image_data = download_image_from_server(source_name, server_config, bbox, 
                                                 config["testing_config"]["default_image_size"], output_path=image_path)
            
            if image_data:
                results["image_download"] = True
                                
                # Create ground truth annotations
                ground_truth = create_ground_truth_annotations(tree_df, bbox, config["testing_config"]["default_image_size"])
                
                # Create visualization
                viz_path = output_dir / "visualizations" / f"{source_name}_comparison.png"
                viz_success = create_basic_visualization(image_path, ground_truth, viz_path)
                results["visualization_created"] = viz_success
                
    # Generate recommendations
    if results["server_test"] and results["server_test"]["status"] != "functional":
        results["recommendations"].append("Image server appears to be down or misconfigured")
        
    if not results["tree_data_loaded"]:
        results["recommendations"].append("No ground truth tree data available for validation")
        
    return results

def generate_summary_report(all_results, output_dir):
    """Generate a comprehensive summary report."""
    logger.info("Generating summary report")
    
    report_path = output_dir / "reports" / "summary_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# MillionTrees Image Sources Test Report (Lite Version)\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("*Note: This report was generated without DeepForest tree detection. For full functionality including tree detection, install DeepForest and use test_image_sources.py*\n\n")
        
        f.write("## Executive Summary\n\n")
        
        total_sources = len(all_results)
        functional_servers = sum(1 for r in all_results if r.get("server_test", {}).get("status") == "functional")
        successful_downloads = sum(1 for r in all_results if r.get("image_download", False))
        
        f.write(f"- **Total data sources tested**: {total_sources}\n")
        f.write(f"- **Functional image servers**: {functional_servers}/{total_sources}\n")
        f.write(f"- **Successful image downloads**: {successful_downloads}/{total_sources}\n\n")
        
        f.write("## Server Status\n\n")
        f.write("| Source | Server Status | Image Download | Ground Truth Trees |\n")
        f.write("|--------|---------------|----------------|--------------------|\n")
        
        for result in all_results:
            server_status = result.get("server_test", {}).get("status", "not_tested")
            download_status = "✓" if result.get("image_download", False) else "✗"
            ground_truth = result.get("ground_truth_trees", 0)
            
            f.write(f"| {result['source']} | {server_status} | {download_status} | {ground_truth} |\n")
            
        f.write("\n## Detailed Results\n\n")
        
        for result in all_results:
            f.write(f"### {result['source']}\n\n")
            
            if result.get("server_test"):
                server_test = result["server_test"]
                f.write(f"**Server Status**: {server_test['status']}\n")
                if "response_time" in server_test:
                    f.write(f"**Response Time**: {server_test['response_time']:.2f}s\n")
                if "response_size" in server_test:
                    f.write(f"**Response Size**: {server_test['response_size']} bytes\n")
                f.write("\n")
                
            if result.get("recommendations"):
                f.write("**Recommendations**:\n")
                for rec in result["recommendations"]:
                    f.write(f"- {rec}\n")
                f.write("\n")
                
        f.write("## Next Steps\n\n")
        f.write("1. Install DeepForest for tree detection functionality: `pip install deepforest`\n")
        f.write("2. Run the full test script: `python test_image_sources.py`\n")
        f.write("3. Review server configurations in `image_server_config.json`\n")
        f.write("4. Add missing tree location data for cities without ground truth\n")
        
    logger.info(f"Summary report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Test image sources (lite version without DeepForest)")
    parser.add_argument("--output_dir", default="test_results_lite", help="Output directory for results")
    parser.add_argument("--max_sources", type=int, default=4, help="Maximum number of sources to test")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout for server requests (seconds)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Setup output directory
    output_dir = setup_output_directory(args.output_dir)
    logger.info(f"Results will be saved to: {output_dir}")
    
    # Test all available sources
    all_results = []
    
    image_servers = config["image_servers"]
    sources_to_test = list(image_servers.keys())[:args.max_sources]
    
    # Update timeout in config
    config["testing_config"]["default_timeout"] = args.timeout
    
    for source_name in sources_to_test:
        try:
            result = test_single_data_source(source_name, output_dir, image_servers, TREE_DATA_SOURCES, config)
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
    logger.info("TESTING COMPLETE (LITE VERSION)")
    logger.info(f"{'='*50}")
    logger.info(f"Tested {len(all_results)} data sources")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("Check the summary report for detailed recommendations")
    logger.info("\nTo run full testing with tree detection, install DeepForest:")
    logger.info("pip install deepforest")

if __name__ == "__main__":
    main()