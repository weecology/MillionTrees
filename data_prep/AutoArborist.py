"""
AutoArborist: Automated overhead imagery acquisition and annotation for tree locations

This script downloads overhead imagery from ArcGIS Rest servers that match tree locations
from CSV files and creates annotations in the MillionTrees format.
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import requests
import time
from shapely.geometry import Point, box
import rasterio
from rasterio.warp import transform_bounds, calculate_default_transform
from rasterio.crs import CRS
from pyproj import Transformer
import warnings
from deepforest.preprocess import split_raster
from deepforest.utilities import read_file
from deepforest.visualize import plot_results
from PIL import Image
import io
import argparse
import glob
import matplotlib.pyplot as plt

# ArcGIS Rest Server endpoints for various cities
IMAGERY_SOURCES = {
    "calgary": {
        "url": "https://gis.calgary.ca/arcgis/rest/services/pub_Orthophotos/CurrentOrthophoto/ImageServer/exportImage",
        "crs": "EPSG:3776",  # Calgary 3TM (NAD83(CSRS) / UTM zone 11N)
        "pixel_size": 0.1  # 10cm resolution
    },
    "edmonton": {
        "url": "https://gis.edmonton.ca/site1/rest/services/Imagery_Public/2019_RGB_Pictometry/ImageServer/exportImage",
        "crs": "EPSG:3776", 
        "pixel_size": 0.1
    },
    # "vancouver": {
    #     "url": "https://opendata.vancouver.ca/explore/dataset/orthophoto-imagery-2015/api",
    #     "crs": "EPSG:26910",  # NAD83 / UTM zone 10N
    #     "pixel_size": 0.1
    # },
    "new_york": {
        "url": "https://orthos.its.ny.gov/arcgis/rest/services/wms/Latest/MapServer/export?",
        "crs": "EPSG:2263",  # NAD83 / New York Long Island (ftUS)
        "pixel_size": 0.15
    },
    "washington_dc": {
        "url": "https://imagery.dcgis.dc.gov/dcgis/rest/services/Ortho/Ortho_2023/ImageServer/exportImage",
        "crs": "EPSG:26985",  # NAD83(HARN) / Maryland
        "pixel_size": 0.1
    },
    # "bloomington": {
    #     'url':"https://gis.hennepin.us/arcgis/rest/services/Imagery/UTM_Aerial_2022/MapServer/export?",
    #     "crs": "EPSG:26915",  
    #     "pixel_size": 0.1
    #     },
    #"pittsburgh": {
    #    "url": "https://imagery.pasda.psu.edu/arcgis/rest/services/PEMAImagery2021_2023/MapServer/export?",  
    #    "crs": "EPSG:3857",  # Web Mercator
    #    "pixel_size": 0.1  # 3 inch ≈ 0.0762 meters
    #    },
    "charlottesville": {
            "url":"https://vginmaps.vdem.virginia.gov/arcgis/rest/services/VBMP_Imagery/MostRecentImagery_WGS/MapServer/export",
            "crs": "EPSG:3857",
            "pixel_size": 0.1  # Approximate pixel size in meters
        },
    # "bloomington": {
    #     "url": "https://imageserver.gisdata.mn.gov/cgi-bin/wms?",
    #     "crs": "EPSG:26915",  # NAD83 / UTM zone 15N
    #     "pixel_size": 0.1  # Approximate pixel size in meters
    # },
    "seattle": {
        "url":"https://gis.seattle.gov/image/rest/services/BaseMaps/WM_Aerial/MapServer/export?",
        "crs": "EPSG:3857",
        "pixel_size": 0.1  # Approximate pixel size in meters
        }
        ,
    "montreal": {
            "url": "https://gociteweb.longueuil.quebec/arcgis/rest/services/image/Orthophoto2022_WebMercator/MapServer/export",
            "crs": "EPSG:3857",
            "pixel_size": 0.1  # Approximate pixel size in meters
        },
    "columbus": {
        "url": "https://maps.columbus.gov/arcgis/rest/services/Imagery/Imagery2023/MapServer/export",
        "crs": "EPSG:3857",
        "pixel_size": 0.1  # Approximate pixel size in meters
    }
    # "cambridge": {
    #     "url": "https://tiles.arcgis.com/tiles/hGdibHYSPO59RG1h/arcgis/rest/services/orthos2023/MapServer/export",
    #     "crs": "EPSG:3857",  # Web Mercator (as shown in the service description)
    #     "pixel_size": 0.15  # 15cm resolution as mentioned in the service description
    # }
}

def load_tree_locations(csv_path):
    """
    Load tree locations from CSV file
    
    Args:
        csv_path: Path to CSV file with columns IDX, SHAPE_LNG, SHAPE_LAT, GENUS, TAXONOMY_ID
    
    Returns:
        geopandas.GeoDataFrame: Tree locations with geometry
    """
    df = pd.read_csv(csv_path)
    
    # Create point geometries from lat/lng
    geometry = [Point(lng, lat) for lng, lat in zip(df['SHAPE_LNG'], df['SHAPE_LAT'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    
    return gdf

def get_city_bounds(tree_locations, buffer_km=2.0):
    """
    Get bounding box for all tree locations with buffer
    
    Args:
        tree_locations: GeoDataFrame of tree locations
        buffer_km: Buffer in kilometers around tree locations
    
    Returns:
        tuple: (minx, miny, maxx, maxy) in WGS84
    """
    # Buffer in degrees (approximate)
    buffer_deg = buffer_km / 111.0  # roughly 111 km per degree
    bounds = tree_locations.total_bounds
    
    return (bounds[0] - buffer_deg, bounds[1] - buffer_deg, 
            bounds[2] + buffer_deg, bounds[3] + buffer_deg)

def split_into_grid_cells(tree_locations, cell_size_meters=100):
    """
    Split tree locations into grid cells of specified size, only for cells containing trees.
    
    Args:
        tree_locations: GeoDataFrame of tree locations in WGS84
        cell_size_meters: Size of grid cells in meters
    
    Returns:
        list: List of (GeoDataFrame, bounds) tuples for each grid cell
    """
    # Transform to a projected CRS for accurate distance measurements
    center_lon = tree_locations.geometry.x.mean()
    utm_zone = int((center_lon + 180) / 6) + 1
    utm_crs = f'EPSG:{32600 + utm_zone}'  # WGS84 UTM zone
    
    # Transform to UTM
    tree_locations_utm = tree_locations.to_crs(utm_crs)
    
    # Calculate grid cell indices for each tree
    tree_locations_utm = tree_locations_utm.copy()
    tree_locations_utm['grid_x'] = (tree_locations_utm.geometry.x // cell_size_meters).astype(int)
    tree_locations_utm['grid_y'] = (tree_locations_utm.geometry.y // cell_size_meters).astype(int)
    
    # Group by unique grid cell
    grouped = tree_locations_utm.groupby(['grid_x', 'grid_y'])
    
    # Limit to cells that contain more than 5 trees
    grouped = grouped.filter(lambda x: len(x) > 20)
    grid_cells = []
    for (grid_x, grid_y), group in grouped.groupby(['grid_x', 'grid_y']):
        cell_minx = grid_x * cell_size_meters
        cell_miny = grid_y * cell_size_meters
        cell_maxx = cell_minx + cell_size_meters
        cell_maxy = cell_miny + cell_size_meters
        
        # Transform back to WGS84 for imagery download
        cell_bounds = transform_bounds(utm_crs, 'EPSG:4326',
                                       cell_minx, cell_miny, cell_maxx, cell_maxy)
        # Transform group back to WGS84
        group_wgs84 = group.to_crs('EPSG:4326')
        grid_cells.append((group_wgs84, cell_bounds))
    
    return grid_cells

def download_arcgis_imagery(bounds, imagery_config, output_path, size=(2048, 2048)):
    """
    Download imagery from ArcGIS Rest Server
    
    Args:
        bounds: tuple of (minx, miny, maxx, maxy) in WGS84
        imagery_config: dict with url, crs, pixel_size
        output_path: path to save the downloaded image
        size: tuple of (width, height) for the output image
    
    Returns:
        str: path to downloaded image file
    """
    # Transform bounds to target CRS
    transformer = Transformer.from_crs('EPSG:4326', imagery_config['crs'], always_xy=True)
    minx, miny = transformer.transform(bounds[0], bounds[1])
    maxx, maxy = transformer.transform(bounds[2], bounds[3])
    
    # Construct ArcGIS REST API request
    params = {
        'bbox': f'{minx},{miny},{maxx},{maxy}',
        'bboxSR': imagery_config['crs'].split(':')[1],  # Extract EPSG code
        'imageSR': imagery_config['crs'].split(':')[1],
        'size': f'{size[0]},{size[1]}',
        'format': 'tiff',
        'pixelType': 'U8',
        'noData': '255,255,255',
        'interpolation': 'RSP_BilinearInterpolation',
        'f': 'image'
    }
    
    try:
        response = requests.get(imagery_config['url'], params=params, timeout=60)
        response.raise_for_status()
        
        # Check the actual response content to determine file format
        # PNG files start with the signature: 89 50 4E 47 0D 0A 1A 0A
        # TIFF files start with either: 49 49 2A 00 (little-endian) or 4D 4D 00 2A (big-endian)
        content_bytes = response.content[:8]
        
        # Determine file extension based on file signature
        if content_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
            file_extension = '.png'
            detected_format = 'PNG'
        elif content_bytes.startswith(b'II*\x00') or content_bytes.startswith(b'MM\x00*'):
            file_extension = '.tif'
            detected_format = 'TIFF'
        elif content_bytes.startswith(b'\xff\xd8\xff'):
            file_extension = '.jpg'
            detected_format = 'JPEG'
        else:
            # Default to .tif if signature is unclear
            file_extension = '.tif'
            detected_format = 'Unknown'
            print(f"Warning: Unknown file signature {content_bytes.hex()}, defaulting to .tif")
        
        # Update output path with correct extension
        base_path = os.path.splitext(output_path)[0]
        actual_output_path = base_path + file_extension
        
        # Save the image
        os.makedirs(os.path.dirname(actual_output_path), exist_ok=True)
        with open(actual_output_path, 'wb') as f:
            f.write(response.content)
            
        print(f"Downloaded imagery: {actual_output_path} (Detected format: {detected_format})")
        return actual_output_path
        
    except requests.exceptions.RequestException as e:
        print(f"Failed to download imagery: {e}")
        return None

def transform_tree_locations(tree_locations, target_crs):
    """
    Transform tree locations to target CRS
    
    Args:
        tree_locations: GeoDataFrame in WGS84
        target_crs: target CRS string (e.g., 'EPSG:3776')
    
    Returns:
        GeoDataFrame: transformed tree locations
    """
    return tree_locations.to_crs(target_crs)

def create_annotations_from_trees(tree_locations, image_path, image_bounds, pixel_size):
    """
    Create annotations DataFrame from tree locations
    
    Args:
        tree_locations: GeoDataFrame of tree locations in image CRS
        image_path: path to the image file
        image_bounds: bounds of the image in CRS coordinates
        pixel_size: pixel size in CRS units
    
    Returns:
        pandas.DataFrame: annotations in MillionTrees format
    """
    # Filter trees within image bounds
    image_box = box(image_bounds[0], image_bounds[1], image_bounds[2], image_bounds[3])
    trees_in_image = tree_locations[tree_locations.intersects(image_box)]
    
    if len(trees_in_image) == 0:
        print("No trees found in image bounds")
        return pd.DataFrame()
    
    # Convert to image coordinates
    trees_in_image = trees_in_image.copy()
    trees_in_image['x'] = (trees_in_image.geometry.x - image_bounds[0]) / pixel_size
    trees_in_image['y'] = (image_bounds[3] - trees_in_image.geometry.y) / pixel_size  # Flip Y
    
    # Create annotations dataframe
    annotations = pd.DataFrame({
        'image_path': os.path.basename(image_path),
        'x': trees_in_image['x'],
        'y': trees_in_image['y'],
        'label': 'Tree',
        'source': 'AutoArborist',
        'score': 1.0,
        'genus': trees_in_image['GENUS'].fillna('unknown'),
        'taxonomy_id': trees_in_image['TAXONOMY_ID'].fillna(0)
    })
    
    return annotations

def process_large_imagery(image_path, annotations, chip_size=2048, overlap=0.1):
    """
    Process large imagery by splitting into chips and updating annotations
    
    Args:
        image_path: path to large image
        annotations: DataFrame of annotations
        chip_size: size of image chips in pixels
        overlap: overlap between chips (0.0 to 1.0)
    
    Returns:
        pandas.DataFrame: updated annotations for image chips
    """
    # Check image size
    with rasterio.open(image_path) as src:
        height, width = src.height, src.width
        
    if height <= chip_size and width <= chip_size:
        print(f"Image {image_path} is small enough, no splitting needed")
        return annotations
    
    print(f"Splitting large image {image_path} ({width}x{height}) into {chip_size}px chips")
    
    # Use deepforest split_raster function
    output_dir = os.path.join(os.path.dirname(image_path), 'chips')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a temporary annotations file
    temp_annotations_path = os.path.join(output_dir, 'temp_annotations.csv')
    annotations.to_csv(temp_annotations_path, index=False)
    
    try:
        # Split the raster and annotations
        split_annotations = split_raster(
            path_to_raster=image_path,
            annotations_file=temp_annotations_path,
            base_dir=output_dir,
            patch_size=chip_size,
            patch_overlap=overlap
        )
        
        # Clean up temp file
        if os.path.exists(temp_annotations_path):
            os.remove(temp_annotations_path)
            
        return split_annotations
        
    except Exception as e:
        print(f"Error splitting raster: {e}")
        # Clean up temp file
        if os.path.exists(temp_annotations_path):
            os.remove(temp_annotations_path)
        return annotations

def create_sample_plots(annotations, output_dir, city_name, max_samples=3):
    """
    Create sample plots of annotations for quality control
    
    Args:
        annotations: DataFrame of annotations
        output_dir: directory to save plots
        city_name: name of the city for plot titles
        max_samples: maximum number of sample plots to create
    """
    if len(annotations) == 0:
        print(f"No annotations to plot for {city_name}")
        return
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get unique image paths and sample up to max_samples
    unique_images = annotations['image_path'].unique()
    if len(unique_images) > max_samples:
        sample_images = np.random.choice(unique_images, max_samples, replace=False)
    else:
        sample_images = unique_images
    
    print(f"Creating {len(sample_images)} sample plots for {city_name}")
    
    for i, image_path in enumerate(sample_images):
        try:
            # Get annotations for this image
            image_annotations = annotations[annotations['image_path'] == image_path].copy()
            
            # Set root directory for plotting
            image_annotations['root_dir'] = output_dir
            
            # Create plot
            plt.figure(figsize=(12, 8))
            plot_annotations(image_annotations, radius=10)
            plt.title(f"{city_name.title()} - Sample {i+1} ({len(image_annotations)} trees)")
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"{city_name}_sample_{i+1}.png"
            plot_path = os.path.join(plots_dir, plot_filename)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved plot: {plot_path}")
            
        except Exception as e:
            print(f"  Error creating plot for {image_path}: {e}")
            plt.close()  # Close any open figure

def process_city(city_name, csv_path, output_dir, create_plots=True):
    """
    Process a single city: download imagery and create annotations
    
    Args:
        city_name: name of the city (must be in IMAGERY_SOURCES)
        csv_path: path to CSV file with tree locations
        output_dir: directory to save outputs
        create_plots: whether to create sample plots
    
    Returns:
        tuple: (success: bool, num_trees: int, annotations_path: str)
    """
    if city_name.lower() not in IMAGERY_SOURCES:
        print(f"No imagery source configured for {city_name}")
        return False, 0, None
    
    try:
        # Load tree locations
        print(f"Processing {city_name}...")
        tree_locations = load_tree_locations(csv_path)
        print(f"Loaded {len(tree_locations)} tree locations")
        
        if city_name.lower() == "montreal":
            # remove the trees that are outside the bounds of the image
            # transform the tree locations to the image CRS
            proj_tree_locations = tree_locations.to_crs(IMAGERY_SOURCES['montreal']['crs'])
            tree_locations = tree_locations[proj_tree_locations.intersects(box(-8185982.510891517, 5686259.959149192, -8158840.010891517, 5726331.709149192))]

        if len(tree_locations) == 0:
            print(f"No tree locations found in {csv_path}")
            return False, 0, None
        
        # Get imagery configuration
        imagery_config = IMAGERY_SOURCES[city_name.lower()]
        
        # Split into grid cells
        grid_cells = split_into_grid_cells(tree_locations)
        print(f"Split into {len(grid_cells)} grid cells")
        
        all_annotations = []
        
        # Process each grid cell
        for i, (cell_trees, cell_bounds) in enumerate(grid_cells):
            print(f"Processing grid cell {i+1}/{len(grid_cells)} with {len(cell_trees)} trees")
            
            # Download imagery for this cell
            image_path = os.path.join(output_dir, f"{city_name}_cell_{i}_imagery.tif")
            downloaded_path = download_arcgis_imagery(cell_bounds, imagery_config, image_path)
            
            try:
                with rasterio.open(downloaded_path) as src:
                    image = src.read()
                    if image.shape[0] == 4:
                        image = image[:3, :, :]  # Keep only first 3 bands (RGB)
                        # Create a new profile with updated band count
                        profile = src.profile.copy()
                        profile['count'] = 3  # Update to 3 bands
                        
                        with rasterio.open(downloaded_path, 'w', **profile) as dst:
                            dst.write(image)
                    
                    image_bounds = src.bounds
                    pixel_size = src.res[0]
                    crs = src.crs
                    print(f"Image dimensions: {src.width}x{src.height}, bounds: {image_bounds}")
            except Exception as e:
                print(f"Error reading downloaded image: {e}")
                continue

            if not downloaded_path:
                print(f"Failed to download imagery for cell {i}")
                continue
            
            if src.crs is None:
                # transform the cell bounds to the image CRS
                image_bounds = transform_bounds( "EPSG:4326",imagery_config['crs'],
                                       cell_bounds[0], cell_bounds[1], cell_bounds[2], cell_bounds[3])
                pixel_size = (image_bounds[3] - image_bounds[1])/2048

            # Transform tree locations to image CRS
            cell_trees_proj = transform_tree_locations(cell_trees, imagery_config['crs'])
                
            # Create annotations for this cell
            cell_annotations = create_annotations_from_trees(
                cell_trees_proj, downloaded_path, image_bounds, pixel_size)
            
            if len(cell_annotations) > 0:
                all_annotations.append(cell_annotations)
        
        if not all_annotations:
            print(f"No trees found within image bounds for {city_name}")
            return False, 0, None
        
        # Combine all annotations
        annotations = pd.concat(all_annotations, ignore_index=True)
        
        # Save annotations
        annotations_path = os.path.join(output_dir, f"{city_name.lower()}_annotations.csv")
        annotations.to_csv(annotations_path, index=False)
        
        print(f"Created annotations for {len(annotations)} trees in {city_name}")
        print(f"Saved to: {annotations_path}")
        
        # Create sample plots if requested
        if create_plots:
            create_sample_plots(annotations, output_dir, city_name)
        
        return True, len(annotations), annotations_path
        
    except Exception as e:
        print(f"Error processing {city_name}: {e}")
        return False, 0, None

def cli():
    parser = argparse.ArgumentParser(description="AutoArborist: Download imagery and create MillionTrees annotations for a city.")
    parser.add_argument("csv_path", type=str, help="Path to the city tree locations CSV file")
    parser.add_argument("--output_dir", type=str, default="/orange/ewhite/DeepForest/AutoArborist/downloaded_imagery", help="Directory to save outputs")
    parser.add_argument("--no-plots", action="store_true", help="Skip creating sample plots")
    args = parser.parse_args()

    # Extract city name from filename
    csv_file = os.path.basename(args.csv_path)
    city_name = csv_file.replace('Trees.csv', '').replace('_train.csv', '').replace('_test.csv', '').replace('_sample.csv', '')
    if city_name.lower().startswith('calgary'):
        city_name = 'calgary'
    elif city_name.lower().startswith('edmonton'):
        city_name = 'edmonton'
    elif city_name.lower().startswith('vancouver'):
        city_name = 'vancouver'
    elif city_name.lower().startswith('new_york') or city_name.lower().startswith('newyork'):
        city_name = 'new_york'
    elif city_name.lower().startswith('washington') or city_name.lower().startswith('dc'):
        city_name = 'washington_dc'
    elif city_name.lower().startswith('hennepin'):
        city_name = 'minneapolis'
    elif city_name.lower().startswith('pittsburgh'):
        city_name = 'pittsburgh'

    print(f"Processing city: {city_name}")

    success, num_trees, annotations_path = process_city(city_name, args.csv_path, args.output_dir, create_plots=not args.no_plots)
    if success:
        print(f"Success: {num_trees} trees, annotations at {annotations_path}")
    else:
        print("Processing failed.")

def main():
    """
    Main function to process all cities with tree location data
    """
    # Directory containing tree location CSV files
    tree_locations_dir = "/orange/ewhite/DeepForest/AutoArborist/auto_arborist_cvpr2022_v0.27/tree_locations"
    output_dir = "/orange/ewhite/DeepForest/AutoArborist/downloaded_imagery"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files
    csv_files = [f for f in os.listdir(tree_locations_dir) if f.endswith('.csv')]
    
    results = []
    total_trees = 0
    successful_cities = []
    
    for csv_file in csv_files:
        # Extract city name from filename
        city_name = csv_file.replace('Trees.csv', '').replace('_train.csv', '').replace('_test.csv', '').replace('_sample.csv', '')
        
        # Handle different filename patterns
        if city_name.lower().startswith('calgary'):
            city_name = 'calgary'
        elif city_name.lower().startswith('edmonton'):
            city_name = 'edmonton'
        elif city_name.lower().startswith('vancouver'):
            city_name = 'vancouver'
        elif city_name.lower().startswith('new_york') or city_name.lower().startswith('newyork'):
            city_name = 'new_york'
        elif city_name.lower().startswith('washington') or city_name.lower().startswith('dc'):
            city_name = 'washington_dc'
        
        # Skip if already processed train/test files
        if '_train' in csv_file or '_test' in csv_file:
            continue
            
        csv_path = os.path.join(tree_locations_dir, csv_file)
        
        success, num_trees, annotations_path = process_city(city_name, csv_path, output_dir, create_plots=True)
        
        results.append({
            'city': city_name,
            'success': success,
            'num_trees': num_trees,
            'annotations_path': annotations_path
        })
        
        if success:
            total_trees += num_trees
            successful_cities.append(city_name)
    
    # Create summary report
    summary_path = os.path.join(output_dir, "AutoArborist_summary.md")
    with open(summary_path, 'w') as f:
        f.write("# AutoArborist Dataset Summary\n\n")
        f.write("Automated overhead imagery acquisition and annotation for tree locations.\n\n")
        f.write("## Overview\n\n")
        f.write(f"- **Total cities processed**: {len(csv_files)}\n")
        f.write(f"- **Successful cities**: {len(successful_cities)}\n")
        f.write(f"- **Total tree annotations**: {total_trees}\n\n")
        f.write("## Cities with Successful Imagery\n\n")
        
        for result in results:
            if result['success']:
                f.write(f"- **{result['city']}**: {result['num_trees']} trees\n")
        
        f.write("\n## Failed Cities\n\n")
        for result in results:
            if not result['success']:
                f.write(f"- **{result['city']}**: No imagery available or processing failed\n")
        
        f.write("\n## Dataset Format\n\n")
        f.write("The AutoArborist dataset follows the MillionTrees format with point annotations:\n\n")
        f.write("- `image_path`: Relative path to the image file\n")
        f.write("- `x`, `y`: Pixel coordinates of tree locations\n")
        f.write("- `label`: Always 'Tree'\n")
        f.write("- `source`: 'AutoArborist'\n")
        f.write("- `score`: 1.0 (ground truth)\n")
        f.write("- `genus`: Tree genus from original data\n")
        f.write("- `taxonomy_id`: Taxonomy ID from original data\n\n")
        f.write("## Imagery Sources\n\n")
        for city, config in IMAGERY_SOURCES.items():
            f.write(f"- **{city.title()}**: {config['url']}\n")
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed {len(successful_cities)} cities with {total_trees} total trees")
    print(f"Summary saved to: {summary_path}")
    
    return results

def create_combined_sample_plots(combined_df, output_dir, max_samples_per_city=2):
    """
    Create sample plots from the combined dataset, showing examples from each city
    
    Args:
        combined_df: Combined annotations DataFrame
        output_dir: Directory to save plots
        max_samples_per_city: Maximum number of sample plots per city
    """
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Group by city and create sample plots
    for city_name, city_df in combined_df.groupby('city'):
        print(f"Creating sample plots for {city_name}...")
        
        # Get unique image paths for this city
        unique_images = city_df['image_path'].unique()
        if len(unique_images) > max_samples_per_city:
            sample_images = np.random.choice(unique_images, max_samples_per_city, replace=False)
        else:
            sample_images = unique_images
        
        for i, image_path in enumerate(sample_images):
            try:
                # Get annotations for this image
                image_annotations = city_df[city_df['image_path'] == image_path].copy(deep=True)
                
                # Set root directory for plotting
                image_annotations.root_dir = output_dir
                
                # Plot annotations need relative paths to the image
                image_annotations['image_path'] = image_annotations['image_path'].apply(lambda x: os.path.basename(x))
                # Create plot
                plt.figure(figsize=(12, 8))
                plot_results(image_annotations, radius=10)
                plt.title(f"{city_name.title()} - Combined Sample {i+1} ({len(image_annotations)} trees)")
                plt.tight_layout()
                
                # Save plot
                plot_filename = f"{city_name}_combined_sample_{i+1}.png"
                plot_path = os.path.join(plots_dir, plot_filename)
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"  Saved plot: {plot_path}")
                
            except Exception as e:
                print(f"  Error creating plot for {image_path}: {e}")
                plt.close()
    
    print(f"All sample plots saved to: {plots_dir}")

if __name__ == "__main__":
    # Check if running as CLI or main function
    import sys
    if len(sys.argv) > 1:
        cli()
    else:
        main()

    # # Combine all the csv files into a single CSV for easier access
    output_dir = "/orange/ewhite/DeepForest/AutoArborist/downloaded_imagery"
    combined_csv_path = os.path.join(output_dir, "AutoArborist_combined_annotations.csv")

    all_annotations = []
    completed_csvs = glob.glob(os.path.join(output_dir, "*_annotations.csv"))

    # don't include AutoArborist_combined_annotations.csv
    completed_csvs = [csv_file for csv_file in completed_csvs if csv_file != combined_csv_path]

    for csv_file in completed_csvs:
        df = read_file(csv_file)
        df['city'] = os.path.basename(csv_file).replace('_annotations.csv', '')
        df["source"] = "Beery et al. 2022"
        # Full path to the image
        df['image_path'] = df.image_path.apply(lambda x: os.path.join(output_dir, x))
        all_annotations.append(df)

    combined_df = pd.concat(all_annotations, ignore_index=True)
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Combined annotations saved to: {combined_csv_path}")
    
    # Create sample plots from combined dataset
    print("\nCreating sample plots from combined dataset...")
    create_combined_sample_plots(combined_df, output_dir, max_samples_per_city=10)


# Calgary orthophoto reference:
# https://www.arcgis.com/apps/mapviewer/index.html?webmap=823b8c06c5544c1b825c7dd5da96d35a