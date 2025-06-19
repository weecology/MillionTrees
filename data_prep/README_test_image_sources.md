# Image Source Testing and Visualization Tool

This tool tests image sources for the MillionTrees project by downloading sample imagery, running DeepForest tree detection, and overlaying results with ground truth data to help verify data source quality and alignment.

## Features

- **Image Server Testing**: Tests connectivity and functionality of various image servers
- **DeepForest Integration**: Runs tree detection on downloaded imagery
- **Ground Truth Overlay**: Visualizes predicted vs actual tree locations
- **Comprehensive Reporting**: Generates detailed reports with recommendations
- **Flexible Configuration**: JSON-based configuration for easy server management

## Usage

### Basic Usage

```bash
# Test all configured image sources
python test_image_sources.py

# Test with custom output directory
python test_image_sources.py --output_dir my_results

# Limit number of sources to test
python test_image_sources.py --max_sources 3

# Increase timeout for slow servers
python test_image_sources.py --timeout 60
```

### Configuration

The script uses `image_server_config.json` for server configurations. To add a new image server:

```json
{
  "image_servers": {
    "your_city": {
      "name": "Your City Name",
      "url": "https://your-server.com/arcgis/rest/services/...",
      "type": "arcgis_imageserver",
      "crs": "EPSG:3857",
      "test_bbox": [-long_min, lat_min, -long_max, lat_max],
      "resolution": 0.3,
      "description": "Description of your imagery"
    }
  }
}
```

## Output Structure

```
test_results/
├── images/           # Downloaded test images
├── annotations/      # Ground truth data
├── visualizations/   # Comparison plots
└── reports/         # Summary reports
```

## Requirements

- Python 3.7+
- deepforest
- requests
- rasterio
- geopandas
- pandas
- matplotlib
- pillow

Install dependencies:
```bash
pip install deepforest requests rasterio geopandas pandas matplotlib pillow
```

## Image Server Types Supported

### ArcGIS ImageServer
- High-resolution orthoimagery
- Real-time image processing
- Multiple output formats

### ArcGIS MapServer  
- Cached tile services
- Multiple layers support
- Fast response times

### WMS (Web Map Service)
- OGC standard protocol
- Cross-platform compatibility
- Flexible styling options

## Recommendations for Improved Flexibility

### 1. Configuration Management

**Current**: Hardcoded server configurations in Python
**Recommended**: JSON/YAML configuration files

Benefits:
- Easy to add new servers without code changes
- Version control for configurations
- Environment-specific configurations (dev/prod)

Example implementation:
```python
def load_server_config(config_file="image_server_config.json"):
    with open(config_file, 'r') as f:
        return json.load(f)
```

### 2. Fallback Server Strategy

**Current**: Single server per region
**Recommended**: Multiple fallback servers

```json
{
  "calgary": {
    "primary": "https://gis.calgary.ca/...",
    "fallbacks": [
      "https://backup-server.ca/...",
      "https://satellite-imagery.esri.com/..."
    ]
  }
}
```

### 3. Caching System

**Current**: Downloads every time
**Recommended**: Intelligent caching

```python
import hashlib
from pathlib import Path

def get_cached_image(bbox, server_config, cache_dir="cache"):
    bbox_hash = hashlib.md5(str(bbox).encode()).hexdigest()
    cache_file = Path(cache_dir) / f"{server_config['name']}_{bbox_hash}.png"
    
    if cache_file.exists():
        return str(cache_file)
    return None
```

### 4. Server Health Monitoring

**Current**: Test only when running script
**Recommended**: Continuous monitoring

```python
def monitor_server_health():
    """Run periodic health checks and update server status"""
    results = {}
    for server_name, config in IMAGE_SERVERS.items():
        status = test_image_server(server_name, config)
        results[server_name] = {
            "status": status["status"],
            "last_checked": datetime.now().isoformat(),
            "response_time": status.get("response_time", 0)
        }
    
    # Save to status file
    with open("server_status.json", "w") as f:
        json.dump(results, f, indent=2)
```

### 5. Dynamic Coordinate System Support

**Current**: Manual CRS specification
**Recommended**: Automatic CRS detection and transformation

```python
import pyproj

def transform_bbox(bbox, from_crs, to_crs):
    """Transform bounding box between coordinate systems"""
    transformer = pyproj.Transformer.from_crs(from_crs, to_crs)
    x1, y1 = transformer.transform(bbox[1], bbox[0])  # lat, lon -> x, y
    x2, y2 = transformer.transform(bbox[3], bbox[2])
    return [x1, y1, x2, y2]
```

### 6. Multi-Protocol Support

**Current**: ArcGIS REST only
**Recommended**: Multiple protocols

```python
class ImageServerClient:
    def __init__(self, server_config):
        self.config = server_config
        self.client = self._create_client()
    
    def _create_client(self):
        if self.config["type"] == "arcgis_rest":
            return ArcGISRestClient(self.config)
        elif self.config["type"] == "wms":
            return WMSClient(self.config)
        elif self.config["type"] == "wmts":
            return WMTSClient(self.config)
        else:
            raise ValueError(f"Unsupported server type: {self.config['type']}")
```

### 7. Tree Density-Based Area Selection

**Current**: Fixed test bounding boxes
**Recommended**: Dynamic area selection based on tree density

```python
def find_optimal_test_area(tree_df, target_tree_count=20):
    """Find area with optimal tree density for testing"""
    # Use spatial clustering to find dense tree areas
    from sklearn.cluster import DBSCAN
    
    coords = tree_df[['SHAPE_LNG', 'SHAPE_LAT']].values
    clustering = DBSCAN(eps=0.01, min_samples=target_tree_count).fit(coords)
    
    # Find largest cluster
    labels = clustering.labels_
    unique_labels = set(labels)
    
    best_cluster = None
    best_size = 0
    
    for label in unique_labels:
        if label == -1:  # noise
            continue
        cluster_points = coords[labels == label]
        if len(cluster_points) > best_size:
            best_size = len(cluster_points)
            best_cluster = cluster_points
    
    if best_cluster is not None:
        # Create bounding box around cluster with buffer
        min_x, min_y = best_cluster.min(axis=0)
        max_x, max_y = best_cluster.max(axis=0)
        buffer = 0.005  # ~500m buffer
        return [min_x - buffer, min_y - buffer, max_x + buffer, max_y + buffer]
    
    return None
```

### 8. Advanced Error Handling

**Current**: Basic exception handling
**Recommended**: Comprehensive error recovery

```python
import time
from functools import wraps

def retry_with_backoff(max_retries=3, backoff_factor=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.RequestException as e:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

@retry_with_backoff(max_retries=3)
def download_image_with_retry(server_name, server_config, bbox):
    return download_image_from_server(server_name, server_config, bbox)
```

### 9. Performance Optimization

**Current**: Sequential processing
**Recommended**: Parallel processing and async operations

```python
import asyncio
import aiohttp

async def test_servers_parallel(server_configs):
    """Test multiple servers in parallel"""
    async with aiohttp.ClientSession() as session:
        tasks = [
            test_server_async(session, name, config) 
            for name, config in server_configs.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

### 10. Quality Assessment Metrics

**Current**: Basic tree count comparison
**Recommended**: Comprehensive quality metrics

```python
def calculate_image_quality_metrics(image_path):
    """Calculate various image quality metrics"""
    image = cv2.imread(image_path)
    
    # Sharpness (Laplacian variance)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Contrast (standard deviation)
    contrast = gray.std()
    
    # Brightness (mean pixel value)
    brightness = gray.mean()
    
    # Color richness (unique colors)
    unique_colors = len(np.unique(image.reshape(-1, image.shape[2]), axis=0))
    
    return {
        "sharpness": sharpness,
        "contrast": contrast,
        "brightness": brightness,
        "color_richness": unique_colors
    }
```

## Server Status Dashboard

For production use, consider implementing a web dashboard to monitor server health:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Image Server Status</title>
    <meta http-equiv="refresh" content="300"> <!-- Auto-refresh every 5 minutes -->
</head>
<body>
    <h1>MillionTrees Image Server Status</h1>
    <div id="server-status">
        <!-- Dynamically populated -->
    </div>
    
    <script>
        // Fetch and display server status
        fetch('/api/server-status')
            .then(response => response.json())
            .then(data => updateStatusDisplay(data));
    </script>
</body>
</html>
```

## Contributing

To add a new image server:

1. Add server configuration to `image_server_config.json`
2. Add corresponding tree location data (if available) to `TREE_DATA_SOURCES`
3. Test the configuration with `python test_image_sources.py --max_sources 1`
4. Update documentation

## Troubleshooting

### Common Issues

**"Server returned non-image content"**
- Check server URL and parameters
- Verify bounding box coordinates
- Test server manually in browser

**"No trees detected by DeepForest"**
- Image may lack tree coverage
- Image quality may be poor
- Adjust bounding box to area with more trees

**"Ground truth data not loading"**
- Check CSV file format (requires SHAPE_LNG, SHAPE_LAT columns)
- Verify file path exists
- Check coordinate system alignment

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

This tool is part of the MillionTrees project. See project license for details.