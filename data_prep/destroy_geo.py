import rasterio
import geopandas

rgb_path = "/Users/benweinstein/Downloads/2022_HARV_7_731000_4713000_image.tif"

# Read RGB image
src = rasterio.open(rgb_path)
print("The input crs is: ", src.crs)

# Read shapefile
shp_path = "/Users/benweinstein/Downloads/sample.shp"
gdf = geopandas.read_file(shp_path)
print("The shapefile crs is: ", gdf.crs)
gdf.head()

def geo_to_image_coordinates(gdf, image_bounds, image_resolution):
    """
    Convert from projected coordinates to image coordinates
    Args:
        gdf: a pandas type dataframe with columns: name, xmin, ymin, xmax, ymax. Name is the relative path to the root_dir arg.
        image_bounds: bounds of the image
        image_resolution: resolution of the image
    Returns:
        gdf: a geopandas dataframe with the transformed to image origin. CRS is removed
        """
    
    # unpack image bounds
    left, bottom, right, top = image_bounds
    gdf.geometry = gdf.geometry.translate(xoff=-left, yoff=-top)

    # Rasterio origin is upper left, so flip y axis
    gdf.geometry = gdf.geometry.scale(xfact=1/image_resolution, yfact=1/image_resolution, origin=(0,0))
    
    gdf.crs = None  

    return gdf

# Destroy the shapefile crs by translating to image origin
translated_gdf = geo_to_image_coordinates(gdf, src.bounds, src.res[0])

# Save to file
translated_gdf.to_file("/Users/benweinstein/Downloads/sample_translated.shp")

# Save a png of the image to file without CRS
new_meta = src.meta.copy()
new_meta['crs'] = None
new_meta["driver"] = "PNG"
new_meta["transform"] = None

with rasterio.open("/Users/benweinstein/Downloads/2022_HARV_7_731000_4713000_image_translated.png", 'w', **new_meta) as dst:
    dst.write(src.read())