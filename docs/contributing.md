# Contributing

## Data

The essential data are

1. **Airborne imagery**

    Drone or piloted aircraft preferred, we are not yet convinced about tree segmentation from satellite sources, but open to discussion. Preferably in .tif format with georeferencing.

2. **Tree annotations**

    Point, box, or polygon annotations of trees in the imagery. In shapefile format with matching georeferencing as the airborne imagery.

The spatial location of the points will be destroyed, such that the point locations will only be relative to the image crop. This will prevent any user from being able to use the data for analysis outside of the benchmark. All species, DBH and other metadata will be removed. For the images, if the geospatial location is the problem, as is it with many datasets, let the provider know that we are destroying the geospatial position, such that we crop images into pieces and remove the coordinate reference system and make any tree annotations relative to the 0,0 image origin, this way we are not releasing any geolocated data that might have privacy issues.

For open source data in which authors don't have concerns about privacy, the best way to contribute is to make data available on [Zenodo](https://zenodo.org/), and then make an issue in this repo documenting the location of the data. Your Zenodo record will now have a DOI that is citable. We are sensitive to the contributions and efforts of the hundreds of field researchers that make data collection possible. Authorship will be extended to any team with unpublished data.

## What does a successful dataset look like?

We welcome any geometric representation of trees in airborne imagery. Points, polygons, or boxes. Usually we ask for a shapefile or text coordinates of trees and a corresponding .tif geospatial file of airborne imagery. When you overlay the two, the data should look coherent. There has been a narrow view of the task that has been overly constrained by off-the-shelf architectures, rather than the essential nature of the task. Tree localization, counting, and crown detection are all interrelated tasks using geometric representations of trees. We should not design benchmarks around current model architectures, we should put the problem first and build architectures that meet that need.

![Image Placeholder](public/street_trees.png)
![Image Placeholder](public/polygon_example.png)
![Image Placeholder](public/HARV_037.png)

## How will the data be shared?

After we work with an author to find a suitable data sharing agreement, we will remove the spatial information from the images and create a Zenodo record to document a train/test split for the benchmark. A manuscript, which all contributors are invited to join, will be published outlining the strengths, limitations, and potential uses of the dataset. The working document describing technical details of evaluation is still in its infancy is [here](https://docs.google.com/document/d/1K6G1tcdTuAv3FgGiDWq5QhO-kSoBrxzTiic5jH1CZF4/edit?usp=sharing).

## Dataset structure
The MillionTrees structure is a tabular dataframe with each annotation as a row, and the following columns.

image_path: the full path to the image on disk
source: the abbreviated citation name used to identify the dataset, e.g. 'Weinstein et al. 2021'
geometry: a shapely geometry column in wkt with the annotation coordinates in the image coordinate reference, not the geospatial coordinate system. Top left of the image is 0,0. For example, 'POINT (115 90)'
'existing_split': If the annotation belongs to a train or test split. This is optional can be excluded.


There are useful utilities in the [DeepForest](https://deepforest.readthedocs.io/en/stable/user_guide/01_Reading_data.html) python package for converting from geographic to image coordinate systems. 

### Tiling large .tif files

Many datasets come with large orthomosaics that are too memory intensive to fit into model training or prediction as a single chunk. Deepforest has a utility, deepforest.preprocess.split_tile that takes in annotations and a path to an image and returns the annotations for each tiled piece of the image.

```python```
# Example: Splitting large images and annotation files into smaller tiles for training
from pathlib import Path
import pandas as pd
import os
import rasterio
import geopandas as gpd
from deepforest.utilities import read_file
from deepforest.preprocess import split_raster

def split_images_and_annotations(
    annotations_csv: str,                # Path to annotation CSV (see table format above)
    output_dir: str,                     # Directory to save tiled images and annotations
    patch_size: int = 500                # Size of split patches in pixels
) -> pd.DataFrame:
    """
    Split large images and corresponding annotations into smaller tiles for efficient training.
    Saves new tiled image crops and an updated annotations CSV in output_dir.

    Args:
        annotations_csv: Path to the preprocessed MillionTrees-format CSV.
        output_dir: Directory where cropped images and tile annotations will be saved.
        patch_size: Size (in pixels) for tiles.
    
    Returns:
        DataFrame of updated (tiled) annotations.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(annotations_csv)
    unique_images = df["image_path"].unique()
    all_tile_annotations = []

    for img_path in unique_images:
        # Select annotations for this image
        img_annotations = df[df["image_path"] == img_path].copy(deep=True)
        
        # Load image data as numpy array
        with rasterio.open(img_path) as src:
            image_array = src.read()
        
        # Optionally, convert image-relative paths
        img_annotations["image_path"] = img_annotations["image_path"].apply(os.path.basename)
        
        # Convert DataFrame to geodataframe, map annotations to new tile crops
        gdf = read_file(img_annotations, root_dir=os.path.dirname(img_path), label="Tree")
        gdf = gpd.GeoDataFrame(gdf)
        
        # Use DeepForest utility to split raster and annotations to tiles
        split_tiles = split_raster(
            image_name=os.path.basename(img_path),
            annotations_file=gdf,
            numpy_image=image_array,
            patch_size=patch_size,
            allow_empty=False,
            save_dir=out_dir,
            root_dir=out_dir
        )
        all_tile_annotations.append(split_tiles)
    
    # Concatenate annotation DataFrames for all tiles
    tile_annotations = pd.concat(all_tile_annotations, ignore_index=True)
    
    # Update image_path to full (or relative) location in output_dir
    tile_annotations["image_path"] = tile_annotations["image_path"].apply(lambda x: str(out_dir / x))
    
    # Set a new source string or retain existing if preferred
    tile_annotations["source"] = "YourSourceNameHere"
    
    # Save updated annotations CSV
    annotations_out_csv = out_dir / "annotations.csv"
    tile_annotations.to_csv(annotations_out_csv, index=False)
    
    return tile_annotations
```