import geopandas as gpd
import pandas as pd
import os
import glob
from shapely.geometry import Polygon
from deepforest.utilities import read_file
from deepforest.preprocess import split_raster
import rasterio

def _gpkg_to_tif_name(gpkg_basename: str, raster_dir: str) -> str:
    """
    Map a Ground_Truth file basename like '12_Validation_GT.gpkg'
    to the corresponding raster basename like '12_Validation_dop20_False_Color.tif'
    located in raster_dir.
    """
    # Keep the "<idx>_<Split>" prefix; append expected raster suffix
    # Example:
    #  12_Validation_GT.gpkg -> 12_Validation_dop20_False_Color.tif
    prefix = gpkg_basename.replace("_GT.gpkg", "")
    tif_name = f"{prefix}_dop20_False_Color.tif"
    tif_path = os.path.join(raster_dir, tif_name)
    return tif_path

def _read_polygons_fix(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure only Polygon geometries and drop or explode MultiPolygons. Convert Z to 2D if needed."""
    # Explode MultiPolygons into individual polygons if any
    if "MultiPolygon" in gdf.geometry.geom_type.unique():
        # explode returns GeoDataFrame; ignore_index for a clean index in newer geopandas
        try:
            gdf = gdf.explode(index_parts=False)
        except TypeError:
            # for older geopandas versions without index_parts
            gdf = gdf.explode()
            gdf = gdf.reset_index(drop=True)

    # Keep only polygons
    gdf = gdf[gdf.geometry.geom_type == "Polygon"]
    # Convert Z polygons to 2D
    def to_2d(geom):
        if geom is None:
            return geom
        if hasattr(geom, "has_z") and geom.has_z:
            return Polygon([(x, y) for x, y, *_ in geom.exterior.coords])
        return geom
    gdf = gdf.set_geometry(gdf.geometry.apply(to_2d))
    return gdf

def build_annotations() -> pd.DataFrame:
    base_dir = "/orange/ewhite/DeepForest/Schutte_Germany/ITCD_Urban_Berlin_Osnabrueck"
    crop_dir = "/orange/ewhite/DeepForest/Schutte_Germany/crops"
    os.makedirs(crop_dir, exist_ok=True)
    cities = ["Berlin", "Osnabrueck"]
    all_annotations = []

    for city in cities:
        city_dir = os.path.join(base_dir, city)
        gt_dir = os.path.join(city_dir, "Ground_Truth")
        raster_dir = os.path.join(city_dir, "Rasterdata")

        gpkg_files = sorted(glob.glob(os.path.join(gt_dir, "*.gpkg")))
        if not gpkg_files:
            print(f"No GeoPackage files found in {gt_dir}")
            continue

        for gpkg in gpkg_files:
            try:
                gdf = gpd.read_file(gpkg)
            except Exception as e:
                print(f"Could not read {gpkg}: {e}")
                continue

            gdf = _read_polygons_fix(gdf)
            if len(gdf) == 0:
                print(f"No polygon annotations found in {gpkg}, skipping.")
                continue

            gpkg_basename = os.path.basename(gpkg)
            tif_path = _gpkg_to_tif_name(gpkg_basename, raster_dir)
            if not os.path.exists(tif_path):
                print(f"Missing raster for {gpkg_basename}: {tif_path}")
                continue

            # Use filename (basename) for read_file root_dir handling
            gdf["image_path"] = os.path.basename(tif_path)
            gdf["label"] = "tree"

            # Standardize to expected annotation format
            ann = read_file(gdf, root_dir=raster_dir)
            ann["source"] = "Schütte et al. 2025"
            ann.root_dir = raster_dir
            
            # match crs from raster to annotations
            with rasterio.open(tif_path) as src:
                raster_crs = src.crs

            ann.crs = raster_crs
            # Split raster into smaller chunks and get cropped annotations
            print(f"Splitting {tif_path} into smaller chunks...")
            ann.reset_index(drop=True, inplace=True)
            try:
                crop_annotations =split_raster(
                ann,
                path_to_raster=tif_path,
                save_dir=crop_dir,
                patch_size=1000,
                patch_overlap=0,
                allow_empty=False
            )
            except Exception as e:
                print(f"Error splitting {tif_path}: {e}")
                continue
            # Update image paths to full paths for cropped images
            crop_annotations["image_path"] = crop_annotations["image_path"].apply(
                lambda x: os.path.join(crop_dir, x)
            )
            
            all_annotations.append(crop_annotations)

    if not all_annotations:
        return pd.DataFrame()

    return pd.concat(all_annotations, ignore_index=True)

if __name__ == "__main__":
    annotations = build_annotations()
    output_csv = "/orange/ewhite/DeepForest/Schutte_Germany/annotations.csv"
    if annotations.empty:
        print("No annotations generated; nothing to write.")
    else:
        annotations.to_csv(output_csv, index=False)
        print(f"Annotations saved to {output_csv}")


