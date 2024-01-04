import glob
import os
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box
import pandas as pd
import shutil
from deepforest.utilities import read_file, crop_raster

def Siberia_polygons():
    shps = glob.glob("/blue/ewhite/DeepForest/Siberia/vanGeffen-etal_2021b_shapefiles_allfiles/vanGeffen_et_al_SiDroForest_Individual_Polygon_Labelled/*.shp")
    annotations = []
    cropped_images = []
    # There were several .tif files that did not have the correct crs compared to the .shp, read them in and covert them to the correct crs
    for path in shps:
        print(path)
        ID = os.path.basename(path).split("_")[0]
        rgb_path = "/blue/ewhite/DeepForest/Siberia/orthos/{}_RGB_orthomosaic.tif".format(ID)
        df = gpd.read_file(path)
        df["image_path"] = rgb_path
        df["label"] = "Tree"
        df = read_file(input=df)
        src = rasterio.open(rgb_path)

        if src.count == 4:
            # Remove alpha channel
            new_rgb_path = "/blue/ewhite/DeepForest/Siberia/orthos/{}_RGB_orthomosaic_corrected.tif".format(ID)
            with rasterio.open(rgb_path) as src:
                kwargs = src.meta.copy()
                kwargs.update(count=3)
                with rasterio.open(new_rgb_path, 'w', **kwargs) as dst:
                    dst.write(src.read()[:3,:,:])
        elif src.crs != df.crs:
            dst_crs = df.crs
            new_rgb_path = "/blue/ewhite/DeepForest/Siberia/orthos/{}_RGB_orthomosaic_corrected.tif".format(ID)

            with rasterio.open(rgb_path) as src:
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds)
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': dst_crs,
                    'transform': transform,
                    'width': width,
                    'height': height
                })

                with rasterio.open(new_rgb_path, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.nearest)
        else:
            new_rgb_path = rgb_path
            
        
        df = df[df.Type=="Tree"]
        if df.empty:
            continue
        df["source"] = "Kruse et al. 2021"
        df["original_image"] = new_rgb_path

        # Crop to bounded area with a small buffer
        buffered_bounds = box(*df.total_bounds).bounds

        # Translate to crop coordinates, destroys the CRS
        df.geometry = df.geometry.translate(xoff=-buffered_bounds[0], yoff=-buffered_bounds[1])
        filename = cropped_raster = crop_raster(
            bounds=buffered_bounds,
            rgb_path=new_rgb_path,
            savedir="/blue/ewhite/DeepForest/Siberia/images/",
            filename=ID,
            driver="PNG"
            )
        df["image_path"] = os.path.basename(filename)
        cropped_images.append(cropped_raster)
        df.crs = None
        
        annotations.append(df)

    annotations = pd.concat(annotations)
    annotations["split"] = "train"
    annotations.to_csv("/blue/ewhite/DeepForest/Siberia/annotations.csv")

    # Move all data to the common images dir
    for image_path in annotations.image_path.unique():
        src = os.path.join("/blue/ewhite/DeepForest/Siberia/images/", image_path)
        dst = os.path.join("/blue/ewhite/DeepForest/MillionTrees/images/", image_path)
        shutil.copy(src, dst) 