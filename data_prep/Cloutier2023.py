import glob
import re
import pandas as pd
from preprocess_polygons import split_raster_with_polygons
import geopandas as gpd
from utilities import read_file
import os

def Cloutier2023():
    # Zone 3 is test, Zone 1 and 2 is train. Intentionally vary window size.

    drone_flights = glob.glob("/orange/ewhite/DeepForest/Cloutier2023/**/*-rgb-cog.tif",recursive=True)

    annotations = []
    for flight in drone_flights:
            # Each flight raster covers a single zone/date. Pair it with that
            # zone's polygons from the same date folder. Applying every zone's
            # polygons to every raster (the previous behaviour) projects e.g.
            # Z1 polygons onto the Z3 raster, where they land in the nodata
            # corner and produce all-black crops with overlaid annotations.
            zone = int(re.search(r"-z(\d)-rgb", os.path.basename(flight)).group(1))
            date_dir = os.path.dirname(os.path.dirname(os.path.dirname(flight)))
            gpkg = os.path.join(date_dir, f"Z{zone}_polygons.gpkg")

            zone_annotations = read_file(gpkg, rgb=flight)
            split_annotations = split_raster_with_polygons(zone_annotations, path_to_raster=flight, patch_overlap=0, patch_size=2000, allow_empty=False, base_dir="/orange/ewhite/DeepForest/Cloutier2023/images/")
            # Zone 3 is the held-out test zone; zones 1 and 2 are train.
            split_annotations["split"] = "test" if zone == 3 else "train"
            annotations.append(split_annotations)

    combined = pd.concat(annotations)
    combined["source"] = "Cloutier et al. 2023"


    # Make full path
    combined["image_path"] = "/orange/ewhite/DeepForest/Cloutier2023/images/" + combined["image_path"]

    combined.to_csv("/orange/ewhite/DeepForest/Cloutier2023/images/annotations.csv")
    print(f"Saved {len(combined)} annotations across {combined['image_path'].nunique()} tiles")



if __name__ == "__main__":
    Cloutier2023()

