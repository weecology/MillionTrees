"""
Scan original raster sources for each MillionTrees dataset and report GSD.
Uses rasterio to read projection/transform; only reports datasets with valid
georeferenced rasters.
"""
import glob
import os
import sys

import numpy as np

try:
    import rasterio
    from rasterio.crs import CRS
except ImportError:
    sys.exit("rasterio not installed")

# Map dataset name -> list of candidate paths/globs (first resolvable file wins)
DATASETS = {
    "BCI_2020": [
        "/orange/ewhite/DeepForest/BCI/BCI_50ha_2020_08_01_crownmap_raw/BCI_50ha_2020_08_01_global.tif",
    ],
    "BCI_2022": [
        "/orange/ewhite/DeepForest/BCI/BCI_50ha_2022_09_29_crownmap_raw/BCI_50ha_2022_09_29_global.tif",
    ],
    "NeonTreeEvaluation": [
        "/orange/idtrees-collab/NeonTreeEvaluation/evaluation/RGB/*.tif",
    ],
    "Krkonose_BileLabe": [
        "/orange/ewhite/DeepForest/Zenodo_15591546/HIGH_upper_tree_limit/plot_level/BL_high_orthoimagery.tif",
        "/orange/ewhite/DeepForest/Zenodo_15591546/**/plot_level/*.tif",
    ],
    "Schutte_Germany": [
        "/orange/ewhite/DeepForest/Schutte_Germany/ITCD_Urban_Berlin_Osnabrueck/*/Rasterdata/dop20_False_Color.tif",
        "/orange/ewhite/DeepForest/Schutte_Germany/**/*.tif",
    ],
    "SiDroneForest": [
        "/blue/ewhite/DeepForest/Siberia/orthos/*_RGB_orthomosaic.tif",
        "/blue/ewhite/DeepForest/Siberia/orthos/*.tif",
    ],
    "Quebec_Lefebvre": [
        "/orange/ewhite/DeepForest/Quebec_Lefebvre/Dataset/Photogrammetry_Products/**/*_rgb.cog.tif",
        "/orange/ewhite/DeepForest/Quebec_Lefebvre/Dataset/Photogrammetry_Products/**/*.tif",
    ],
    "OSBS_megaplot": [
        "/orange/ewhite/DeepForest/OSBS_megaplot/2025/mosaic_2025.tif",
    ],
    "DeepTrees_Halle": [
        "/orange/ewhite/DeepForest/Zenodo_19695972/tiles/tile_*.tif",
        "/orange/ewhite/DeepForest/Zenodo_19695972/*.tif",
    ],
    "Alejandro_Chile": [
        "/orange/ewhite/DeepForest/Alejandro_Chile/alejandro/mos_*.tif",
        "/orange/ewhite/DeepForest/Alejandro_Chile/alejandro/*.tif",
    ],
    "Takeshige2025": [
        "/orange/ewhite/DeepForest/takeshige2025/Ortho/*.tif",
    ],
    "UrbanLondon": [
        "/orange/ewhite/DeepForest/UrbanLondon/rgb/*.tif",
    ],
    "Jansen2023": [
        "/orange/ewhite/DeepForest/Jansen_2023/images/*_RGB.tif",
        "/orange/ewhite/DeepForest/Jansen_2023/images/*.tif",
    ],
    "Troles_bamberg": [
        "/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/images/*.tif",
    ],
    "Cloutier2023": [
        "/orange/ewhite/DeepForest/Cloutier2023/**/*.tif",
        "/orange/ewhite/DeepForest/Cloutier2023/*.tif",
    ],
    "Paracou_Ball": [
        "/orange/ewhite/DeepForest/paracou_ball/*.tif",
    ],
    "Wagner_Australia": [
        "/orange/ewhite/DeepForest/Wagner_Australia/Cropped plots/*.tif",
        "/orange/ewhite/DeepForest/Wagner_Australia/**/*.tif",
    ],
    "Araujo_2020": [
        "/orange/ewhite/DeepForest/Araujo_2020/Orthomosaic_WGS84_UTM20S.tif",
        "/orange/ewhite/DeepForest/Araujo_2020/*.tif",
    ],
    "Harz_Mountain": [
        "/orange/ewhite/DeepForest/Harz_Mountains/ML_TreeDetection_Harz/all_images/aerial_*.tif",
        "/orange/ewhite/DeepForest/Harz_Mountains/ML_TreeDetection_Harz/all_images/*.tif",
    ],
    "Ventura2022": [
        "/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/*.tif",
    ],
    "Hickman2021": [
        "/orange/ewhite/DeepForest/Hickman2021/RCD105_MA14_21_orthomosaic_*.tif",
        "/orange/ewhite/DeepForest/Hickman2021/*.tif",
    ],
    "Tonga": [
        "/orange/ewhite/DeepForest/Tonga/Kolovai-Trees-20180108.tif",
    ],
    "Kaggle_LiDAR_RGB": [
        "/orange/ewhite/DeepForest/Kaggle_LiDAR_RGB/ortho/*.tif",
    ],
    "SelvaBox": [
        "/orange/ewhite/DeepForest/SelvaBox/images/*.tif",
    ],
    "BohlmanBCI": [
        "/orange/ewhite/DeepForest/BohlmanBCI/*.tif",
    ],
    "Puliti_2022": [
        "/orange/ewhite/DeepForest/Puliti_2022/*.tif",
    ],
    "HemmingSchroeder": [
        "/orange/ewhite/NeonData/*/DP3.30010.001/**/Camera/**/*.tif",
    ],
    "NEON_points": [
        "/orange/ewhite/NeonData/*/DP3.30010.001/**/Camera/**/*.tif",
    ],
    "MultiTemporal_NEON": [
        "/orange/ewhite/DeepForest/MultiTemporal/NEON/MLBS/2022/NEONPlots/Camera/L3/MLBS_026_2022.tif",
        "/orange/ewhite/DeepForest/MultiTemporal/NEON/**/Camera/L3/*.tif",
    ],
    "Firoze2023": [
        "/orange/ewhite/DeepForest/Firoze2023/*.tif",
        "/orange/ewhite/DeepForest/Firoze2023/**/*.tif",
    ],
    "KagglePalm": [
        "/orange/ewhite/DeepForest/KagglePalm/Palm-Counting-349images/*.tif",
    ],
    "DetectTree2": [
        "/orange/ewhite/DeepForest/DetectTree2/RCD105_MA14_21_orthomosaic_*.tif",
        "/orange/ewhite/DeepForest/DetectTree2/*.tif",
    ],
    "Dumortier2025": [
        "/orange/ewhite/DeepForest/Zenodo_15155081/jelled_annotations_finetuning_DF/images/*.tif",
    ],
    "SPREAD": [
        "/orange/ewhite/DeepForest/SPREAD/**/*.tif",
        "/orange/ewhite/DeepForest/SPREAD/*.tif",
    ],
    "Radogoshi_Sweden": [
        "/orange/ewhite/DeepForest/Radogoshi_Sweden/*.tif",
    ],
    "Kattenborn_NewZealand": [
        "/orange/ewhite/DeepForest/Kattenborn/uav_newzealand_waititu/ecm_uav_rgb_dsm.tif",
        "/orange/ewhite/DeepForest/Kattenborn/uav_newzealand_waititu/wcm_uav_rgb_dsm.tif",
        "/orange/ewhite/DeepForest/Kattenborn/uav_newzealand_waititu/*.tif",
    ],
    "OAM_TCD": [
        "/orange/ewhite/DeepForest/OAM_TCD/images/*.tif",
    ],
}


def find_first_file(patterns):
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        if matches:
            return sorted(matches)[0]
    return None


def get_gsd_meters(filepath):
    """Return (res_x_m, res_y_m, crs_str) or None if not georeferenced."""
    try:
        with rasterio.open(filepath) as src:
            if src.crs is None:
                return None
            transform = src.transform
            res_x = abs(transform.a)
            res_y = abs(transform.e)
            crs = src.crs
            # If projected (units = metres), return directly
            if crs.is_projected:
                return res_x, res_y, crs.to_string()
            # LOCAL_CS with metre units (e.g. S-JTSK Krovak) - treat as projected
            try:
                if "metre" in crs.to_string().lower() or "meter" in crs.to_string().lower():
                    return res_x, res_y, crs.to_string()
            except Exception:
                pass
            # If geographic (degrees), convert using approximate metres-per-degree at centre
            if crs.is_geographic:
                cy = (src.bounds.top + src.bounds.bottom) / 2.0
                m_per_deg_lat = 111320.0
                m_per_deg_lon = 111320.0 * np.cos(np.radians(cy))
                res_x_m = res_x * m_per_deg_lon
                res_y_m = res_y * m_per_deg_lat
                return res_x_m, res_y_m, crs.to_string()
    except Exception as e:
        print(f"  [warn] {filepath}: {e}", file=sys.stderr)
    return None


rows = []
print(f"{'Dataset':<35} {'GSD_x (m)':>12} {'GSD_y (m)':>12}  File")
print("-" * 100)
for name, patterns in sorted(DATASETS.items()):
    fpath = find_first_file(patterns)
    if fpath is None:
        print(f"  [skip] {name}: no file found", file=sys.stderr)
        rows.append((name, None, None, None))
        continue
    result = get_gsd_meters(fpath)
    if result is None:
        print(f"  [skip] {name}: no projection in {os.path.basename(fpath)}", file=sys.stderr)
        rows.append((name, None, None, fpath))
        continue
    rx, ry, crs_str = result
    print(f"{name:<35} {rx:>12.4f} {ry:>12.4f}  {os.path.basename(fpath)}")
    rows.append((name, rx, ry, fpath))

# Summary table (datasets with valid GSD)
print("\n\n=== MANUSCRIPT TABLE ===")
print(f"{'Dataset':<35} {'GSD (cm)':>12}")
print("-" * 50)
for name, rx, ry, fpath in rows:
    if rx is not None:
        gsd_cm = ((rx + ry) / 2.0) * 100
        print(f"{name:<35} {gsd_cm:>10.1f}")
