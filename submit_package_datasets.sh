#!/bin/bash
#SBATCH --job-name=MillionTrees_DeepForest   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=ben.weinstein@weecology.org # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI r
#SBATCH --cpus-per-task=1
#SBATCH --mem=100GB
#SBATCH --time=48:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/format_MillionTrees_%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/format_MillionTrees_%j.err

#Add to path

MASK_DIR="/orange/ewhite/DeepForest/tree_coverage_masks"

uv run python -m data_prep.precompute_tree_coverage_masks \
    --output-dir "$MASK_DIR" \
    --annotations-csv \
        /orange/ewhite/DeepForest/Radogoshi_Sweden/annotations.csv \
        /orange/ewhite/DeepForest/Guangzhou2022/annotations.csv \
        /orange/ewhite/DeepForest/NEON_benchmark/NeonTreeEvaluation_annotations.csv \
        /orange/ewhite/DeepForest/NEON_benchmark/University_of_Florida.csv \
        /orange/ewhite/DeepForest/ReForestTree/images/train.csv \
        /orange/ewhite/DeepForest/Zenodo_15155081/parsed_annotations.csv \
        /orange/ewhite/DeepForest/OAM_TCD/annotations.csv \
        /orange/ewhite/DeepForest/SelvaBox/annotations.csv \
        /orange/ewhite/DeepForest/neon_unsupervised/TreeBoxes_neon_unsupervised.csv \
        /orange/ewhite/DeepForest/OpenForestObservatory/images/TreeBoxes_OFO_unsupervised.csv \
        /orange/ewhite/DeepForest/MultiTemporal/annotations/TreeBoxes_NEON_MultiTemporal.csv \
        /orange/ewhite/DeepForest/Puliti_2022/annotations.csv \
        /orange/ewhite/DeepForest/TreeFormer/all_images/annotations.csv \
        /orange/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/annotations.csv \
        /orange/ewhite/MillionTrees/NEON_points/annotations.csv \
        /orange/ewhite/DeepForest/AutoArborist/downloaded_imagery/AutoArborist_combined_annotations_tcd_filtered.csv \
        /orange/ewhite/DeepForest/Yosemite/tiles/yosemite_all_annotations.csv \
        /orange/ewhite/DeepForest/OpenForestObservatory/images/TreePoints_OFO_unsupervised.csv \
        /orange/ewhite/DeepForest/Kaggle_LiDAR_RGB/pngs/annotations.csv \
        /orange/ewhite/DeepForest/MultiTemporal/annotations/TreePoints_NEON_MultiTemporal.csv \
        /orange/ewhite/DeepForest/OSBS_megaplot/2025/pngs/annotations.csv \
        /orange/ewhite/DeepForest/Jansen_2023/pngs/annotations.csv \
        /orange/ewhite/DeepForest/Troles_Bamberg/coco2048/annotations/annotations.csv \
        /orange/ewhite/DeepForest/Cloutier2023/images/annotations.csv \
        /orange/ewhite/DeepForest/Firoze2023/crops/annotations.csv \
        /orange/ewhite/DeepForest/paracou_ball/pngs/annotations.csv \
        /orange/ewhite/DeepForest/UrbanLondon/annotations.csv \
        /orange/ewhite/DeepForest/BCI/BCI_50ha_2020_08_01_crownmap_raw/annotations.csv \
        /orange/ewhite/DeepForest/BCI/BCI_50ha_2022_09_29_crownmap_raw/annotations.csv \
        /orange/ewhite/DeepForest/Harz_Mountains/ML_TreeDetection_Harz/annotations.csv \
        /orange/ewhite/DeepForest/SPREAD/annotations.csv \
        /orange/ewhite/DeepForest/KagglePalm/Palm-Counting-349images/crops/annotations.csv \
        /orange/ewhite/DeepForest/Kattenborn/uav_newzealand_waititu/crops/annotations.csv \
        /orange/ewhite/DeepForest/Quebec_Lefebvre/Dataset/Crops/annotations.csv \
        "/orange/ewhite/DeepForest/TreeCountSegHeight/extracted_data_2aux_v4_cleaned_centroid_raw 2/crops/annotations.csv" \
        /orange/ewhite/DeepForest/Schutte_Germany/annotations.csv \
        /orange/ewhite/DeepForest/MultiTemporal/annotations/TreePolygons_NEON_MultiTemporal.csv

MILLIONTREES_MASKS_DIR="$MASK_DIR" uv run python data_prep/package_datasets.py