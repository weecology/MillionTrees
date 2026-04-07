#!/bin/bash
#SBATCH --job-name=MillionTrees_DeepForest   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=ben.weinstein@weecology.org # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI r
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=100GB
#SBATCH --time=48:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/format_MillionTrees_%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/format_MillionTrees_%j.err

#Add to path

MASK_DIR="/orange/ewhite/DeepForest/tree_coverage_masks"
CFG="data_prep/annotation_csvs.cfg"

# Read all active (non-commented) CSV paths from the config into an array
mapfile -t ALL_CSVS < <(grep -v '^\s*#' "$CFG" | grep -v '^\s*\[' | grep -v '^\s*$')

uv run python -m data_prep.precompute_tree_coverage_masks \
    --output-dir "$MASK_DIR" \
    --annotations-csv "${ALL_CSVS[@]}"

MILLIONTREES_MASKS_DIR="$MASK_DIR" uv run python data_prep/package_datasets.py