#!/bin/bash
#SBATCH --job-name=tcd_filter
#SBATCH --output=logs/slurm/filter_auto_arborist_tcd_%j.out
#SBATCH --error=logs/slurm/filter_auto_arborist_tcd_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

# Filter AutoArborist annotations with Restor TCD SegFormer (tree/no-tree).
# Run full dataset (no --debug). For a quick test, run locally:
#   uv run python data_prep/filter_auto_arborist_tcd.py --debug --max-tiles 2

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
mkdir -p logs/slurm

uv sync --extra dev

ANNOTATIONS_CSV="/orange/ewhite/DeepForest/AutoArborist/downloaded_imagery/AutoArborist_combined_annotations.csv"
OUTPUT_CSV="/orange/ewhite/DeepForest/AutoArborist/downloaded_imagery/AutoArborist_combined_annotations_tcd_filtered.csv"

echo "Starting TCD filter at $(date)"
echo "Input:  $ANNOTATIONS_CSV"
echo "Output: $OUTPUT_CSV"

uv run python data_prep/filter_auto_arborist_tcd.py \
  --annotations-csv "$ANNOTATIONS_CSV" \
  --output-csv "$OUTPUT_CSV" \
  --device cuda
  --max-per-city 3

echo "Finished at $(date)"
