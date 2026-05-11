#!/usr/bin/env bash
set -euo pipefail

cd /blue/ewhite/b.weinstein/src/MillionTrees
mkdir -p logs/slurm
# Create a per-run log directory and point 'latest' to it
RUN_ID="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="logs/slurm/${RUN_ID}"
mkdir -p "${RUN_DIR}"
ln -sfn "${RUN_ID}" logs/slurm/latest

SCRIPTS=(
  slurm/df_points.sbatch
  slurm/df_boxes.sbatch
  slurm/df_polygons.sbatch
  slurm/sam3_points.sbatch
  slurm/sam3_boxes.sbatch
  slurm/sam3_polygons.sbatch
)

for s in "${SCRIPTS[@]}"; do
  echo "Submitting: $s -> ${RUN_DIR}"
  sbatch --output "${RUN_DIR}/%x_%A_%a.out" --error "${RUN_DIR}/%x_%A_%a.err" "$s"
done


