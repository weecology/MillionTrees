#!/usr/bin/env bash
set -euo pipefail

cd /blue/ewhite/b.weinstein/src/MillionTrees
mkdir -p logs/slurm

SCRIPTS=(
  slurm/df_points.sbatch
  slurm/df_boxes.sbatch
  slurm/df_polygons.sbatch
  slurm/sam3_points.sbatch
  slurm/sam3_boxes.sbatch
  slurm/sam3_polygons.sbatch
  slurm/yolo_boxes.sbatch
)

for s in "${SCRIPTS[@]}"; do
  echo "Submitting: $s"
  sbatch "$s"
done


