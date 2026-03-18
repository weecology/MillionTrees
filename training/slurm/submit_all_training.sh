#!/usr/bin/env bash
set -euo pipefail

cd /blue/ewhite/b.weinstein/src/MillionTrees

echo "Submitting box training..."
sbatch training/slurm/train_boxes.sbatch

echo "Submitting point training..."
sbatch training/slurm/train_points.sbatch

echo "Submitting polygon training..."
sbatch training/slurm/train_polygons.sbatch

echo "All training jobs submitted."
