#!/usr/bin/env bash
# Submit all fine-tuned training jobs (one array job per geometry, over
# random + zeroshot splits). Results land in training/<geometry>/outputs/<split>/.
set -euo pipefail

cd /blue/ewhite/b.weinstein/src/MillionTrees

echo "Submitting box training (DeepForest)..."
sbatch training/slurm/train_boxes.sbatch

echo "Submitting point training (TreeFormer)..."
sbatch training/slurm/train_points.sbatch

echo "Submitting polygon training (Mask R-CNN)..."
sbatch training/slurm/train_polygons.sbatch

echo "All training jobs submitted."
