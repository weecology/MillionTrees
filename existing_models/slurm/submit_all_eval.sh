#!/usr/bin/env bash
# Submit all existing-model evaluation jobs.
set -euo pipefail

cd /blue/ewhite/b.weinstein/src/MillionTrees

echo "Submitting DeepForest (pretrained) evaluation..."
sbatch existing_models/slurm/eval_deepforest.sbatch

echo "Submitting SAM3 evaluation..."
sbatch existing_models/slurm/eval_sam3.sbatch

echo "All eval jobs submitted."
