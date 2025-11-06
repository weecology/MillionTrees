#!/bin/bash
#SBATCH --job-name=mt-examples
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60GB

# Common args
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
OUT_DIR=${OUT_DIR:-runs/examples/${SLURM_JOB_ID:-local}}
mkdir -p "$OUT_DIR"

# echo "Running baseline_boxes.py"
uv run python docs/examples/baseline_boxes.py --root-dir=$ROOT_DIR --output-dir=$OUT_DIR

# echo "Running baseline_polygons.py"
uv run python docs/examples/baseline_polygons.py --root-dir=$ROOT_DIR --output-dir=$OUT_DIR

echo "Running baseline_points.py"
uv run python docs/examples/baseline_points.py --root-dir=$ROOT_DIR --output-dir=$OUT_DIR
