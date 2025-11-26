#!/bin/bash
#SBATCH --job-name=mt_df_points
#SBATCH --output=logs/slurm/df_points_%x_%A_%a.out
#SBATCH --error=logs/slurm/df_points_%x_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
mkdir -p logs/slurm

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLITS=(random zeroshot crossgeometry)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

uv run python docs/examples/baseline_points.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --plot-interval 0 \
  --batch-size 32


