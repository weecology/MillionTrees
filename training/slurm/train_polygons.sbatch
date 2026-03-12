#!/bin/bash
#SBATCH --job-name=mt_train_polygons
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
mkdir -p training/polygons/outputs

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLITS=(random zeroshot crossgeometry)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

uv run python training/polygons/train_polygons.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --batch-size 2 \
  --max-epochs 20 \
  --lr 1e-4 \
  --num-workers 2 \
  --image-size 448 \
  --output-dir "training/polygons/outputs/$SPLIT"
