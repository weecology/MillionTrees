#!/bin/bash
#SBATCH --job-name=mt_polygons_lrsweep
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-4
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLIT=random

# Diagnosing flat train loss / rising val loss: sweep learning rate across two
# orders of magnitude with early stopping OFF so every run trains the full
# --max-epochs and the val_loss curve can be read end to end.
LRS=(1e-3 3e-4 1e-4 3e-5 1e-5)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

OUT_DIR="training/polygons/outputs/lr_sweep/${SPLIT}_lr${LR}"
mkdir -p "$OUT_DIR"

uv run python training/polygons/train.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --batch-size 8 \
  --max-epochs 30 \
  --lr "$LR" \
  --num-workers 8 \
  --image-size 448 \
  --no-early-stopping \
  --init-mode coco \
  --output-dir "$OUT_DIR" \
  --comet
