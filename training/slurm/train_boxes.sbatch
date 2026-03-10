#!/bin/bash
#SBATCH --job-name=mt_train_boxes
#SBATCH --output=training/boxes/outputs/%x_%A_%a.out
#SBATCH --error=training/boxes/outputs/%x_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
mkdir -p training/boxes/outputs

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLITS=(random zeroshot crossgeometry)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

uv run python training/boxes/train_boxes.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --batch-size 8 \
  --max-epochs 20 \
  --lr 1e-4 \
  --num-workers 4 \
  --output-dir "training/boxes/outputs/$SPLIT"
