#!/bin/bash
#SBATCH --job-name=mt_polygons_overfit
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# Sanity check, NOT a leaderboard run: train and evaluate on the SAME rows
# (--debug-overfit points val + final eval at the train set). Question being
# answered: is the model learning at all, and is the eval pipeline wired up?
# Expectation on a healthy setup: val losses fall and train-set mask_acc/AP50
# climb well above the ~0.40 test-set numbers. Checkpointing, early stopping,
# and leaderboard JSON/results writes are all disabled in this mode.
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SPLIT=random
OUT_DIR="training/polygons/outputs/debug_overfit/$SPLIT"
mkdir -p "$OUT_DIR"

uv run python training/polygons/train.py \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme "$SPLIT" \
  --batch-size 8 \
  --max-epochs 25 \
  --lr 2e-4 \
  --num-workers 8 \
  --image-size 448 \
  --init-mode coco \
  --debug-overfit \
  --output-dir "$OUT_DIR" \
  --comet
