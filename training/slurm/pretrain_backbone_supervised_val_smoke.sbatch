#!/bin/bash
#SBATCH --job-name=mt_pretrain_supval_smoke
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=90GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# Smoke test for the instrumented stage-1 pretraining (supervised validation).
# Verifies, in ~15 min instead of ~7 h, that:
#   1. build_supervised_val_subset() returns a non-empty, multi-source holdout
#   2. Lightning actually runs validation_step (val_loss appears in the metrics dump)
#   3. the ModelCheckpoint fires and reports a best/last path
#   4. the pre-training BASELINE eval and the post-training eval both produce
#      real numbers instead of the nan block every previous stage-1 run wrote
#   5. the cosine scheduler attaches (LearningRateMonitor logs lr-SGD)
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

OUT_DIR="training/weak_supervision/outputs/pretrain_supval_smoke"
mkdir -p "$OUT_DIR"

uv run python training/boxes/pretrain_backbone_for_polygons.py \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme within-distribution \
  --include-unsupervised \
  --init-mode coco \
  --batch-size 8 \
  --max-epochs 2 \
  --lr 1e-4 \
  --scheduler cosine \
  --supervised-val-images 64 \
  --num-workers 4 \
  --limit-train-batches 30 \
  --limit-val-batches 8 \
  --eval-max-batches 8 \
  --output-dir "$OUT_DIR"
