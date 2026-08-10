#!/bin/bash
#SBATCH --job-name=mt_val_complete_deepforest
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

# End-to-end check of --complete-tiles-only: the same DeepForest validation eval as
# eval_validation_dump_deepforest.sbatch, restricted to the 59 wall-to-wall tiles. AP50/AP40
# here must match the "complete" regime of scripts/rescore_validation_ap.py (0.036 / 0.130).
set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
OUT_BASE=$REPO/existing_models/deepforest/outputs/validation_complete_tiles

cd $REPO/existing_models/deepforest

echo "=== DeepForest pretrained, validation, complete tiles only ==="

uv run python eval_boxes.py \
  --root-dir "$ROOT_DIR" --split-scheme out-of-distribution --eval-split validation \
  --complete-tiles-only \
  --batch-size 12 --num-workers 4 --score-threshold 0.1 --output-dir "$OUT_BASE"

echo "=== done. Results in $OUT_BASE ==="
