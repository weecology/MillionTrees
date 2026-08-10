#!/bin/bash
# Score-threshold sweep: pretrained DeepForest on boxes, random split, 3 imgs/source.
#SBATCH --job-name=mt_sweep_deepforest
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLIT=random
OUT_BASE=$REPO/existing_models/deepforest/outputs/sweep

cd "$REPO/existing_models/deepforest"

echo "=== DeepForest threshold sweep: split=$SPLIT ==="
uv run python eval_boxes.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --score-threshold 0.01 \
  --per-source 3 \
  --sweep \
  --output-dir "$OUT_BASE"

echo "=== DeepForest sweep done -> $OUT_BASE/threshold_sweep.csv ==="
