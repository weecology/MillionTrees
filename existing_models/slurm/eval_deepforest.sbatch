#!/bin/bash
#SBATCH --job-name=mt_eval_deepforest
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLITS=(random zeroshot)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_BASE=$REPO/existing_models/deepforest/outputs/$SPLIT

cd $REPO/existing_models/deepforest

echo "=== DeepForest eval: split=$SPLIT ==="

# uv run python eval_boxes.py \
#   --root-dir "$ROOT_DIR" \
#   --split-scheme "$SPLIT" \
#   --batch-size 12 \
#   --num-workers 4 \
#   --output-dir "$OUT_BASE"

# uv run python eval_points.py \
#   --root-dir "$ROOT_DIR" \
#   --split-scheme "$SPLIT" \
#   --batch-size 32 \
#   --num-workers 4 \
#   --output-dir "$OUT_BASE"

uv run python eval_polygons.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --batch-size 32 \
  --num-workers 4 \
  --output-dir "$OUT_BASE"

echo "=== DeepForest eval done. Results in $OUT_BASE ==="
