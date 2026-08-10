#!/bin/bash
# Score-threshold sweep for PRETRAINED DeepForest on v0.19 boxes, both splits.
# Inference once at model score_thresh 0.01, re-score P/R/F1 across
# --sweep-thresholds to locate the F1-optimal operating point vs the current 0.1.
# Full test set (no --per-source). Eval-only.
#SBATCH --job-name=mt_sweep_deepforest_v019
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR=$REPO/existing_models/deepforest/outputs/sweep/$SPLIT

cd "$REPO/existing_models/deepforest"

echo "=== Pretrained DeepForest threshold sweep: split=$SPLIT ==="
uv run python eval_boxes.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --score-threshold 0.01 \
  --sweep \
  --sweep-thresholds 0.025,0.05,0.075,0.1 \
  --output-dir "$OUT_DIR"

echo "=== Done -> $OUT_DIR/threshold_sweep.csv ==="
