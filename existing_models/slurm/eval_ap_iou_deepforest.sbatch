#!/bin/bash
#SBATCH --job-name=mt_apiou_deepforest
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail

# AP50-vs-AP40 comparison run: identical to eval_deepforest.sbatch (same score
# threshold, same test split) but writing to outputs_ap_iou/ so the leaderboard
# results stay untouched. The datasets now score AP at IoU 0.5 (AP50) and IoU
# 0.4 (AP40, the same IoU the recall/precision that feed F1 use) on the same
# predictions.

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_BASE=$REPO/existing_models/deepforest/outputs_ap_iou/$SPLIT

cd $REPO/existing_models/deepforest

echo "=== DeepForest pretrained AP50/AP40 eval (boxes): split=$SPLIT ==="

uv run python eval_boxes.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --batch-size 12 \
  --num-workers 4 \
  --score-threshold 0.1 \
  --output-dir "$OUT_BASE"

echo "=== DeepForest AP50/AP40 eval done. Results in $OUT_BASE ==="
