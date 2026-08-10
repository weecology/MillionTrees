#!/bin/bash
#SBATCH --job-name=mt_apiou_sam3
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail

# AP50-vs-AP40 comparison run: same settings as eval_sam3.sbatch, writing to
# outputs_ap_iou/. The points stage is skipped -- TreePoints matching is
# distance-based, so there is no IoU threshold to vary.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_BASE=$REPO/existing_models/sam3/outputs_ap_iou/$SPLIT

cd $REPO/existing_models/sam3

echo "=== SAM3 AP50/AP40 eval: split=$SPLIT ==="

uv run python eval_boxes.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --device cuda \
  --batch-size 8 \
  --score-threshold 0.1 \
  --output-dir "$OUT_BASE"

uv run python eval_polygons.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --device cuda \
  --batch-size 4 \
  --score-threshold 0.1 \
  --output-dir "$OUT_BASE"

echo "=== SAM3 AP50/AP40 eval done. Results in $OUT_BASE ==="
