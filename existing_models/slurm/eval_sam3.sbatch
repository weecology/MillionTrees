#!/bin/bash
#SBATCH --job-name=mt_eval_sam3
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLITS=(random zeroshot)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_BASE=$REPO/existing_models/sam3/outputs/$SPLIT

cd $REPO/existing_models/sam3

echo "=== SAM3 eval: split=$SPLIT ==="

uv run python eval_boxes.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --device cuda \
  --batch-size 8 \
  --score-threshold 0.1 \
  --output-dir "$OUT_BASE"

uv run python eval_points.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --device cuda \
  --batch-size 16 \
  --score-threshold 0.1 \
  --output-dir "$OUT_BASE"

uv run python eval_polygons.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --device cuda \
  --batch-size 8 \
  --score-threshold 0.1 \
  --output-dir "$OUT_BASE"

echo "=== SAM3 eval done. Results in $OUT_BASE ==="
