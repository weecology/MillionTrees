#!/bin/bash
# SAM3 points-only eval at image_size=672 to match the TreeFormer point rows
# (pretrained 34733817 / finetuned 34733818). keypoint_acc tolerance scales with
# image_size, so the existing SAM3 points results (default 448) are NOT comparable
# to the 672 TreeFormer rows — re-run at 672 for an iso-resolution manuscript
# table3 points block. Writes to outputs/<split>_img672 so the 448 results are
# preserved. Bohlman 2008 already dropped from the v0.18 config.
#SBATCH --job-name=mt_eval_sam3_points_672
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLITS=(random zeroshot)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_BASE=$REPO/existing_models/sam3/outputs/${SPLIT}_img672

cd $REPO/existing_models/sam3

echo "=== SAM3 points-only eval: split=$SPLIT image_size=672 ==="

uv run python eval_points.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --device cuda \
  --batch-size 8 \
  --score-threshold 0.1 \
  --image-size 672 \
  --viz-dir "" \
  --output-dir "$OUT_BASE"

echo "=== SAM3 points eval done. Results in $OUT_BASE ==="
