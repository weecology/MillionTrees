#!/bin/bash
#SBATCH --job-name=mt_eval_sam3_points
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail

# Points-only SAM3 re-eval. Box/polygon eval data is unchanged, so only the
# points stage is re-run (writes results_points_<split>.txt into the same
# outputs/<split>/ dir, leaving the box and polygon result files untouched).
# Mirrors the points stage of eval_sam3.sbatch without paying for all three
# geometries.
#
# Split names were `random`/`zeroshot` until the split rename; those values are
# no longer in eval_points.py's --split-scheme choices, so this script crashed on
# startup until corrected to the current names.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_BASE=$REPO/existing_models/sam3/outputs/$SPLIT

cd $REPO/existing_models/sam3

echo "=== SAM3 points-only eval: split=$SPLIT ==="

uv run python eval_points.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --device cuda \
  --batch-size 8 \
  --score-threshold 0.1 \
  --output-dir "$OUT_BASE"

echo "=== SAM3 points eval done. Results in $OUT_BASE ==="
