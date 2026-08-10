#!/bin/bash
#SBATCH --job-name=mt_profile_tfsam2
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# Profiling run: 10 batches on a B200, large batch/prompt sizes.
# Output shows per-batch breakdown: TreeFormer time vs SAM2 time vs throughput.
# Use results to tune batch-size and prompt-batch-size for production runs.

set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}

cd $REPO/existing_models/treeformer_sam2

echo "=== TreeFormer+SAM2 profiling on $(hostname) ($(date)) ==="
echo "batch-size=${BATCH_SIZE:-8}  prompt-batch-size=${PROMPT_BATCH:-128}"

uv run python eval_polygons_crossgeometry.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme crossgeometry \
  --device cuda \
  --batch-size "${BATCH_SIZE:-8}" \
  --num-workers 4 \
  --iou-threshold 0.3 \
  --mask-threshold 0.3 \
  --prompt-batch-size "${PROMPT_BATCH:-128}" \
  --max-batches 10 \
  --profile

echo "=== Profiling done ($(date)) ==="
