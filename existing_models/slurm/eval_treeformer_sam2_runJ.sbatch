#!/bin/bash
# Run J: patch-fraction 0.5 (~4 patches) + negative prompts + hiera-large
#SBATCH --job-name=mt_tf_sam2_runJ
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail
export UV_NO_CACHE=1

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
OUT_DIR=$REPO/existing_models/treeformer_sam2/outputs/crossgeometry_runJ

cd $REPO/existing_models/treeformer_sam2

echo "=== Run J: patch-fraction 0.5 + negative prompts + hiera-large ==="

uv run python eval_polygons_crossgeometry.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme crossgeometry \
  --device cuda \
  --batch-size 1 \
  --num-workers 4 \
  --patch-fraction 0.5 \
  --sam2-model "facebook/sam2.1-hiera-large" \
  --negative-prompts \
  --score-thresh-tf 0.3 \
  --iou-threshold 0.3 \
  --mask-threshold 0.3 \
  --output-dir "$OUT_DIR" \
  --viz-dir "$OUT_DIR/viz" \
  --viz-size 512

echo "=== Done. Results in $OUT_DIR ==="
