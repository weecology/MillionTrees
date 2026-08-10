#!/bin/bash
#SBATCH --job-name=mt_treeformer_sam2
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
OUT_DIR=$REPO/existing_models/treeformer_sam2/outputs/crossgeometry

cd $REPO/existing_models/treeformer_sam2

echo "=== TreeFormer+SAM2 cross-geometry polygon eval ==="
echo "Root dir: $ROOT_DIR"
echo "Output dir: $OUT_DIR"

uv run python eval_polygons_crossgeometry.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme crossgeometry \
  --device cuda \
  --batch-size 2 \
  --num-workers 4 \
  --iou-threshold 0.3 \
  --mask-threshold 0.3 \
  --output-dir "$OUT_DIR" \
  --viz-dir "$OUT_DIR/viz" \
  --viz-size 512

echo "=== Done. Results in $OUT_DIR ==="
