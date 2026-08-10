#!/bin/bash
# Run B: score_thresh=0.2, nms_distance_thresh=3.0 (moderate confidence + tighter NMS)
#SBATCH --job-name=mt_tf_sam2_runB
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
OUT_DIR=$REPO/existing_models/treeformer_sam2/outputs/crossgeometry_runB

cd $REPO/existing_models/treeformer_sam2

echo "=== Run B: score_thresh_tf=0.2, nms_distance_thresh=3.0 ==="

uv run python eval_polygons_crossgeometry.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme crossgeometry \
  --device cuda \
  --batch-size 2 \
  --num-workers 4 \
  --score-thresh-tf 0.2 \
  --nms-distance-thresh 3.0 \
  --iou-threshold 0.3 \
  --mask-threshold 0.3 \
  --output-dir "$OUT_DIR" \
  --viz-dir "$OUT_DIR/viz" \
  --viz-size 512

echo "=== Done. Results in $OUT_DIR ==="
