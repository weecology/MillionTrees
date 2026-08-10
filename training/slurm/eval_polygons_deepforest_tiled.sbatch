#!/bin/bash
#SBATCH --job-name=mt_poly_df_tiled_eval
#SBATCH --output=/home/b.weinstein/logs/%x_%A.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

# Tiled native-resolution eval of the DeepForest polygon Mask R-CNN.
#
# The previous eval fed the model whole images squished to 448, but the model
# trains on RandomResizedCrop-640 crops of native-resolution imagery -> a
# train/eval scale mismatch that produced oversized blob masks. This run uses
# the new tiled_predict_for_eval path (DeepForest predict_tile, 640 windows over
# the native image, polygons projected back into the 448 metric space).
#
# Writes to a fresh deepforest_tiled/ dir so the old (broken) results+viz under
# deepforest/within-distribution stay available for side-by-side comparison.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLIT=within-distribution
CKPT_DIR="training/polygons/outputs/deepforest/${SPLIT}/logs/checkpoints"
# Lowest val_loss among converged checkpoints (epoch >= 1).
CKPT=$(ls -1 "$CKPT_DIR"/polygons-epoch=*.ckpt \
  | awk -F'epoch=' '{split($2,a,"-"); if (a[1]+0 > 0) print}' \
  | awk -F'val_loss=' '{print $2"\t"$0}' \
  | sort -g \
  | head -1 \
  | cut -f2-)

OUT_DIR="training/polygons/outputs/deepforest_tiled/${SPLIT}"
mkdir -p "$OUT_DIR"

echo "=== Tiled eval DeepForest polygons: split=$SPLIT ==="
echo "Checkpoint: $CKPT"

uv run python training/polygons/eval.py \
  --checkpoint "$CKPT" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --batch-size 4 \
  --num-workers 4 \
  --image-size 448 \
  --score-threshold 0.1 \
  --init-mode coco \
  --data-scope subset \
  --output-dir "$OUT_DIR"
