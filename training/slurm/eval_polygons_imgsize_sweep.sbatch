#!/bin/bash
#SBATCH --job-name=mt_eval_poly_imgsize
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# Image-size sensitivity sweep for the trained polygon Mask R-CNN.
# Holds the checkpoint fixed (best random val_mask_acc) and the split fixed
# (random test), varying only dataset image_size. NOTE: the model uses the
# default GeneralizedRCNNTransform (min_size=800), so image_size mainly changes
# the GT/pred mask grid resolution and pre-model input detail, not the
# backbone's working resolution. This sweep measures how much that matters.

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLIT=random
CKPT="training/polygons/outputs/random/checkpoints/polygons-epoch=25-val_mask_acc=0.3656.ckpt"

SIZES=(448 672 896 1280)
SIZE=${SIZES[$SLURM_ARRAY_TASK_ID]}

OUT_DIR="training/polygons/outputs/imgsize_sweep/random/size_${SIZE}"
mkdir -p "$OUT_DIR"

echo "=== Polygon image-size sweep: split=$SPLIT image_size=$SIZE ==="
echo "Checkpoint: $CKPT"

uv run python training/polygons/eval.py \
  --checkpoint "$CKPT" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --batch-size 4 \
  --num-workers 4 \
  --image-size "$SIZE" \
  --score-threshold 0.1 \
  --output-dir "$OUT_DIR" \
  --viz-dir ""

echo "=== Done. Results in $OUT_DIR ==="
