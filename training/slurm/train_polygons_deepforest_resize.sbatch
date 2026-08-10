#!/bin/bash
#SBATCH --job-name=mt_poly_deepforest_resize
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

# Polygon Mask R-CNN, RESIZE-NO-CROP variant of train_polygons_deepforest.sbatch.
#
# Experiment (2026-06-25): instead of RandomResizedCrop 640 of native-resolution
# imagery, scale the WHOLE image (and annotations) to a fixed 640x640 square so
# the model sees the entire scene. --train-aug resize swaps RandomResizedCrop for
# Resize(LongestMaxSize 640)+PadIfNeeded 640; --eval-inference resize runs the
# eval at the SAME scale (whole image -> 640, no tiling) so train and eval are
# scale-consistent (the prior tiled eval would put trees at a scale the
# resize-trained model never saw). --image-size 640 sets that shared scale (the
# GT masks are rasterized at image_size too).
#
# Single-GPU like the baseline (sidesteps the NCCL /orange-contention hang). Same
# recipe otherwise (batch 16, lr 0.01, coco init) so it is directly comparable to
# the crop baseline polygons-<split>-pretrained-lr0.01-deepforest-1gpu.
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="training/polygons/outputs/deepforest_resize/$SPLIT"
mkdir -p "$OUT_DIR"

srun uv run python training/polygons/train.py \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme "$SPLIT" \
  --gpus 1 \
  --batch-size 16 \
  --lr 0.01 \
  --max-epochs 100 \
  --num-workers 16 \
  --image-size 640 \
  --train-aug resize \
  --eval-inference resize \
  --init-mode coco \
  --data-scope subset \
  --output-dir "$OUT_DIR" \
  --comet \
  --comet-name "polygons-${SPLIT}-pretrained-lr0.01-deepforest-resize"
