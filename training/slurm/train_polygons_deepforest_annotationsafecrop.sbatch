#!/bin/bash
#SBATCH --job-name=mt_poly_annotationsafecrop
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

# Annotation-safe crop arm of the crop-vs-crop comparison (extends the
# nativecrop vs randomresizedcrop comparison from train_polygons_deepforest_cropcompare).
#
# Problem: ~75% of MillionTrees training images are incompletely annotated, so
# a plain RandomCrop(640) (nativecrop arm) frequently lands on blank or
# unannotated regions -- the model trains on nothing for those windows.
#
# Fix: RandomSizedBBoxSafeCrop (DeepForest PR bw4sz#2, patched into venv) centers
# every crop on the union of the image's bounding boxes, with random context
# padding (1x-2x the union). Each training window is guaranteed to contain at
# least one annotated polygon. The crop is then resized to 640 to match the
# tiled-eval predict_tile patch size.
#
# arm 0: within-distribution   arm 1: out-of-distribution
# lr 0.01, batch 16, 100 epochs, coco init, eval-inference tiled

export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="training/polygons/outputs/deepforest_annotationsafecrop/$SPLIT"
mkdir -p "$OUT_DIR"

srun uv run python training/polygons/train.py \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme "$SPLIT" \
  --gpus 1 \
  --batch-size 16 \
  --lr 0.01 \
  --max-epochs 100 \
  --num-workers 8 \
  --image-size 448 \
  --train-aug annotationsafecrop \
  --eval-inference tiled \
  --init-mode coco \
  --data-scope subset \
  --output-dir "$OUT_DIR" \
  --comet \
  --comet-name "polygons-${SPLIT}-pretrained-lr0.01-annotationsafecrop"
