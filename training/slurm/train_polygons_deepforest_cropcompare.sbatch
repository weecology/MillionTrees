#!/bin/bash
#SBATCH --job-name=mt_poly_cropcompare
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

# Josh's 640-crop + tiled-eval idea, as a controlled crop-vs-crop comparison on
# the within-distribution split (v0.20, all images / no completeness filter).
# Both arms: lr 0.01, batch 16, 100 epochs, coco init, eval-inference tiled
# (native predict_tile at patch 640). Only the TRAIN crop differs:
#
#   arm 0  crop        RandomResizedCrop scale[0.64,1.0] -> on median-2000px tiles
#                      keeps >=64% of the image then downsamples ~3x to 640
#                      (a whole-image resize in disguise; few blank crops).
#   arm 1  nativecrop  true 640px RandomCrop at native GSD (small trees keep
#                      their pixels, train scale matches the tiled eval). Some
#                      windows go blank / hit unlabeled trees because only ~25%
#                      of train images are fully annotated -- accepted per user.
#
# Compare AP50 / recall / mask-aware precision against the resize640 (35812465)
# and resize1280 (35797052_0) within-distribution runs.
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

AUGS=(crop nativecrop)
NAMES=(randomresizedcrop nativecrop640)
AUG=${AUGS[$SLURM_ARRAY_TASK_ID]}
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="training/polygons/outputs/deepforest_cropcompare/$NAME"
mkdir -p "$OUT_DIR"

srun uv run python training/polygons/train.py \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme within-distribution \
  --gpus 1 \
  --batch-size 16 \
  --lr 0.01 \
  --max-epochs 100 \
  --num-workers 16 \
  --image-size 448 \
  --train-aug "$AUG" \
  --eval-inference tiled \
  --init-mode coco \
  --data-scope subset \
  --output-dir "$OUT_DIR" \
  --comet \
  --comet-name "polygons-within-distribution-pretrained-lr0.01-deepforest-$NAME"
