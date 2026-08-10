#!/bin/bash
#SBATCH --job-name=mt_poly_deepforest_resize1280
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

# Polygon Mask R-CNN, RESIZE-NO-CROP at HIGHER RESOLUTION (1280).
#
# Experiment (2026-06-26): direct recall test against the 640 resize run
# (35742211). The 640 resize arm wins precision (0.24 vs crop 0.20) but loses
# recall (0.60 vs 0.64) -- hypothesised to be a resolution artifact: small trees
# downsampled to ~4px at 640 vanish. At 1280 (4x the pixels) they survive as
# ~8px, so this tests whether higher resolution recovers recall while keeping the
# whole-scene precision advantage. Same recipe otherwise (resize train + scale-
# matched resize eval, lr 0.01, coco init).
#
# Memory: 640/batch16 used 27.4GB on the 192GB B200; activation scales ~(S/640)^2,
# so 1280/batch20 ~= 4GB overhead + 20*5.85 ~= 121GB -- comfortable. Batch raised
# 16->20 (slightly larger as requested); lr kept at 0.01 for clean comparison to
# the 640 resize arm. 50 epochs (the model overfits early; 100 was wasteful).
# Single-GPU like the baseline (sidesteps the NCCL /orange-contention hang).
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="training/polygons/outputs/deepforest_resize1280/$SPLIT"
mkdir -p "$OUT_DIR"

srun uv run python training/polygons/train.py \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme "$SPLIT" \
  --gpus 1 \
  --batch-size 20 \
  --lr 0.01 \
  --max-epochs 50 \
  --num-workers 16 \
  --image-size 1280 \
  --train-aug resize \
  --eval-inference resize \
  --init-mode coco \
  --data-scope subset \
  --output-dir "$OUT_DIR" \
  --comet \
  --comet-name "polygons-${SPLIT}-pretrained-lr0.01-deepforest-resize1280"
