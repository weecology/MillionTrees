#!/bin/bash
#SBATCH --job-name=mt_boxes_coco_v023
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# Control arm of the same-geometry (box -> box) pretraining ablation, v0.23 / AP40.
# Clean COCO RetinaNet: torchvision COCO weights only, no weecology/deepforest-tree
# load, so no NEON exposure. Paired with train_boxes_box_pretrained_coco_v023.sbatch,
# which differs ONLY in backbone init.
#
# vs. the Jun-30 v0.20 run (36058067): same recipe plus --gradient-clip-val 1.0, which
# the treatment arm needs (36058068_0 diverged to NaN at lr 0.01). Applied to both arms
# so the comparison stays apples-to-apples.
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="training/boxes/outputs/coco_v023/$SPLIT"
mkdir -p "$OUT_DIR"

if [ "$SPLIT" = "out-of-distribution" ]; then
  EARLY_STOP_PATIENCE=15
else
  EARLY_STOP_PATIENCE=10
fi

srun uv run python training/boxes/train.py \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme "$SPLIT" \
  --batch-size 32 \
  --max-epochs 200 \
  --lr 0.01 \
  --num-workers 8 \
  --gpus 1 \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --gradient-clip-val 1.0 \
  --init-mode coco \
  --output-dir "$OUT_DIR" \
  --comet \
  --comet-name "boxes-${SPLIT}-coco-lr0.01-v023-ablation"
