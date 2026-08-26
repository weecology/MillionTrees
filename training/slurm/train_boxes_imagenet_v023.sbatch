#!/bin/bash
#SBATCH --job-name=mt_boxes_imagenet_v023
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# COLD-START floor arm of the box ablation, v0.23 / AP40.
#
# Every existing arm (coco / box_pretrained / box_pretrained_full) starts from
# COCO, so the ladder could only ever measure weak-label-vs-COCO, never whether
# detection pretraining helps at all. This arm supplies the missing rung:
# ImageNet ResNet50 trunk, freshly initialized FPN and heads, no detection
# pretraining of any kind. It is also what manuscript Table 6 already calls
# "an ImageNet backbone".
#
# Note when reading the result: ImageNet and COCO trunks agree on 223/265
# tensors (COCO RetinaNet is itself ImageNet-initialized and freezes the early
# layers), so the coco-vs-imagenet delta is carried by 42 trunk tensors plus
# the 16-tensor FPN. A small gap here is expected and is not evidence the
# guards failed -- assert_imagenet_trunk verifies the init positively.
#
# Recipe is byte-identical to train_boxes_coco_v023_cleaninit.sbatch (batch 32,
# max-epochs 200, lr 0.01, patience 10/15, clip 1.0) so the arms stay
# comparable; early stopping, not the epoch cap, governs when a slower-
# converging cold start finishes.
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="training/boxes/outputs/imagenet_v023/$SPLIT"
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
  --init-mode imagenet \
  --output-dir "$OUT_DIR" \
  --comet \
  --comet-name "boxes-${SPLIT}-imagenet-lr0.01-v023"
