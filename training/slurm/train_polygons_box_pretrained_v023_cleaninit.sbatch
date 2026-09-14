#!/bin/bash
#SBATCH --job-name=mt_polygons_boxinit_v023_cleaninit
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

# Treatment arm of the CORRECTED polygon (Table 6) ablation, v0.23 / AP40.
#
# Mask R-CNN with backbone.body.* swapped for the unsupervised box-pretrained trunk,
# FPN/RPN/heads left at COCO. Two things differ from the 08-05 run (job 38750867):
#
#   1. The base is now a real COCO Mask R-CNN. assert_coco_trunk() verifies all 366
#      backbone+FPN tensors against torchvision maskrcnn_resnet50_fpn_v2 COCO_V1
#      BEFORE the swap, so a guard failure is unambiguously the base and not the
#      exported backbone. Previously the base ResNet was randomly initialized (see
#      train_polygons_coco_v023_cleaninit.sbatch for the measurement).
#   2. The trunk comes from the COCO-init pretraining (job 39129719), whose log carries
#      "Verified COCO init: all 281 backbone/FPN tensors match torchvision RetinaNet
#      COCO_V1" -- not the old training/weak_supervision/outputs/pretrain/ trunk, which
#      was exported from a deepforest-tree (NEON) RetinaNet.
#
# Stage 1 is NOT re-run: 39129719 already produced verified COCO-init v0.23 backbones
# for both splits, and they are exactly what this arm needs.
#
# This is the first version of Table 6 where neither arm has seen a NEON tree label and
# both arms have a genuinely pretrained starting point, i.e. the first one that isolates
# the unsupervised boxes rather than "pretraining vs. random init".
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="training/weak_supervision/outputs/polygon_box_pretrained_v023_cleaninit/$SPLIT"
BACKBONE="training/weak_supervision/outputs/pretrain_v023_cocoinit/$SPLIT/box_backbone_${SPLIT}.pt"
mkdir -p "$OUT_DIR"

if [ ! -f "$BACKBONE" ]; then
  echo "Missing COCO-init backbone: $BACKBONE (expected from job 39129719)" >&2
  exit 1
fi

srun uv run python training/polygons/train.py \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme "$SPLIT" \
  --gpus 1 \
  --batch-size 16 \
  --max-epochs 100 \
  --lr 0.01 \
  --num-workers 16 \
  --image-size 448 \
  --init-mode box_pretrained \
  --box-backbone-checkpoint "$BACKBONE" \
  --data-scope subset \
  --output-dir "$OUT_DIR" \
  --comet \
  --comet-name "polygons-${SPLIT}-box_pretrained-lr0.01-v023-cleaninit"
