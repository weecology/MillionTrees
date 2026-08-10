#!/bin/bash
#SBATCH --job-name=mt_boxes_boxinit_coco_v023
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# Treatment arm of the same-geometry (box -> box) pretraining ablation, v0.23 / AP40.
# COCO RetinaNet with backbone.body.* swapped for the v0.23 unsupervised box-pretrained
# backbone (stage 1: pretrain_backbone_v023.sbatch). FPN and detection heads stay at COCO
# init, so the ONLY difference from train_boxes_coco_v023.sbatch is the ResNet trunk.
#
# Deliberately NOT the deepforest-tree-init variant (train_boxes_box_pretrained.sbatch,
# job 36050607): that one loads NEON-exposed heads and is disqualified as a baseline.
#
# --gradient-clip-val 1.0: the v0.20 run of this arm (36058068_0, within-distribution)
# diverged to NaN at epoch <=11 and scored 0.000 on every metric. Backbones were verified
# finite, so the divergence was optimizer instability at lr 0.01, not bad weights.
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
BACKBONE="training/weak_supervision/outputs/pretrain_v023/$SPLIT/box_backbone_${SPLIT}.pt"
OUT_DIR="training/boxes/outputs/box_pretrained_coco_v023/$SPLIT"
mkdir -p "$OUT_DIR"

if [ ! -f "$BACKBONE" ]; then
  echo "ERROR: v0.23 backbone not found: $BACKBONE" >&2
  echo "Run training/slurm/pretrain_backbone_v023.sbatch first." >&2
  exit 1
fi

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
  --init-mode box_pretrained \
  --box-backbone-checkpoint "$BACKBONE" \
  --output-dir "$OUT_DIR" \
  --comet \
  --comet-name "boxes-${SPLIT}-box_pretrained_coco-lr0.01-v023-ablation"
