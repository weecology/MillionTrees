#!/bin/bash
#SBATCH --job-name=mt_boxes_boxinit_coco_v023_cleaninit
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# Treatment arm of the CORRECTED same-geometry (box -> box) ablation, v0.23 / AP40.
# True COCO RetinaNet with backbone.body.* swapped for the COCO-init unsupervised
# box-pretrained trunk from pretrain_backbone_v023_cocoinit.sbatch. FPN and heads stay
# at COCO init, so the ONLY difference from train_boxes_coco_v023_cleaninit.sbatch is
# the ResNet trunk — and unlike job 38840965, neither side has seen a NEON tree label.
#
# --gradient-clip-val 1.0: the v0.20 run of this arm (36058068_0) diverged to NaN at
# lr 0.01. Applied to both arms so the comparison stays apples-to-apples.
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
BACKBONE="training/weak_supervision/outputs/pretrain_v023_cocoinit/$SPLIT/box_backbone_${SPLIT}.pt"
OUT_DIR="training/boxes/outputs/box_pretrained_coco_v023_cleaninit/$SPLIT"
mkdir -p "$OUT_DIR"

if [ ! -f "$BACKBONE" ]; then
  echo "ERROR: COCO-init v0.23 backbone not found: $BACKBONE" >&2
  echo "Run training/slurm/pretrain_backbone_v023_cocoinit.sbatch first." >&2
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
  --comet-name "boxes-${SPLIT}-box_pretrained_coco-lr0.01-v023-cleaninit"
