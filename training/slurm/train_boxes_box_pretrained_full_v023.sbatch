#!/bin/bash
#SBATCH --job-name=mt_boxes_boxinit_full_v023
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# Treatment arm of the WHOLE-NETWORK box -> box ablation, v0.23 / AP40.
#
# Supersedes train_boxes_box_pretrained_coco_v023_cleaninit.sbatch (job 39129721), which
# transferred only backbone.body (265 of 301 tensors). That truncation was inherited from
# the polygon ablation — pretrain_backbone_for_polygons.py was written for box -> Mask
# R-CNN, where the ResNet trunk is the only architecturally compatible piece. In box ->
# box both stages are the identical DeepForest RetinaNetHub with num_classes=1, so
# nothing forced it: the FPN was silently reset to COCO and BOTH detection heads to
# random init (RetinaNetHub always builds fresh heads; COCO's 91-class head does not fit
# a 1-class model). Every layer encoding "this is a tree and this is its extent" was
# discarded, which plausibly explains 39129721's null result (AP40 -0.016 WD, -0.015 OOD).
#
# --init-mode box_pretrained_full loads all 301 tensors with strict=True and refuses a
# trunk-only export, so the truncation cannot come back unnoticed.
#
# NO CONTROL ARM IS RE-RUN: the pretraining stage still starts from COCO, so the finished
# COCO arm (39129720: AP40 0.476 WD / 0.517 OOD) is exactly "the thing pretraining started
# from, fine-tuned" and remains the correct baseline.
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
NETWORK="training/weak_supervision/outputs/pretrain_v023_cocoinit_full/$SPLIT/box_network_${SPLIT}.pt"
OUT_DIR="training/boxes/outputs/box_pretrained_full_v023/$SPLIT"
mkdir -p "$OUT_DIR"

if [ ! -f "$NETWORK" ]; then
  echo "ERROR: full-network export not found: $NETWORK" >&2
  echo "Run training/slurm/pretrain_backbone_v023_cocoinit_full.sbatch first." >&2
  echo "(box_backbone_*.pt is the trunk-only export and will be REJECTED here.)" >&2
  exit 1
fi

# Same lr/batch/patience as 39129720 and 39129721 so the three-way comparison —
# COCO / trunk-only / whole-network — differs in exactly one factor.
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
  --init-mode box_pretrained_full \
  --box-backbone-checkpoint "$NETWORK" \
  --output-dir "$OUT_DIR" \
  --comet \
  --comet-name "boxes-${SPLIT}-box_pretrained_full-lr0.01-v023"
