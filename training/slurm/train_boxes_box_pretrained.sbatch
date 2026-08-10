#!/bin/bash
#SBATCH --job-name=mt_boxes_boxinit
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# Box pretraining ablation (Table 6 same-geometry arm).
# Stage 1 (backbone pretrain on NEON unsupervised) already complete:
#   training/weak_supervision/outputs/pretrain/$SPLIT/box_backbone_$SPLIT.pt
# Stage 2 (this script): load that backbone into DeepForest RetinaNet,
# fine-tune on supervised-only TreeBoxes, evaluate.  Comparison arm is the
# standard deepforest-tree COCO init in training/boxes/outputs/$SPLIT.
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
BACKBONE="training/weak_supervision/outputs/pretrain/$SPLIT/box_backbone_${SPLIT}.pt"
OUT_DIR="training/boxes/outputs/box_pretrained/$SPLIT"
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
  --init-mode box_pretrained \
  --box-backbone-checkpoint "$BACKBONE" \
  --output-dir "$OUT_DIR" \
  --comet \
  --comet-name "boxes-${SPLIT}-box_pretrained-lr0.01"
