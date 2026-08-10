#!/bin/bash
#SBATCH --job-name=mt_polygons_boxinit_safecrop
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

# Table 6 box-pretrained arm with annotation-safe crop.
# Paired with train_polygons_deepforest_annotationsafecrop.sbatch (COCO arm, job 35998098)
# so the COCO vs box-pretrained backbone comparison is apples-to-apples on the same
# augmentation. Backbone .pt files already exist from Jun 21-22 runs — no retraining
# needed because backbone pretraining bypasses DeepForest's dataset/augmentation stack.
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="training/weak_supervision/outputs/polygon_box_pretrained_annotationsafecrop/$SPLIT"
BACKBONE="training/weak_supervision/outputs/pretrain/$SPLIT/box_backbone_${SPLIT}.pt"
mkdir -p "$OUT_DIR"

srun uv run python training/polygons/train.py \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme "$SPLIT" \
  --gpus 1 \
  --batch-size 16 \
  --max-epochs 100 \
  --lr 0.01 \
  --num-workers 8 \
  --image-size 448 \
  --train-aug annotationsafecrop \
  --eval-inference tiled \
  --init-mode box_pretrained \
  --box-backbone-checkpoint "$BACKBONE" \
  --data-scope subset \
  --output-dir "$OUT_DIR" \
  --comet \
  --comet-name "polygons-${SPLIT}-box_pretrained-lr0.01-annotationsafecrop"
