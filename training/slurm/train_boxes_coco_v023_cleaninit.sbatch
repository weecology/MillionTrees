#!/bin/bash
#SBATCH --job-name=mt_boxes_coco_v023_cleaninit
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# Control arm of the CORRECTED same-geometry (box -> box) ablation, v0.23 / AP40.
#
# The predecessor (train_boxes_coco_v023.sbatch, job 38840964) claimed "clean COCO,
# no NEON exposure" and was wrong: DeepForest's config default is
# model.name = "weecology/deepforest-tree", and deepforest.__init__ -> create_model()
# -> load_model() pulls it before train.py can intervene. Not calling load_model()
# does not undo that. train.py now sets model.name=None for --init-mode coco and
# verifies against torchvision COCO_V1 (assert_coco_trunk), so this arm really is
# a tree-naive baseline. Same recipe otherwise; new output dir keeps 38840964
# reproducible.
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="training/boxes/outputs/coco_v023_cleaninit/$SPLIT"
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
  --comet-name "boxes-${SPLIT}-coco-lr0.01-v023-cleaninit"
