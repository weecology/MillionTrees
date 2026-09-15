#!/bin/bash
#SBATCH --job-name=mt_boxes_imagenet_v024
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# v0.24 re-run of the COLD-START floor rung of the box ablation (was
# train_boxes_imagenet_v023.sbatch). ImageNet ResNet50 trunk, freshly initialized
# FPN and heads, no detection pretraining of any kind -- this is the rung that
# measured "removing detection pretraining entirely costs only 0.020 AP40 WD /
# 0.006 OOD", i.e. there is no headroom for weak-label pretraining to occupy, and
# it is what manuscript Table 6 calls "an ImageNet backbone".
#
# Byte-identical to the v0.23 script except the dataset version (loader default,
# now 0.24 -- repairs the OOD split), the _v024 output dir, and the comet name.
# Recipe matches the converged ablation arms (batch 32, max-epochs 200, lr 0.01,
# clip 1.0); OOD gets patience 15, WD gets 10.
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="training/boxes/outputs/imagenet_v024/$SPLIT"
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
  --comet-name "boxes-${SPLIT}-imagenet-lr0.01-v024"
