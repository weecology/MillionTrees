#!/bin/bash
#SBATCH --job-name=mt_poly_deepforest
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

# Polygon Mask R-CNN trained through the DeepForest polygon stack
# (PolygonDataset panoptic targets + deepforest.models.maskrcnn). Recipe lives
# in training/polygons/deepforest_polygon.yaml (OAM-TCD alignment).
#
# Single-GPU (2026-06-25): the 2x B200 DDP version (35722209) kept dying on the
# NCCL ALLREDUCE watchdog timeout (/orange-contention straggler hang, see
# project_nccl_hang_orange_contention), as did both Table 6 arms. Single GPU
# sidesteps it. Keep batch 16/GPU (already proven to fit) and linear-scale lr to
# 0.01 for the now-effective batch 16 (2x validated base 8 -> 2x base 0.005);
# workers 16 keep the input pipeline ahead of the GPU.
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="training/polygons/outputs/deepforest/$SPLIT"
mkdir -p "$OUT_DIR"

srun uv run python training/polygons/train.py \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme "$SPLIT" \
  --gpus 1 \
  --batch-size 16 \
  --lr 0.01 \
  --max-epochs 100 \
  --num-workers 16 \
  --image-size 448 \
  --init-mode coco \
  --data-scope subset \
  --output-dir "$OUT_DIR" \
  --comet \
  --comet-name "polygons-${SPLIT}-pretrained-lr0.01-deepforest-1gpu"
