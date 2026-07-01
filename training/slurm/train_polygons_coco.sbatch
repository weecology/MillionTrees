#!/bin/bash
#SBATCH --job-name=mt_polygons_coco
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

# Single-GPU (2026-06-25): the 2x B200 DDP version (35646623) died on the NCCL
# ALLGATHER/ALLREDUCE watchdog timeout (/orange-contention straggler hang), like
# the box-pretrained arm. Single GPU sidesteps it. Keep batch 16, linear-scale
# lr to 0.01 (effective batch 16 = 2x validated base 8 -> 2x base 0.005),
# workers 16. Box-pretrained and coco arms share this identical single-GPU
# recipe so the Table 6 comparison stays apples-to-apples. Runs BOTH splits: the
# Table 6 coco rows must live under weak_supervision/outputs/
# polygon_coco_deepforest/, which the main DeepForest train does NOT populate.
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
# DeepForest polygon stack. Outputs to a *_deepforest dir to preserve the old
# torchvision-stack Table 6 results.
OUT_DIR="training/weak_supervision/outputs/polygon_coco_deepforest/$SPLIT"
mkdir -p "$OUT_DIR"

srun uv run python training/polygons/train.py \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme "$SPLIT" \
  --gpus 1 \
  --batch-size 16 \
  --max-epochs 100 \
  --lr 0.01 \
  --num-workers 16 \
  --image-size 448 \
  --init-mode coco \
  --data-scope subset \
  --output-dir "$OUT_DIR" \
  --comet \
  --comet-name "polygons-${SPLIT}-pretrained-lr0.01-deepforest-1gpu"
