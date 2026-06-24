#!/bin/bash
#SBATCH --job-name=mt_polygons_coco
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:2

# 2-GPU bump (2026-06-24): single-GPU batch-24 arms (35540850) crashed on
# read-during-write during packaging AND left the B200 data-starved. Aligned to
# the same 2x B200 DDP recipe as the box-pretrained arm (batch 16/GPU =
# effective 32, lr 0.02, workers 16/rank) so the Table 6 comparison is
# apples-to-apples. Runs BOTH splits: the Table 6 coco rows must live under
# weak_supervision/outputs/polygon_coco_deepforest/, which the main DeepForest
# train (outputs/deepforest/, job 35646340) does NOT populate.
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
  --gpus 2 \
  --batch-size 16 \
  --max-epochs 100 \
  --lr 0.02 \
  --num-workers 16 \
  --image-size 448 \
  --init-mode coco \
  --data-scope subset \
  --output-dir "$OUT_DIR" \
  --comet \
  --comet-name "polygons-${SPLIT}-pretrained-lr0.02-deepforest-2gpu"
