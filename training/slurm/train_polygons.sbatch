#!/bin/bash
#SBATCH --job-name=mt_train_polygons
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
mkdir -p training/polygons/outputs

# hpg-turin L4 GPUs (24GB) OOM on Mask R-CNN even at batch_size 1; use a larger
# B200 GPU (180GB) and reduce allocator fragmentation. The B200 headroom lets us
# run a much larger batch than the L4 could.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLITS=(random zeroshot)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

uv run python training/polygons/train.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --batch-size 8 \
  --max-epochs 40 \
  --lr 2e-4 \
  --num-workers 8 \
  --image-size 448 \
  --output-dir "training/polygons/outputs/$SPLIT" \
  --comet
