#!/bin/bash
#SBATCH --job-name=mt_poly_deepforest_resize640_v20
#SBATCH --output=/home/b.weinstein/logs/%x_%A.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

# CONTROL for the within-distribution resize1280 polygon run (35797052_0), whose
# train loss is climbing / metrics look awful. Reverts ONLY the image size back
# to the previously-proven 640 resize recipe (batch 16, lr 0.01, 100 epochs,
# resize train + scale-matched resize eval, coco init, single-GPU) while keeping
# the NEW v0.20 data (loader default). If 640 is healthy on v0.20, the 1280 jump
# is the regression, not the data; if it is equally bad, the v0.20 data is.
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

SPLIT=within-distribution
OUT_DIR="training/polygons/outputs/deepforest_resize640_v20/$SPLIT"
mkdir -p "$OUT_DIR"

srun uv run python training/polygons/train.py \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme "$SPLIT" \
  --gpus 1 \
  --batch-size 16 \
  --lr 0.01 \
  --max-epochs 100 \
  --num-workers 16 \
  --image-size 640 \
  --train-aug resize \
  --eval-inference resize \
  --init-mode coco \
  --data-scope subset \
  --output-dir "$OUT_DIR" \
  --comet \
  --comet-name "polygons-within-distribution-pretrained-lr0.01-deepforest-resize640-v20control"
