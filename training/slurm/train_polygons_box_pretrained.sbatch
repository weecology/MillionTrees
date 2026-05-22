#!/bin/bash
#SBATCH --job-name=mt_polygons_boxinit
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SPLITS=(random zeroshot)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="training/weak_supervision/outputs/polygon_box_pretrained/$SPLIT"
BACKBONE="training/weak_supervision/outputs/pretrain/$SPLIT/box_backbone_${SPLIT}.pt"
mkdir -p "$OUT_DIR"

uv run python training/polygons/train_polygons.py \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme "$SPLIT" \
  --batch-size 24 \
  --max-epochs 40 \
  --lr 1e-4 \
  --num-workers 8 \
  --image-size 448 \
  --init-mode box_pretrained \
  --box-backbone-checkpoint "$BACKBONE" \
  --data-scope subset \
  --output-dir "$OUT_DIR" \
  --comet
