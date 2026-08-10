#!/bin/bash
#SBATCH --job-name=mt_polygons_coco_aug
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# COCO-pretrained polygon baseline WITH train-time augmentation (flips, 90-deg
# rotations, mild brightness/contrast) and checkpoint selection on val_mask_acc
# instead of val_loss. Separate output dir from polygon_coco/ so the non-augmented
# baseline (random 0.424 / zeroshot 0.138 mask_acc) stays intact for comparison.
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SPLITS=(random zeroshot)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="training/weak_supervision/outputs/polygon_coco_aug/$SPLIT"
mkdir -p "$OUT_DIR"

uv run python training/polygons/train.py \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme "$SPLIT" \
  --batch-size 8 \
  --max-epochs 40 \
  --lr 0.001 \
  --num-workers 8 \
  --image-size 448 \
  --init-mode coco \
  --augment \
  --data-scope subset \
  --output-dir "$OUT_DIR" \
  --comet
