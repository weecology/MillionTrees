#!/bin/bash
#SBATCH --job-name=mt_eval_polygons_ft
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

CKPT_DIR="training/polygons/outputs/${SPLIT}/checkpoints"
# Checkpoints are named polygons-epoch=NN-val_mask_acc=*.ckpt; pick the highest accuracy.
CKPT=$(ls -1 "$CKPT_DIR"/polygons-epoch=*-val_mask_acc=*.ckpt \
  | awk -F'val_mask_acc=' '{print $2"\t"$0}' \
  | sort -rg \
  | head -1 \
  | cut -f2-)

echo "=== Eval fine-tuned polygons: split=$SPLIT ==="
echo "Checkpoint: $CKPT"

VIZ_DIR="outputs/viz/trained_polygons/${SPLIT}"
mkdir -p "$VIZ_DIR"

uv run python training/polygons/eval.py \
  --checkpoint "$CKPT" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --batch-size 4 \
  --num-workers 4 \
  --image-size 448 \
  --score-threshold 0.1 \
  --output-dir "training/polygons/outputs/${SPLIT}" \
  --viz-dir "$VIZ_DIR"
