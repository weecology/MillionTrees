#!/bin/bash
#SBATCH --job-name=mt_eval_points_ft
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

CKPT_DIR="training/points/outputs/${SPLIT}/checkpoints"
# TreeFormer checkpoints are named treeformer-epoch=NN-val_loss=*.ckpt. Restrict to the
# val_loss-monitored files (skip the point_recall ones) and pick the lowest val_loss.
CKPT=$(ls -1 "$CKPT_DIR"/treeformer-epoch=*-val_loss=*.ckpt \
  | awk -F'val_loss=' '{print $2"\t"$0}' \
  | sort -g \
  | head -1 \
  | cut -f2-)

echo "=== Eval fine-tuned points: split=$SPLIT ==="
echo "Checkpoint: $CKPT"

VIZ_DIR="outputs/viz/trained_points/${SPLIT}"
mkdir -p "$VIZ_DIR"

uv run python training/points/eval.py \
  --checkpoint "$CKPT" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --batch-size 16 \
  --num-workers 8 \
  --score-thresh 0.10 \
  --output-dir "training/points/outputs/${SPLIT}" \
  --viz-dir "$VIZ_DIR"
