#!/bin/bash
# Re-evaluate the existing (448-trained) fine-tuned TreeFormer point checkpoints
# at image_size=896. Per the points diagnosis, the density grid at 448 (112px)
# under-resolves dense canopy and systematically under-counts; re-scoring the
# pretrained model at 896 ~doubled recall and halved counting MAE. This is a
# train/eval resolution MISMATCH (these ckpts were trained at 448) — it gives a
# quick signal, but the clean comparison is the matched 896-trained model
# (train_points_896.sbatch). Writes to *_img896 dirs so it never clobbers the
# 448 leaderboard results/viz.
#SBATCH --job-name=mt_eval_points_img896
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

CKPT_DIR="training/points/outputs/${SPLIT}/checkpoints"
# Checkpoints are named treeformer-epoch=NN-point_recall=*.ckpt; pick the highest recall.
CKPT=$(ls -1 "$CKPT_DIR"/treeformer-epoch=*-point_recall=*.ckpt \
  | awk -F'point_recall=' '{print $2"\t"$0}' \
  | sort -rg \
  | head -1 \
  | cut -f2-)

echo "=== Eval fine-tuned points @896: split=$SPLIT ==="
echo "Checkpoint: $CKPT"

VIZ_DIR="outputs/viz/trained_points_img896/${SPLIT}"
mkdir -p "$VIZ_DIR"

uv run python training/points/eval.py \
  --checkpoint "$CKPT" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --image-size 896 \
  --batch-size 8 \
  --num-workers 8 \
  --score-thresh 0.10 \
  --output-dir "training/points/outputs/${SPLIT}_img896" \
  --viz-dir "$VIZ_DIR"
