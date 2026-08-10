#!/bin/bash
#SBATCH --job-name=mt_eval_points_thresh01
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

# Evaluate all remaining TreeFormer rows at eval_score_threshold=0.1.
# The within-distribution fine-tuned row was already run in the sweep.
# Task 0: fine-tuned OOD
# Task 1: pretrained (KCL) within-distribution
# Task 2: pretrained (KCL) out-of-distribution

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
KCL_CKPT=/home/veitchmichaelisj/code/DeepForest_jvm/kcl_hf_checkpoint
FT_OOD_CKPT="training/points/outputs/out-of-distribution/checkpoints/treeformer-epoch=00-point_recall=0.1286.ckpt"

case $SLURM_ARRAY_TASK_ID in
  0)
    LABEL="finetuned-ood"
    CKPT="$FT_OOD_CKPT"
    SPLIT="out-of-distribution"
    OUT_DIR="training/points/outputs/out-of-distribution/sweep/thresh_0.1"
    ;;
  1)
    LABEL="pretrained-within"
    CKPT="$KCL_CKPT"
    SPLIT="within-distribution"
    OUT_DIR="existing_models/treeformer/outputs/within-distribution/sweep/thresh_0.1"
    ;;
  2)
    LABEL="pretrained-ood"
    CKPT="$KCL_CKPT"
    SPLIT="out-of-distribution"
    OUT_DIR="existing_models/treeformer/outputs/out-of-distribution/sweep/thresh_0.1"
    ;;
esac

mkdir -p "$OUT_DIR"
echo "=== TreeFormer eval at thresh=0.1: $LABEL split=$SPLIT ==="
echo "Checkpoint: $CKPT"

uv run --group treeformer python training/points/eval.py \
  --checkpoint "$CKPT" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --eval-score-threshold 0.1 \
  --batch-size 16 \
  --num-workers 8 \
  --score-thresh 0.10 \
  --output-dir "$OUT_DIR" \
  --viz-dir ""
