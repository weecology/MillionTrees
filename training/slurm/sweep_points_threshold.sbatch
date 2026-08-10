#!/bin/bash
#SBATCH --job-name=mt_sweep_points_thresh
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
THRESHOLDS=(0.1 0.2)
THRESH=${THRESHOLDS[$SLURM_ARRAY_TASK_ID]}

CKPT="training/points/outputs/within-distribution/checkpoints/treeformer-epoch=13-point_recall=0.2265.ckpt"
OUT_DIR="training/points/outputs/within-distribution/sweep/thresh_${THRESH}"
mkdir -p "$OUT_DIR"

echo "=== eval_score_threshold sweep: thresh=$THRESH ==="
echo "Checkpoint: $CKPT"

uv run --group treeformer python training/points/eval.py \
  --checkpoint "$CKPT" \
  --root-dir "$ROOT_DIR" \
  --split-scheme within-distribution \
  --eval-score-threshold "$THRESH" \
  --batch-size 16 \
  --num-workers 8 \
  --score-thresh 0.10 \
  --output-dir "$OUT_DIR" \
  --viz-dir ""
