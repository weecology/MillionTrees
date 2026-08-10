#!/bin/bash
#SBATCH --job-name=mt_poly_df_evalsweep
#SBATCH --output=/home/b.weinstein/logs/%x_%A.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

# Evaluator score-threshold sweep for the DeepForest polygon Mask R-CNN.
# The leaderboard run scored at the default eval_score_threshold=0.5, which
# hides every prediction in [0.05, 0.5) and makes the model look "timid"
# (within-distribution AP50 0.097). This single-inference-pass sweep re-scores
# the same predictions at several evaluator thresholds to find the F1/AP50
# operating point. roi_heads.score_thresh is auto-dropped to 0.01 so the full
# score distribution reaches the evaluator.

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SPLIT=within-distribution
CKPT_DIR="training/polygons/outputs/deepforest/${SPLIT}/logs/checkpoints"
# Most-negative... here val_loss is positive, so pick the lowest val_loss among
# converged checkpoints (epoch >= 1).
CKPT=$(ls -1 "$CKPT_DIR"/polygons-epoch=*.ckpt \
  | awk -F'epoch=' '{split($2,a,"-"); if (a[1]+0 > 0) print}' \
  | awk -F'val_loss=' '{print $2"\t"$0}' \
  | sort -g \
  | head -1 \
  | cut -f2-)

OUT_DIR="training/polygons/outputs/deepforest/${SPLIT}/sweep"
mkdir -p "$OUT_DIR"

echo "=== Eval-threshold sweep: split=$SPLIT ==="
echo "Checkpoint: $CKPT"

uv run python training/polygons/eval.py \
  --checkpoint "$CKPT" \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme "$SPLIT" \
  --batch-size 4 \
  --num-workers 8 \
  --image-size 448 \
  --sweep-eval-thresholds "0.05,0.1,0.2,0.3,0.4,0.5" \
  --output-dir "$OUT_DIR"
