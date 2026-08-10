#!/bin/bash
# Bohlman-clean fine-tuned TreeFormer point eval at the leaderboard resolution
# (image_size=672), full TreePoints v0.18, random + zeroshot. The E7 imgsize
# sweep (job 34709163) ran @672 but predates the Bohlman 2008 removal AND used
# the best-point_recall checkpoint. This re-runs now that Bohlman is dropped,
# using the lowest-val_loss checkpoint — the same selection eval_points.sbatch
# uses for the leaderboard "Yes" rows — so the manuscript table3 finetuned row
# is apples-to-apples with the pretrained "No" row (eval_treeformer_points_672_clean).
# Caveat: checkpoints were TRAINED at 448, so 672 is a train/eval mismatch; a
# clean test needs a retrain at 672 (follow-up). Inference only.
#SBATCH --job-name=mt_eval_points_ft_672_clean
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
SPLITS=(random zeroshot)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

CKPT_DIR="training/points/outputs/${SPLIT}/checkpoints"
# Match eval_points.sbatch: lowest-val_loss checkpoint (skip the point_recall ones).
CKPT=$(ls -1 "$CKPT_DIR"/treeformer-epoch=*-val_loss=*.ckpt \
  | awk -F'val_loss=' '{print $2"\t"$0}' \
  | sort -g \
  | head -1 \
  | cut -f2-)

echo "=== Eval fine-tuned points (Bohlman-clean) split=$SPLIT image_size=672 ==="
echo "Checkpoint: $CKPT"

OUT_DIR="training/points/outputs/${SPLIT}/eval_ft_img672_clean"
uv run python training/points/eval.py \
  --checkpoint "$CKPT" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --score-thresh 0.10 \
  --score-integration-radius 2 \
  --image-size 672 \
  --batch-size 8 \
  --num-workers 8 \
  --viz-dir "" \
  --output-dir "$OUT_DIR"

echo "=== Done. Results in $OUT_DIR ==="
