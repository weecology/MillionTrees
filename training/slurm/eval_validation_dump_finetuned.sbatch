#!/bin/bash
#SBATCH --job-name=mt_valdump_boxes_ft
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=90GB
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Fine-tuned OOD box model (the leaderboard DeepForest row) on the validation split with
# --save-predictions, so scripts/rescore_validation_ap.py can compute AP50/AP40 on all
# tiles vs. only the completely-annotated ones. Mirrors eval_validation_boxes.sbatch.
set -euo pipefail

export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

cd /blue/ewhite/b.weinstein/src/MillionTrees

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLIT=out-of-distribution
DUMPS=/blue/ewhite/b.weinstein/src/MillionTrees/outputs/validation_preds

CKPT="${CKPT:-training/boxes/outputs/${SPLIT}/checkpoints/boxes-best-v2.ckpt}"
if [ ! -f "$CKPT" ]; then echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; fi
echo "=== Fine-tuned boxes validation eval + prediction dump: split=$SPLIT ==="
echo "Checkpoint: $CKPT"

uv run python training/boxes/eval.py \
  --checkpoint "$CKPT" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --eval-split validation \
  --batch-size 16 \
  --num-workers 8 \
  --score-threshold 0.1 \
  --output-dir "training/boxes/outputs/${SPLIT}" \
  --viz-dir "" \
  --save-predictions "$DUMPS/deepforest_finetuned_boxes_validation.pkl"

echo "=== done. Dump in $DUMPS ==="
