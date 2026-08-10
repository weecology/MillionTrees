#!/bin/bash
#SBATCH --job-name=mt_valdump_deepforest
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

# DeepForest pretrained (zero-shot) on the validation split with --save-predictions,
# for the AP50/AP40 x complete-tile re-scoring. Mirrors eval_validation_existing_deepforest.sbatch.
set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
OUT_BASE=$REPO/existing_models/deepforest/outputs/validation
DUMPS=$REPO/outputs/validation_preds

cd $REPO/existing_models/deepforest

echo "=== DeepForest pretrained validation eval + prediction dump (score 0.1) ==="

uv run python eval_boxes.py \
  --root-dir "$ROOT_DIR" --split-scheme out-of-distribution --eval-split validation \
  --batch-size 12 --num-workers 4 --score-threshold 0.1 --output-dir "$OUT_BASE" \
  --save-predictions "$DUMPS/deepforest_pretrained_boxes_validation.pkl"

echo "=== done. Dumps in $DUMPS ==="
