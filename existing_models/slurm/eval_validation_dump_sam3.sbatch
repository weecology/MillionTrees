#!/bin/bash
#SBATCH --job-name=mt_valdump_sam3
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

# SAM3 zero-shot on the validation split (boxes + polygons) with --save-predictions,
# for the AP50/AP40 x complete-tile re-scoring. Mirrors eval_validation_existing_sam3.sbatch.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
OUT_BASE=$REPO/existing_models/sam3/outputs/validation
DUMPS=$REPO/outputs/validation_preds

cd $REPO/existing_models/sam3

HF_TOKEN_FILE=$REPO/.hf_token
if [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
fi

echo "=== SAM3 validation eval + prediction dump (score 0.1) ==="

uv run python eval_boxes.py \
  --root-dir "$ROOT_DIR" --split-scheme out-of-distribution --eval-split validation \
  --device cuda --batch-size 8 --score-threshold 0.1 --output-dir "$OUT_BASE" \
  --viz-dir "$OUT_BASE/viz_boxes" \
  --save-predictions "$DUMPS/sam3_boxes_validation.pkl"

uv run python eval_polygons.py \
  --root-dir "$ROOT_DIR" --split-scheme out-of-distribution --eval-split validation \
  --device cuda --batch-size 4 --score-threshold 0.1 --output-dir "$OUT_BASE" \
  --viz-dir "$OUT_BASE/viz_polygons" \
  --save-predictions "$DUMPS/sam3_polygons_validation.pkl"

echo "=== done. Dumps in $DUMPS ==="
