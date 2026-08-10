#!/bin/bash
#SBATCH --job-name=mt_eval_val_sam3
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

# SAM3 (pretrained, zero-shot) on the held-out TLS validation set (Allen et al. 2025 +
# Frey et al. 2026) for ALL three geometries. Zero-shot => split-independent; --split-scheme
# is only a required arg. Mirrors eval_sam3.sbatch (score 0.1) but --eval-split validation
# and a dedicated output dir so test results are untouched.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
OUT_BASE=$REPO/existing_models/sam3/outputs/validation

cd $REPO/existing_models/sam3

# SAM3 (facebook/sam3) is gated; authenticate from the untracked token file.
HF_TOKEN_FILE=$REPO/.hf_token
if [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
fi

echo "=== SAM3 eval on validation (Allen+Frey): boxes + points + polygons ==="

# Per-geometry --viz-dir: all three write to $OUT_BASE, so a shared default viz/ would
# be overwritten by the last stage. Keep them separate for side-by-side review.
uv run python eval_boxes.py \
  --root-dir "$ROOT_DIR" --split-scheme out-of-distribution --eval-split validation \
  --device cuda --batch-size 8 --score-threshold 0.1 --output-dir "$OUT_BASE" \
  --viz-dir "$OUT_BASE/viz_boxes"

uv run python eval_points.py \
  --root-dir "$ROOT_DIR" --split-scheme out-of-distribution --eval-split validation \
  --device cuda --batch-size 8 --score-threshold 0.1 --output-dir "$OUT_BASE" \
  --viz-dir "$OUT_BASE/viz_points"

uv run python eval_polygons.py \
  --root-dir "$ROOT_DIR" --split-scheme out-of-distribution --eval-split validation \
  --device cuda --batch-size 4 --score-threshold 0.1 --output-dir "$OUT_BASE" \
  --viz-dir "$OUT_BASE/viz_polygons"

echo "=== done. Results in $OUT_BASE ==="
