#!/bin/bash
#SBATCH --job-name=mt_eval_val_deepforest
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

# DeepForest (pretrained release weights, zero-shot) on the held-out TLS validation set
# (Allen et al. 2025 + Frey et al. 2026), boxes. Zero-shot => split-independent;
# --split-scheme is only a required arg. Mirrors eval_deepforest.sbatch (score 0.1) with
# --eval-split validation and a dedicated output dir so test results are untouched.
set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
OUT_BASE=$REPO/existing_models/deepforest/outputs/validation

cd $REPO/existing_models/deepforest

echo "=== DeepForest pretrained eval on validation (Allen+Frey): boxes ==="

uv run python eval_boxes.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme out-of-distribution \
  --eval-split validation \
  --batch-size 12 \
  --num-workers 4 \
  --score-threshold 0.1 \
  --output-dir "$OUT_BASE"

echo "=== done. Results in $OUT_BASE ==="
