#!/bin/bash
#SBATCH --job-name=mt_eval_val_detectree2
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

# Detectree2 (pretrained, zero-shot) on the held-out TLS validation set
# (Allen et al. 2025 + Frey et al. 2026). Zero-shot => split-independent; --split-scheme
# is only a required arg. Mirrors eval_detectree2.sbatch (score 0.1) but --eval-split
# validation and a dedicated output dir so test results are untouched.
set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
OUT_BASE=$REPO/existing_models/detectree2/outputs/validation

cd $REPO/existing_models/detectree2

MODEL=250312_flexi.pth
if [ ! -f "$MODEL" ]; then
  wget -q https://zenodo.org/records/15863800/files/$MODEL
fi

echo "=== Detectree2 eval on validation (Allen+Frey) ==="

uv run --no-sync python eval_polygons.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme out-of-distribution \
  --eval-split validation \
  --device cuda \
  --batch-size 8 \
  --model-path "$MODEL" \
  --score-threshold 0.1 \
  --output-dir "$OUT_BASE"

echo "=== done. Results in $OUT_BASE ==="
