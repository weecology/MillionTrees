#!/bin/bash
#SBATCH --job-name=mt_eval_val_canopyrs
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# CanopyRS (DINO boxes + DINO/SAM3 SelvaMask polygons; pretrained, zero-shot) on the
# held-out TLS validation set (Allen et al. 2025 + Frey et al. 2026). Zero-shot =>
# split-independent; --split-scheme is only a required arg. Matches eval_canopyrs.sbatch:
# score_threshold=0.30 (best-F1 operating point from the full-test-set sweep, NOT the 0.1
# used by other pretrained models). Dedicated output dir so test results are untouched.
set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
OUT_BASE=$REPO/existing_models/canopyrs/outputs/validation

module load cuda/12.8.1 gdal/3.7.0
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH="$CUDA_HOME/bin:$PATH"

cd "$REPO/existing_models/canopyrs"
source .venv/bin/activate

HF_TOKEN_FILE=$REPO/.hf_token
if [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
fi

echo "=== CanopyRS eval on validation (Allen+Frey): boxes + polygons, score 0.30 ==="

python eval_boxes.py \
  --root-dir "$ROOT_DIR" --split-scheme out-of-distribution --eval-split validation \
  --device cuda --batch-size 4 --score-threshold 0.30 \
  --output-dir "$OUT_BASE" --viz-dir "$OUT_BASE/viz_boxes"

python eval_polygons.py \
  --root-dir "$ROOT_DIR" --split-scheme out-of-distribution --eval-split validation \
  --device cuda --batch-size 2 --score-threshold 0.30 \
  --output-dir "$OUT_BASE" --viz-dir "$OUT_BASE/viz_polygons"

echo "=== done. Results in $OUT_BASE ==="
