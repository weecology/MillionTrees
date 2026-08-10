#!/bin/bash
#SBATCH --job-name=mt_ap_iou_compare_deepforest
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

# AP50 (IoU 0.5) vs AP40 (IoU 0.4) sensitivity check for pretrained DeepForest on the
# held-out TLS validation datasets (Allen et al. 2025 + Frey et al. 2026), boxes.
# Runs the model once and scores the same predictions at both IoU match thresholds,
# overall and per source. Informs whether restricting AP50 to only the validation
# datasets is robust to the IoU threshold choice. Score 0.1 matches the leaderboard eval.
set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
OUT_BASE=$REPO/existing_models/deepforest/outputs/ap_iou_compare

cd $REPO/existing_models/deepforest

echo "=== DeepForest pretrained: AP50 vs AP40 on validation (Allen+Frey) ==="

uv run python compare_ap_iou.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme out-of-distribution \
  --eval-split validation \
  --batch-size 12 \
  --num-workers 4 \
  --score-threshold 0.1 \
  --output-dir "$OUT_BASE"

echo "=== done. Results in $OUT_BASE ==="
