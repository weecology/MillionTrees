#!/bin/bash
#SBATCH --job-name=mt_eval_val_sam3_points
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

# SAM3 (pretrained, zero-shot) POINTS ONLY on the held-out TLS validation set
# (Allen et al. 2025 + Frey et al. 2026, 85 images).
#
# Why points-only rather than eval_validation_existing_sam3.sbatch: the box and polygon
# results in $OUT_BASE were already regenerated on v0.22 by the 2026-07-27 dump/re-score
# runs (jobs 38128357 / 38129319). Only the points row is still v0.21, and the all-geometry
# script would spend ~8h re-running boxes+polygons and clobber their current v0.22 results.
#
# Zero-shot => split-independent; --split-scheme is only a required arg. Config matches the
# superseded v0.21 points run (score 0.1, batch 8) so the numbers differ by dataset version
# only. Picks up v0.22 automatically (loader defaults to the newest _versions_dict key).
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

echo "=== SAM3 eval on validation (Allen+Frey), points only, v0.22 ==="

uv run python eval_points.py \
  --root-dir "$ROOT_DIR" --split-scheme out-of-distribution --eval-split validation \
  --device cuda --batch-size 8 --score-threshold 0.1 --output-dir "$OUT_BASE" \
  --viz-dir "$OUT_BASE/viz_points"

echo "=== done: $OUT_BASE/results_points_out-of-distribution.txt ==="
