#!/bin/bash
# v0.20 leaderboard cross-geometry row: SAM3 polygon eval on crossgeometry split.
#SBATCH --job-name=mt_eval_sam3_crossgeom
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
OUT_BASE=$REPO/existing_models/sam3/outputs/crossgeometry

cd $REPO/existing_models/sam3

echo "=== SAM3 polygon eval: split=crossgeometry ==="

uv run python eval_polygons.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme crossgeometry \
  --device cuda \
  --batch-size 4 \
  --score-threshold 0.1 \
  --output-dir "$OUT_BASE"

echo "=== Done. Results in $OUT_BASE ==="
