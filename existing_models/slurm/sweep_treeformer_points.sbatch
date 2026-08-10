#!/bin/bash
# Score-threshold sweep: pretrained TreeFormer on points, random split, 3 imgs/source.
#SBATCH --job-name=mt_sweep_treeformer
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLIT=random
OUT_BASE=$REPO/existing_models/treeformer/outputs/sweep

cd "$REPO/existing_models/treeformer"

echo "=== TreeFormer threshold sweep: split=$SPLIT ==="
uv run python eval_points.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --per-source 3 \
  --sweep \
  --output-dir "$OUT_BASE"

echo "=== TreeFormer sweep done -> $OUT_BASE/threshold_sweep.csv ==="
