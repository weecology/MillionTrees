#!/bin/bash
#SBATCH --job-name=mt_eval_sam3_poly_rand
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLIT=random
OUT_BASE=$REPO/existing_models/sam3/outputs/$SPLIT

cd $REPO/existing_models/sam3

echo "=== SAM3 polygon eval: split=$SPLIT ==="

uv run python eval_polygons.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --device cuda \
  --batch-size 8 \
  --output-dir "$OUT_BASE"

echo "=== Done. Results in $OUT_BASE ==="
