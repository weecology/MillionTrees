#!/bin/bash
# Score-threshold sweep: SAM3 on points/boxes/polygons, random split, 3 imgs/source.
# Emits the full score range (--score-threshold 0.01) so the eval-side sweep is honest.
#SBATCH --job-name=mt_sweep_sam3
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLIT=random
OUT_BASE=$REPO/existing_models/sam3/outputs/sweep

cd "$REPO/existing_models/sam3"

HF_TOKEN_FILE=$REPO/.hf_token
if [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
fi

echo "=== SAM3 threshold sweep: split=$SPLIT ==="
for geom in eval_boxes eval_points eval_polygons; do
  uv run python $geom.py \
    --root-dir "$ROOT_DIR" \
    --split-scheme "$SPLIT" \
    --device cuda \
    --score-threshold 0.01 \
    --per-source 3 \
    --sweep \
    --output-dir "$OUT_BASE" || echo "[sweep] $geom FAILED (continuing)"
done

echo "=== SAM3 sweep done -> $OUT_BASE/threshold_sweep.csv ==="
