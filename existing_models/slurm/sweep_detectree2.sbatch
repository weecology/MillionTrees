#!/bin/bash
# Score-threshold sweep: detectree2 on polygons, random split, 3 imgs/source.
# --score-threshold sets the Detectron2 SCORE_THRESH_TEST, so 0.01 emits the full range.
#SBATCH --job-name=mt_sweep_detectree2
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLIT=random
OUT_BASE=$REPO/existing_models/detectree2/outputs/sweep

cd "$REPO/existing_models/detectree2"

MODEL=250312_flexi.pth
if [ ! -f "$MODEL" ]; then
  wget -q https://zenodo.org/records/15863800/files/$MODEL
fi

echo "=== Detectree2 threshold sweep: split=$SPLIT ==="
uv run --no-sync python eval_polygons.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --device cuda \
  --model-path "$MODEL" \
  --score-threshold 0.01 \
  --per-source 3 \
  --sweep \
  --output-dir "$OUT_BASE"

echo "=== Detectree2 sweep done -> $OUT_BASE/threshold_sweep.csv ==="
