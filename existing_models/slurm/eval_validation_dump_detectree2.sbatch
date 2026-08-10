#!/bin/bash
#SBATCH --job-name=mt_valdump_detectree2
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

# Detectree2 zero-shot on the validation split with --save-predictions, for the
# AP50/AP40 x complete-tile re-scoring. Mirrors eval_validation_existing_detectree2.sbatch.
set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
OUT_BASE=$REPO/existing_models/detectree2/outputs/validation
DUMPS=$REPO/outputs/validation_preds

cd $REPO/existing_models/detectree2

MODEL=250312_flexi.pth
if [ ! -f "$MODEL" ]; then
  wget -q https://zenodo.org/records/15863800/files/$MODEL
fi

echo "=== Detectree2 validation eval + prediction dump (score 0.1) ==="

uv run --no-sync python eval_polygons.py \
  --root-dir "$ROOT_DIR" --split-scheme out-of-distribution --eval-split validation \
  --device cuda --batch-size 8 --model-path "$MODEL" --score-threshold 0.1 \
  --output-dir "$OUT_BASE" \
  --save-predictions "$DUMPS/detectree2_polygons_validation.pkl"

echo "=== done. Dumps in $DUMPS ==="
