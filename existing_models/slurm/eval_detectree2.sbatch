#!/bin/bash
#SBATCH --job-name=mt_eval_detectree2
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLITS=(random zeroshot)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_BASE=$REPO/existing_models/detectree2/outputs/$SPLIT

cd $REPO/existing_models/detectree2

# Download the model_garden checkpoint once (cached afterwards).
MODEL=250312_flexi.pth
if [ ! -f "$MODEL" ]; then
  wget -q https://zenodo.org/records/15863800/files/$MODEL
fi

echo "=== Detectree2 eval: split=$SPLIT ==="

# --no-sync: keep the hand-built detectron2 + pinned torch (not in pyproject.toml).
uv run --no-sync python eval_polygons.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --device cuda \
  --batch-size 8 \
  --model-path "$MODEL" \
  --score-threshold 0.1 \
  --output-dir "$OUT_BASE"

echo "=== Detectree2 eval done. Results in $OUT_BASE ==="
