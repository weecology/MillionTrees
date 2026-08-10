#!/bin/bash
# Score-threshold sweep: CanopyRS DINO (boxes) + DINO/SAM3 (polygons), random split,
# 3 imgs/source. --score-threshold 0.01 lowers the wrapper keep filter; note the
# detrex detector may impose its own internal floor that limits the low end.
#SBATCH --job-name=mt_sweep_canopyrs
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLIT=random
OUT_BASE=$REPO/existing_models/canopyrs/outputs/sweep

module load cuda/12.8.1 gdal/3.7.0
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH="$CUDA_HOME/bin:$PATH"

cd "$REPO/existing_models/canopyrs"
source .venv/bin/activate

HF_TOKEN_FILE=$REPO/.hf_token
if [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
fi

echo "=== CanopyRS threshold sweep: split=$SPLIT ==="
python eval_boxes.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --device cuda \
  --batch-size 4 \
  --score-threshold 0.01 \
  --per-source 3 \
  --sweep \
  --output-dir "$OUT_BASE"

python eval_polygons.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --device cuda \
  --batch-size 2 \
  --score-threshold 0.01 \
  --per-source 3 \
  --sweep \
  --output-dir "$OUT_BASE"

echo "=== CanopyRS sweep done -> $OUT_BASE/threshold_sweep.csv ==="
