#!/bin/bash
#SBATCH --job-name=mt_eval_canopyrs_smoke
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

# End-to-end smoke test: a couple of batches through both eval scripts to confirm
# HF weight download, detectron2 CUDA ops on L4, and SAM3 gated access all work
# before launching the full eval. Not a real benchmark (only --max-batches).
set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=/orange/ewhite/web/public/MillionTrees
OUT_BASE=$REPO/existing_models/canopyrs/outputs/smoke

module load cuda/12.8.1 gdal/3.7.0
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH="$CUDA_HOME/bin:$PATH"

cd "$REPO/existing_models/canopyrs"
source .venv/bin/activate

HF_TOKEN_FILE=/blue/ewhite/b.weinstein/src/MillionTrees/.hf_token
if [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
fi

nvidia-smi -L

echo "=== SMOKE: eval_boxes (2 batches) ==="
python eval_boxes.py --root-dir "$ROOT_DIR" --split-scheme random \
  --device cuda --batch-size 2 --max-batches 2 --output-dir "$OUT_BASE"

echo "=== SMOKE: eval_polygons (2 batches) ==="
python eval_polygons.py --root-dir "$ROOT_DIR" --split-scheme random \
  --device cuda --batch-size 2 --max-batches 2 --output-dir "$OUT_BASE"

echo "=== SMOKE COMPLETE ==="
