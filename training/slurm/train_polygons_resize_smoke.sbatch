#!/bin/bash
#SBATCH --job-name=mt_poly_resize_smoke
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

# Smoke test for --train-aug resize (whole-image Resize+Pad to 640, no crop) as
# the first leg of a quick --train-aug exploration (crop / resize / nativecrop /
# annotationsafecrop). NOT a leaderboard number -- just confirms the resize +
# --eval-inference resize pairing runs end to end and produces plausible losses
# / F1 on the OOD split before committing to a full 100-epoch run. coco init only,
# single arm. --limit-train/val-batches caps wall time on the real (full) dataset
# -- --mini silently no-ops for supervised-only runs, see AGENTS.md.
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

SPLIT=out-of-distribution
OUT_DIR="training/polygons/outputs/resize_smoke/$SPLIT"
mkdir -p "$OUT_DIR"

srun uv run python training/polygons/train.py \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme "$SPLIT" \
  --gpus 1 \
  --batch-size 16 \
  --lr 0.01 \
  --max-epochs 3 \
  --num-workers 8 \
  --image-size 640 \
  --train-aug resize \
  --eval-inference resize \
  --init-mode coco \
  --data-scope subset \
  --limit-train-batches 0.05 \
  --limit-val-batches 0.2 \
  --output-dir "$OUT_DIR" \
  --comet \
  --comet-name "polygons-${SPLIT}-coco-lr0.01-resize-smoke"
