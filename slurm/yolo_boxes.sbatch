#!/bin/bash
#SBATCH --job-name=mt_yolo_boxes
#SBATCH --output=logs/slurm/yolo_boxes_%x_%A_%a.out
#SBATCH --error=logs/slurm/yolo_boxes_%x_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
mkdir -p logs/slurm

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLITS=(random zeroshot crossgeometry)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

# Ensure ultralytics is available (lightweight check)
if ! uv run python -c 'import importlib.util,sys;sys.exit(0) if importlib.util.find_spec("ultralytics") else sys.exit(1)'; then
  uv pip install -q ultralytics
fi

uv run python docs/examples/yolo_treeboxes.py \
  --root_dir "$ROOT_DIR" \
  --version 0.8 \
  --split_scheme "$SPLIT" \
  --batch_size 8 \
  --device auto \
  --weights yolov8n.pt


