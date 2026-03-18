#!/bin/bash
#SBATCH --job-name=mt_train_boxes
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

# Dataset is loaded from ROOT_DIR/TreeBoxes_v{version} (version from TreeBoxes._versions_dict, e.g. 0.11).
# To use the latest packaged data, re-run data_prep/package_datasets.py first.
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
mkdir -p training/boxes/outputs

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLITS=(random zeroshot)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

uv run python training/boxes/train_boxes.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --batch-size 8 \
  --max-epochs 20 \
  --lr 1e-4 \
  --num-workers 4 \
  --output-dir "training/boxes/outputs/$SPLIT" \
  --comet
