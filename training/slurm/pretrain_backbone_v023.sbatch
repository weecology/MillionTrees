#!/bin/bash
#SBATCH --job-name=mt_pretrain_backbone_v023
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=90GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# Stage 1 of the same-geometry (box -> box) unsupervised pretraining ablation, v0.23.
# Identical recipe to pretrain_backbone_v020.sbatch; only the dataset version moves
# (no --version flag: the loader defaults to the newest _versions_dict key, 0.23).
# Writes to a fresh pretrain_v023/ dir so the v0.19 and v0.20 backbones consumed by
# earlier Table 6 arms stay reproducible.
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="training/weak_supervision/outputs/pretrain_v023/$SPLIT"
mkdir -p "$OUT_DIR"

uv run python training/boxes/pretrain_backbone_for_polygons.py \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme "$SPLIT" \
  --include-unsupervised \
  --batch-size 8 \
  --max-epochs 20 \
  --lr 1e-4 \
  --num-workers 4 \
  --data-scope subset \
  --output-dir "$OUT_DIR" \
  --comet-name "boxes-${SPLIT}-pretrain-v023" \
  --comet
