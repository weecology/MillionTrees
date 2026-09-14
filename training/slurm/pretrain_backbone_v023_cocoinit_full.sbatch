#!/bin/bash
#SBATCH --job-name=mt_pretrain_backbone_v023_cocoinit_full
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=90GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# Stage 1 of the WHOLE-NETWORK same-geometry (box -> box) pretraining ablation, v0.23.
#
# Identical recipe to pretrain_backbone_v023_cocoinit.sbatch (job 39129719) — same
# 20-epoch warmup at lr 1e-4 on the unsupervised boxes, no held-out split, last epoch
# exported. The ONLY reason this re-runs is that 39129719 saved just the ResNet trunk:
# export_backbone_weights() filtered on "backbone.body.", and because the unsupervised-
# only subset leaves the test split empty there was no ModelCheckpoint either. So the
# trained FPN (16 tensors) and detection heads (20 tensors) from 6.4M boxes were
# discarded at write time and are unrecoverable.
#
# pretrain_backbone_for_polygons.py now ALSO writes box_network_<split>.pt with all 301
# tensors. box_backbone_<split>.pt (265, trunk only) is still written unchanged so the
# polygon box -> Mask R-CNN ablation, where the trunk really is the only compatible
# piece, keeps working off the same run.
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="training/weak_supervision/outputs/pretrain_v023_cocoinit_full/$SPLIT"
mkdir -p "$OUT_DIR"

uv run python training/boxes/pretrain_backbone_for_polygons.py \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme "$SPLIT" \
  --include-unsupervised \
  --init-mode coco \
  --batch-size 8 \
  --max-epochs 20 \
  --lr 1e-4 \
  --num-workers 4 \
  --data-scope subset \
  --output-dir "$OUT_DIR" \
  --comet-name "boxes-${SPLIT}-pretrain-v023-cocoinit-full" \
  --comet
