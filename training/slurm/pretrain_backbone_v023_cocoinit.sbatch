#!/bin/bash
#SBATCH --job-name=mt_pretrain_backbone_v023_cocoinit
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=90GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# Stage 1 of the CORRECTED same-geometry (box -> box) pretraining ablation, v0.23.
#
# Why this exists rather than reusing pretrain_backbone_v023.sbatch (job 38840963):
# that script starts pretraining from weecology/deepforest-tree, so its exported trunk
# already carries NEON supervision. Combined with the same contamination on the
# downstream control (see train_boxes_coco_v023_cleaninit.sbatch), the whole 38840963/
# 38840964/38840965 ablation compared deepforest-tree against deepforest-tree nudged
# 6.4% in norm — which is why every metric came out a wash.
#
# --init-mode coco starts from torchvision COCO RetinaNet and asserts the trunk really
# is COCO before training. Whatever this backbone learns is attributable to the
# unsupervised boxes alone.
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="training/weak_supervision/outputs/pretrain_v023_cocoinit/$SPLIT"
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
  --comet-name "boxes-${SPLIT}-pretrain-v023-cocoinit" \
  --comet
