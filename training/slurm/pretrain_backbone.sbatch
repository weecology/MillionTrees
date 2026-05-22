#!/bin/bash
#SBATCH --job-name=mt_pretrain_backbone
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=90GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

SPLITS=(random zeroshot)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="training/weak_supervision/outputs/pretrain/$SPLIT"
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
  --comet
