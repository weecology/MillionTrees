#!/bin/bash
# 672-res TreeFormer point retrain, random split, EXTENDED to 40 epochs WITH an
# LR scheduler. Companion to train_points_672_40ep_nosched.sbatch (identical but
# no scheduler) — the pair tests whether the 20-epoch run (Comet e7c3) was cut
# short before val_loss converged, and whether ReduceLROnPlateau on val_loss helps.
# Early stopping disabled so the full 40-epoch loss curve is visible.
#SBATCH --job-name=mt_train_points_672_40ep_sched
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=480GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
mkdir -p training/points/outputs

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLIT=random

# Scheduler target is val_loss (ReduceLROnPlateau factor 0.5, patience 5); ckpt
# selection + the (disabled) early-stop still key off point_recall. lr 2e-4 base.
srun uv run --group treeformer python training/points/train.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --image-size 672 \
  --batch-size 32 \
  --max-epochs 40 \
  --lr 2e-4 \
  --lr-scheduler \
  --no-early-stop \
  --num-workers 12 \
  --gpus 3 \
  --output-dir "training/points/outputs/${SPLIT}_672_40ep_sched" \
  --comet-name "points-${SPLIT}-pretrained-lr2e-4-672-40ep-sched" \
  --comet
