#!/bin/bash
# 672-res TreeFormer point retrain, random split, EXTENDED to 40 epochs WITHOUT
# an LR scheduler (constant lr 2e-4). Companion to train_points_672_40ep_sched.sbatch
# (identical but adds --lr-scheduler). The pair tests whether the 20-epoch run
# (Comet e7c3) stopped before val_loss converged, isolating the effect of LR
# scheduling. Early stopping disabled so the full 40-epoch loss curve is visible.
#SBATCH --job-name=mt_train_points_672_40ep_nosched
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

# No scheduler: constant lr 2e-4 across all 40 epochs. ckpt selection + the
# (disabled) early-stop key off point_recall, as in the baseline.
srun uv run --group treeformer python training/points/train.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --image-size 672 \
  --batch-size 32 \
  --max-epochs 40 \
  --lr 2e-4 \
  --no-early-stop \
  --num-workers 12 \
  --gpus 3 \
  --output-dir "training/points/outputs/${SPLIT}_672_40ep_nosched" \
  --comet-name "points-${SPLIT}-pretrained-lr2e-4-672-40ep-nosched" \
  --comet
