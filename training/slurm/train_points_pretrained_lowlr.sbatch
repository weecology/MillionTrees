#!/bin/bash
# Pretrained TreeFormer fine-tune at low LR (1e-5) — random split only.
# Hypothesis: lr=2e-4 was destroying pretrained weights in epoch 1.
#SBATCH --job-name=mt_train_pts_pretrained
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=2

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
mkdir -p training/points/outputs/random_pretrained_lowlr

ROOT_DIR=/orange/ewhite/web/public/MillionTrees

srun uv run --extra treeformer python training/points/train.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme random \
  --batch-size 8 \
  --max-epochs 20 \
  --lr 1e-5 \
  --num-workers 4 \
  --output-dir "training/points/outputs/random_pretrained_lowlr" \
  --checkpoint "weecology/deepforest-tree-point" \
  --comet

echo "=== Done: pretrained + lr=1e-5 ==="
