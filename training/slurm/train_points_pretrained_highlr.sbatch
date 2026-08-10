#!/bin/bash
# Pretrained TreeFormer at high LR (1e-3) with no early stopping — random split only.
# Hypothesis: maybe the pretrained weights need a bigger push to adapt; let it run all epochs.
#SBATCH --job-name=mt_train_pts_highlr
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
mkdir -p training/points/outputs/random_pretrained_highlr

ROOT_DIR=/orange/ewhite/web/public/MillionTrees

srun uv run --extra treeformer python training/points/train.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme random \
  --batch-size 8 \
  --max-epochs 20 \
  --lr 1e-3 \
  --early-stop-patience 20 \
  --num-workers 4 \
  --output-dir "training/points/outputs/random_pretrained_highlr" \
  --checkpoint "weecology/deepforest-tree-point" \
  --comet

echo "=== Done: pretrained + lr=1e-3 + no early stop ==="
