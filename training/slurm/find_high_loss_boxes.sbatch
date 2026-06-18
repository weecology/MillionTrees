#!/bin/bash
#SBATCH --job-name=mt_high_loss
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

CKPT=training/boxes/outputs/within-distribution/checkpoints/boxes-epoch=10-val_loss=0.5930.ckpt

uv run python training/boxes/find_high_loss_batches.py \
  --checkpoint "$CKPT" \
  --split-scheme within-distribution \
  --batch-size 1 \
  --loss-threshold 5.0 \
  --top-k 20 \
  --num-workers 4 \
  --output-dir training/boxes/outputs/high_loss_images
