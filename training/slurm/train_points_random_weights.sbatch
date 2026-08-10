#!/bin/bash
# TreeFormer from random weights (ImageNet backbone only) at lr=2e-4 — random split only.
# Control experiment: can the model learn at all from scratch on this data?
#SBATCH --job-name=mt_train_pts_random
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
mkdir -p training/points/outputs/random_random_weights

ROOT_DIR=/orange/ewhite/web/public/MillionTrees

srun uv run --extra treeformer python training/points/train.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme random \
  --batch-size 8 \
  --max-epochs 20 \
  --lr 2e-4 \
  --num-workers 4 \
  --output-dir "training/points/outputs/random_random_weights" \
  --random-weights \
  --comet

echo "=== Done: random weights + lr=2e-4 ==="
