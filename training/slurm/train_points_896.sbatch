#!/bin/bash
# Matched-resolution retrain of the TreeFormer point model at image_size=896 for
# both splits. The 448 leaderboard model under-counts because its density grid
# (112px) under-resolves dense canopy; 896 gives a 224px grid. Mirrors the
# standard leaderboard recipe (pretrained init, lr 2e-4, 20 epochs) so the only
# variable vs the 448 leaderboard rows is resolution. Runs on b200 for the
# memory headroom 896 needs (the proven 672/batch-32 recipe lived on b200; 896
# is ~1.8x the pixels, so batch 16). Output to *_896 dirs — a separate
# experiment, NOT the leaderboard checkpoint dir. Eval these with
# eval_points.sbatch pointed at the _896 ckpt dir once they finish.
#SBATCH --job-name=mt_train_points_896
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
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
SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

# TreeFormer point model lives on the DeepForest treeformer-training branch;
# --group treeformer pulls it in (see pyproject.toml).
srun uv run --group treeformer python training/points/train.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --image-size 896 \
  --batch-size 16 \
  --max-epochs 20 \
  --lr 2e-4 \
  --num-workers 12 \
  --gpus 3 \
  --output-dir "training/points/outputs/${SPLIT}_896" \
  --comet-name "points-${SPLIT}-pretrained-lr2e-4-896" \
  --comet
