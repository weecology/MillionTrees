#!/bin/bash
#SBATCH --job-name=mt_train_points_672
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
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
SPLITS=(random zeroshot)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

# TreeFormer point model lives on the DeepForest treeformer-training branch;
# --group treeformer pulls it in (see pyproject.toml).
# 672-res retrain on B200s: image-size 672, batch 32/GPU (global 96).
# lr held at the proven-stable 2e-4: the 8e-4 linear-scaled value (run 34737954)
# NaN'd the loss, tripping a DDP collective-desync hang in DeepForest's
# training_step non-finite guard. Stability > the high-LR speedup.
srun uv run --group treeformer python training/points/train.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --image-size 672 \
  --batch-size 32 \
  --max-epochs 20 \
  --lr 2e-4 \
  --num-workers 12 \
  --gpus 3 \
  --output-dir "training/points/outputs/${SPLIT}_672" \
  --comet-name "points-${SPLIT}-pretrained-lr2e-4-672" \
  --comet
