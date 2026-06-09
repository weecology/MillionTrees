#!/bin/bash
#SBATCH --job-name=mt_train_points
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --partition=hpg-turin
#SBATCH --gpus=3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3


set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
mkdir -p training/points/outputs

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLITS=(random zeroshot)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

# TreeFormer point model lives on the DeepForest treeformer-training branch;
# --extra treeformer pulls it in (see pyproject.toml).
srun uv run --extra treeformer python training/points/train.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --batch-size 8 \
  --max-epochs 20 \
  --lr 2e-4 \
  --num-workers 4 \
  --gpus 3 \
  --output-dir "training/points/outputs/$SPLIT" \
  --comet
