#!/bin/bash
#SBATCH --job-name=mt_train_boxes
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=90GB
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Dataset is loaded from ROOT_DIR/TreeBoxes_v{version} (version from TreeBoxes._versions_dict, e.g. 0.11).
# To use the latest packaged data, re-run data_prep/package_datasets.py first.
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO

cd /blue/ewhite/b.weinstein/src/MillionTrees
mkdir -p training/boxes/outputs

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLITS=(random zeroshot)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

# Zero-shot needs more time to find a good optimum.
if [ "$SPLIT" = "zeroshot" ]; then
  EARLY_STOP_PATIENCE=15
else
  EARLY_STOP_PATIENCE=10
fi

uv run python training/boxes/train_boxes.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --batch-size 16 \
  --max-epochs 40 \
  --lr 1e-4 \
  --num-workers 8 \
  --gpus 1 \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --output-dir "training/boxes/outputs/$SPLIT" \
  --comet
