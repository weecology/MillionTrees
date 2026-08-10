#!/bin/bash
#SBATCH --job-name=mt_train_boxes_1gpu
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Single-GPU boxes training. Avoids the multi-GPU NCCL allreduce watchdog
# timeout seen in 35306005 (a dataloader worker stalling on a contended
# /orange read made one rank miss the validation allreduce -> 30 min timeout).
# Dataset is loaded from ROOT_DIR/TreeBoxes_v{version} (version from TreeBoxes._versions_dict).

cd /blue/ewhite/b.weinstein/src/MillionTrees
mkdir -p training/boxes/outputs

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

# Out-of-distribution needs more time to find a good optimum.
if [ "$SPLIT" = "out-of-distribution" ]; then
  EARLY_STOP_PATIENCE=15
else
  EARLY_STOP_PATIENCE=10
fi

srun uv run python training/boxes/train.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --batch-size 32 \
  --max-epochs 200 \
  --lr 0.01 \
  --num-workers 8 \
  --gpus 1 \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --output-dir "training/boxes/outputs/$SPLIT" \
  --comet
