#!/bin/bash
#SBATCH --job-name=mt_train_boxes_repro
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --partition=hpg-turin
#SBATCH --gpus=l4:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2

# REPRODUCIBILITY RUN for manuscript Table 3 (DeepForest RetinaNet, fine-tuned).
#
# Byte-identical recipe to training/slurm/train_boxes.sbatch -- same batch size, LR,
# epochs, patience, --augment -- with exactly TWO differences:
#   1. --seed 1 (production runs at the default --seed 0), so the spread between this
#      run and the production run measures training stochasticity alone.
#   2. --output-dir under repro_seed1/, so the manuscript-backing results in
#      training/boxes/outputs/<split>/ are never clobbered.
#
# Why this row: manuscript Table 3's fine-tuned DeepForest cells were transcribed from
# the aug_ab/*_aug-schedf1 A/B arm, not from this production recipe, and the two
# disagree most on the OOD validation cells (AP40 0.280 vs 0.169). This run establishes
# how much of that gap is seed noise.

export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /blue/ewhite/b.weinstein/src/MillionTrees
mkdir -p training/boxes/outputs/repro_seed1

source scripts/slurm_mem_probe.sh

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

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
  --gpus 2 \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --augment \
  --seed 1 \
  --output-dir "training/boxes/outputs/repro_seed1/$SPLIT" \
  --comet \
  --comet-name "boxes-${SPLIT}-pretrained-lr0.01-repro-seed1"
