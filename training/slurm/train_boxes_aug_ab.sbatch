#!/bin/bash
#SBATCH --job-name=mt_boxes_aug_ab
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --partition=hpg-turin
#SBATCH --gpus=l4:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2

# A/B: does training-time augmentation (and, on top of it, annealing LR on the
# RIGHT signal) change the downstream MillionTrees box scores?
#
# Motivation. The production box recipe (train_boxes.sbatch) trains with a
# resize-only transform at a flat lr=0.01 for up to 200 epochs. val_loss is
# U-shaped (classic overfitting); it is only tolerable because checkpoint
# selection / early stopping monitor box_f1, not val_loss. The prior LR A/B
# (job 40722692) added ReduceLROnPlateau on val_loss and moved AP40 within noise
# / AP60 by ~+0.015 on WD -- a small effect from annealing on an anchor-level
# loss whose late rise is partly background/hard-negative noise the post-NMS
# detection metrics never see. This run attacks the overfitting at the source
# (augmentation) and, in the scheduled arm, points ReduceLROnPlateau at box_f1
# instead of val_loss.
#
# --augment adds HorizontalFlip + VerticalFlip + RandomRotate90 (p=0.5 each) to
# the TRAIN subset only (val/test stay resize-only); all label-preserving for
# nadir imagery, no radiometric assumptions. --lr-plateau-monitor box_f1 flips
# the scheduler to mode=max on the same metric that drives checkpoint selection.
#
# Every other knob is held at the train_boxes.sbatch production recipe
# (deepforest init, batch 32, max-epochs 200, lr 0.01, 2x L4, patience 10 WD /
# 15 OOD). Compare against job 40722692's nosched / sched-on-val_loss arms.
#
# 2x2 array: split x arm
#   idx 0: within-distribution   aug            (constant lr 0.01)
#   idx 1: within-distribution   aug + sched-f1 (ReduceLROnPlateau on box_f1)
#   idx 2: out-of-distribution   aug
#   idx 3: out-of-distribution   aug + sched-f1
#
# Read training/boxes/outputs/aug_ab/<split>_<arm>/results_<split>.txt:
# Recall / Mask-aware Precision / F1 / AP40 / AP60 / counting MAE.
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /blue/ewhite/b.weinstein/src/MillionTrees
mkdir -p training/boxes/outputs/aug_ab

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLITS=(within-distribution within-distribution out-of-distribution out-of-distribution)
SCHED_FLAGS=("" "--lr-scheduler --lr-plateau-monitor box_f1" "" "--lr-scheduler --lr-plateau-monitor box_f1")
ARM_NAMES=(aug aug-schedf1 aug aug-schedf1)

SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
SCHED_FLAG=${SCHED_FLAGS[$SLURM_ARRAY_TASK_ID]}
ARM=${ARM_NAMES[$SLURM_ARRAY_TASK_ID]}

if [ "$SPLIT" = "out-of-distribution" ]; then
  EARLY_STOP_PATIENCE=15
else
  EARLY_STOP_PATIENCE=10
fi

OUT_DIR="training/boxes/outputs/aug_ab/${SPLIT}_${ARM}"
mkdir -p "$OUT_DIR"

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
  $SCHED_FLAG \
  --output-dir "$OUT_DIR" \
  --comet \
  --comet-name "boxes-${SPLIT}-deepforest-lr0.01-${ARM}"
