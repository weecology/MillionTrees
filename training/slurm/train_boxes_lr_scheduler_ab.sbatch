#!/bin/bash
#SBATCH --job-name=mt_boxes_lrsched_ab
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

# A/B: does adding an LR schedule change the downstream MillionTrees box scores?
#
# The production box runs (train_boxes.sbatch) are plain SGD at a CONSTANT
# lr=0.01 for up to 200 epochs -- no warmup, no decay. Train loss drops
# precipitously and val_loss is U-shaped (overfitting). DeepForest's
# configure_optimizers silently drops any scheduler when validation.csv_file is
# None, which MillionTrees training always is, so a schedule has never actually
# run on this task. train.py --lr-scheduler now re-attaches ReduceLROnPlateau on
# val_loss (factor 0.5, patience 5) via a configure_optimizers override.
#
# Every knob except --lr-scheduler is held at the train_boxes.sbatch production
# recipe (default deepforest/pretrained init, batch 32, max-epochs 200, lr 0.01,
# 2x L4, patience 10 WD / 15 OOD). Checkpoint selection + early stopping still
# monitor box_recall, so the scored checkpoint is chosen the same way in both
# arms; the only difference is whether the LR anneals on a val_loss plateau.
#
# 2x2 array: split x {nosched, sched}
#   idx 0: within-distribution   nosched  (control -- reproduces the leaderboard run)
#   idx 1: within-distribution   sched
#   idx 2: out-of-distribution   nosched
#   idx 3: out-of-distribution   sched
#
# Compare training/boxes/outputs/lr_sched_ab/<split>_<arm>/results_<split>.txt:
# Recall / Mask-aware Precision / F1 / AP40 / AP60 / counting MAE. If the sched
# arm matches the nosched arm within noise on all splits, the schedule has no
# downstream effect and the constant-LR production recipe stands.
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /blue/ewhite/b.weinstein/src/MillionTrees
mkdir -p training/boxes/outputs/lr_sched_ab

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLITS=(within-distribution within-distribution out-of-distribution out-of-distribution)
SCHED_FLAGS=("" "--lr-scheduler" "" "--lr-scheduler")
ARM_NAMES=(nosched sched nosched sched)

SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
SCHED_FLAG=${SCHED_FLAGS[$SLURM_ARRAY_TASK_ID]}
ARM=${ARM_NAMES[$SLURM_ARRAY_TASK_ID]}

if [ "$SPLIT" = "out-of-distribution" ]; then
  EARLY_STOP_PATIENCE=15
else
  EARLY_STOP_PATIENCE=10
fi

OUT_DIR="training/boxes/outputs/lr_sched_ab/${SPLIT}_${ARM}"
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
  $SCHED_FLAG \
  --output-dir "$OUT_DIR" \
  --comet \
  --comet-name "boxes-${SPLIT}-deepforest-lr0.01-schedAB-${ARM}"
