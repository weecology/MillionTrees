#!/bin/bash
#SBATCH --job-name=mt_pretrain_supval_lrsweep
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=32:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=90GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# Stage-1 pretraining, instrumented and LR-swept.
#
# WHY THIS EXISTS -- the audit of 2026-08-21 found stage 1 was never measured:
#
#   * include_sources=['*unsupervised*'] puts every unsupervised row in the TRAIN
#     split (46,823 train / 0 test), so has_val was False. No ModelCheckpoint, no
#     early stopping, and -- because DeepForest's configure_optimizers only attaches
#     a scheduler when validation.csv_file is set -- no LR schedule either. The
#     stage's own results_<split>.txt were all `nan` (empty eval set), so nobody
#     could tell whether it had learned anything.
#   * Comet shows train_loss plateauing at epoch ~5 then oscillating. bbox_regression
#     moved only 0.649 -> 0.551 in 20 epochs over 7.1M boxes. The OOD run 39614225
#     ENDED at classification 0.651 having reached 0.529 at epoch 7, and exported the
#     worse last-epoch model because nothing was watching.
#   * Zero-shot check of the exported 301-tensor network: recall@IoU0.4 = 0.217 on
#     its OWN unsupervised training distribution (0.203 on supervised test). It is a
#     real detector -- COCO init scores 0.000 -- but nowhere near converged. A
#     detector that recalls 22% of its own training labels has not finished training.
#
# So the "6.4M boxes are worthless" null was measuring an undertrained stage 1, not a
# saturated benchmark. This array asks the direct question: does stage 1 converge if
# it is given a working recipe?
#
#   task 0 = OLD RECIPE, instrumented   (batch 8,  lr 1e-4, constant LR)  <- control
#   task 1 = moderate                   (batch 32, lr 1e-3, cosine)
#   task 2 = matches stage-2 fine-tune  (batch 32, lr 1e-2, cosine)
#
# Task 2 matters for a second reason: stage 2 fine-tunes at lr 0.01 with warmup_epochs=0
# and scheduler None, i.e. 100x the stage-1 LR with no ramp. Even a good stage-1 solution
# would be destroyed in the first epoch of stage 2. Pretraining at the same LR the
# fine-tune uses removes that discontinuity.
#
# All three now validate on 512 held-out SUPERVISED train images (stratified by source,
# seed 42) and export the BEST epoch by supervised val_loss, not the last.
#
# --val-every-n-epochs 1 is load-bearing, not cosmetic: DeepForest's create_trainer sets
# check_val_every_n_epoch from config.validation.val_accuracy_interval, default 20, so a
# 20-epoch run would otherwise validate exactly once at the last epoch and "best" would
# equal "last" -- silently reproducing the very bug this run exists to fix (job 39879499).
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

SPLIT=within-distribution
LRS=(1e-4 1e-3 1e-2)
BATCHES=(8 32 32)
SCHEDS=(none cosine cosine)
TAGS=(oldrecipe lr1e-3-cosine lr1e-2-cosine)

LR=${LRS[$SLURM_ARRAY_TASK_ID]}
BS=${BATCHES[$SLURM_ARRAY_TASK_ID]}
SCHED=${SCHEDS[$SLURM_ARRAY_TASK_ID]}
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}

OUT_DIR="training/weak_supervision/outputs/pretrain_supval_lrsweep/$TAG"
mkdir -p "$OUT_DIR"

uv run python training/boxes/pretrain_backbone_for_polygons.py \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme "$SPLIT" \
  --include-unsupervised \
  --init-mode coco \
  --batch-size "$BS" \
  --max-epochs 20 \
  --lr "$LR" \
  --scheduler "$SCHED" \
  --supervised-val-images 512 \
  --val-every-n-epochs 1 \
  --ckpt-monitor val_loss \
  --ckpt-mode min \
  --num-workers 4 \
  --output-dir "$OUT_DIR" \
  --comet-name "boxes-${SPLIT}-pretrain-supval-${TAG}" \
  --comet
