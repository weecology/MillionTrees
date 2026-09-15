#!/bin/bash
#SBATCH --job-name=mt_poly_table6_ood_ckptselect
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

# Table 6 (weak supervision), out-of-distribution split, WITH validation-based
# checkpoint selection.
#
# Motivation: the v0.24 annotationsafecrop pair (jobs 40371583 / 40371590) both had
# ModelCheckpoint(monitor="val_loss") hand the final eval an epoch-9 checkpoint. The
# polygon Mask R-CNN's val_loss bottoms out by ~epoch 9 then rises while detection
# quality keeps climbing, so the COCO control OOD row collapsed (F1 0.21 / recall 0.13)
# while the box-pretrained arm held up (F1 0.46) -- an artifact of *where each arm's
# val_loss happened to minimize*, not a real generalization gap.
#
# This run keeps a checkpoint at every validation epoch (--keep-all-checkpoints),
# then select_checkpoint.py re-scores all of them on the held-out Allen et al. 2025
# validation split (never in train/test, identical rows in every split-scheme), picks
# the best validation F1, and writes the canonical test results_<split>.txt from that
# checkpoint. Both arms get the same treatment so the comparison stays apples-to-apples.
#
# arm 0: COCO control        arm 1: box-pretrained backbone
# Recipe identical to train_polygons_deepforest_annotationsafecrop.sbatch.

export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

SPLIT=out-of-distribution
INITS=(coco box_pretrained)
INIT=${INITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="training/weak_supervision/outputs/table6_ckptselect_v024/${INIT}/${SPLIT}"
BACKBONE="training/weak_supervision/outputs/pretrain/${SPLIT}/box_backbone_${SPLIT}.pt"
mkdir -p "$OUT_DIR"

TRAIN_ARGS=(
  --root-dir /orange/ewhite/web/public/MillionTrees
  --split-scheme "$SPLIT"
  --gpus 1
  --batch-size 16
  --lr 0.01
  --max-epochs 100
  --num-workers 8
  --image-size 448
  --train-aug annotationsafecrop
  --eval-inference tiled
  --init-mode "$INIT"
  --data-scope subset
  --keep-all-checkpoints
  --output-dir "$OUT_DIR"
  --comet
  --comet-name "polygons-${SPLIT}-${INIT}-lr0.01-annotationsafecrop-ckptselect"
)
if [ "$INIT" = "box_pretrained" ]; then
  TRAIN_ARGS+=(--box-backbone-checkpoint "$BACKBONE")
fi

srun uv run python training/polygons/train.py "${TRAIN_ARGS[@]}"

# Validation-based checkpoint selection + canonical test eval.
srun uv run python training/polygons/select_checkpoint.py \
  --checkpoints-dir "$OUT_DIR/logs/checkpoints" \
  --split-scheme "$SPLIT" \
  --init-mode "$INIT" \
  --image-size 448 \
  --select-metric f1 \
  --output-dir "$OUT_DIR"
