#!/bin/bash
#SBATCH --job-name=mt_poly_aug_compare_ood
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

# Full, uniform comparison of the four TreePolygons --train-aug strategies on the
# out-of-distribution split, COCO init only (pure augmentation isolation -- the
# backbone axis is covered by the Table 6 coco/box_pretrained pairs). Supersedes
# the piecemeal runs: nativecrop pair 40778999 (cancelled), annotationsafecrop
# 40777946 (cancelled), resize smoke 40780078 (timed out before eval).
#
# Every arm is identical except the train crop op and its scale-matched eval:
#
#   idx  --train-aug          --eval-inference  --image-size  notes
#   0    crop                 tiled             448           RandomResizedCrop 640
#                                                             (area 0.64-1.0; a
#                                                             downsample-resize in
#                                                             disguise on big tiles)
#   1    resize               resize            640           whole-image Resize+Pad,
#                                                             no crop; eval at same
#                                                             scale (no tiling)
#   2    nativecrop           tiled             448           true 640px RandomCrop
#                                                             at native GSD
#   3    annotationsafecrop   tiled             448           RandomSizedBBoxSafeCrop,
#                                                             each crop has >=1 anno
#
# Shared recipe: OOD split, batch 16, lr 0.01, 100 epochs, --data-scope subset,
# --keep-all-checkpoints. select_checkpoint.py then re-scores every kept
# checkpoint on the held-out Allen et al. 2025 validation split (val-F1
# selection) and writes the canonical test results_<split>.txt -- the polygon
# Mask R-CNN's val_loss bottoms out ~epoch 9 while detection quality keeps
# climbing, so val_loss checkpointing would bias arms unequally
# (notes/polygon_maskrcnn_overfits, CLAUDE.md). Both select_checkpoint.py and
# eval.py now take --eval-inference so the resize arm is selected & scored at its
# own scale.

export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

SPLIT=out-of-distribution

AUGS=(crop     resize  nativecrop  annotationsafecrop)
EVALS=(tiled   resize  tiled       tiled)
SIZES=(448     640     448         448)

AUG=${AUGS[$SLURM_ARRAY_TASK_ID]}
EVAL_INF=${EVALS[$SLURM_ARRAY_TASK_ID]}
IMG_SIZE=${SIZES[$SLURM_ARRAY_TASK_ID]}

OUT_DIR="training/polygons/outputs/aug_compare_v024/${AUG}/${SPLIT}"
mkdir -p "$OUT_DIR"

srun uv run python training/polygons/train.py \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme "$SPLIT" \
  --gpus 1 \
  --batch-size 16 \
  --lr 0.01 \
  --max-epochs 100 \
  --num-workers 16 \
  --image-size "$IMG_SIZE" \
  --train-aug "$AUG" \
  --eval-inference "$EVAL_INF" \
  --init-mode coco \
  --data-scope subset \
  --keep-all-checkpoints \
  --output-dir "$OUT_DIR" \
  --comet \
  --comet-name "polygons-${SPLIT}-coco-lr0.01-${AUG}-augcompare"

# Validation-based checkpoint selection + canonical test eval, scored at the same
# inference scale the arm trained at.
srun uv run python training/polygons/select_checkpoint.py \
  --checkpoints-dir "$OUT_DIR/logs/checkpoints" \
  --split-scheme "$SPLIT" \
  --init-mode coco \
  --image-size "$IMG_SIZE" \
  --eval-inference "$EVAL_INF" \
  --select-metric f1 \
  --output-dir "$OUT_DIR"
