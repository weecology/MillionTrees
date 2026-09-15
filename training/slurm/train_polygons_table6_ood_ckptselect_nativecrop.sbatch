#!/bin/bash
#SBATCH --job-name=mt_poly_table6_ood_ckptselect_nativecrop
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

# A/B control for train_polygons_table6_ood_ckptselect.sbatch (jobs 40777946_0/_1,
# running as of 2026-09-01). annotationsafecrop was an experimental feature added
# alongside the ckptselect methodology fix; this run isolates it by swapping only
# --train-aug annotationsafecrop -> nativecrop (true 640px RandomCrop at native GSD,
# same --eval-inference tiled). Everything else -- OOD split, coco/box_pretrained
# arms, --keep-all-checkpoints + select_checkpoint.py val-F1 selection -- is
# identical, so any gap between this run and 40777946 isolates the crop-safety
# feature's contribution rather than the ckptselect fix or backbone comparison.
#
# arm 0: COCO control        arm 1: box-pretrained backbone

export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

SPLIT=out-of-distribution
INITS=(coco box_pretrained)
INIT=${INITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="training/weak_supervision/outputs/table6_ckptselect_v024_nativecrop/${INIT}/${SPLIT}"
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
  --train-aug nativecrop
  --eval-inference tiled
  --init-mode "$INIT"
  --data-scope subset
  --keep-all-checkpoints
  --output-dir "$OUT_DIR"
  --comet
  --comet-name "polygons-${SPLIT}-${INIT}-lr0.01-nativecrop-ckptselect"
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
