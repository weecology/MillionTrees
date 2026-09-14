#!/bin/bash
#SBATCH --job-name=mt_pretrain_d2
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

# STAGE 1 of the native Detectron2 polygon weak-supervision ablation.
#
# Trains a D2 Mask R-CNN with the mask branch OFF on the unsupervised Weinstein et al.
# 2018 boxes (41,691 tiles / 6.4M boxes within-distribution; 46,823 / 7.1M OOD), then
# exports the 295 tensors stage 2 can reuse: backbone + FPN + RPN + ROI box head. The
# mask head is never trained here, so it reaches stage 2 still at COCO init.
#
# Both defects the DeepForest stage-1 LR sweep (39879762) exposed are fixed by
# construction, because they are what made the earlier stage-1 exports uninterpretable:
#   * grad clipping at norm 1.0 -- without it lr 1e-2 went NaN from epoch 0
#   * checkpoint selected on holdout box_recall, not loss -- the val_loss-best epoch
#     exported a near-dead detector (recall 0.003) while a worse-loss epoch was 10x better
#
# The split MUST be split-specific: reusing the within-distribution network on the OOD
# arm would leak imagery that is OOD test data.
set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="$REPO/training/weak_supervision/outputs/pretrain_detectron2/$SPLIT"
mkdir -p "$OUT_DIR"

# Detectron2 (CanopyRS venv) CUDA ops are a fat binary for L4 (sm_89) + B200 (sm_100).
module load cuda/12.8.1 gdal/3.7.0
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Block-buffered stdout makes a running job unreadable; the ledger workflow reads .out
# while the job is still going.
export PYTHONUNBUFFERED=1

cd "$REPO/existing_models/canopyrs"
source .venv/bin/activate

echo "=== Detectron2 stage 1: weak-box pretraining, split=$SPLIT ==="
python "$REPO/training/polygons/pretrain_detectron2.py" \
  --root-dir "$ROOT_DIR" \
  --boxes-dataset TreeBoxes_v0.23 \
  --split-scheme "$SPLIT" \
  --weak-sources '*unsupervised*' \
  --holdout-images 512 \
  --batch-size 8 \
  --max-epochs 10 \
  --lr 1e-3 \
  --grad-clip 1.0 \
  --eval-period 4000 \
  --num-workers 16 \
  --output-dir "$OUT_DIR" \
  --comet \
  --comet-name "stage1-${SPLIT}-coco-lr0.001-detectron2"

echo "=== Detectron2 stage 1 done: $OUT_DIR ==="
