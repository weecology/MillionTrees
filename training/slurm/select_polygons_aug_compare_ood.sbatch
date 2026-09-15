#!/bin/bash
#SBATCH --job-name=mt_poly_aug_compare_ood_select
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

# Selection-only recovery for job 40820668. All four --train-aug arms finished
# training (100 epochs, --keep-all-checkpoints => epochs 9..99 on disk) but the
# in-line select_checkpoint.py step crashed on every arm: eval.py does
# `from training.polygons.train import ...` and the aug_compare sbatch never
# exported PYTHONPATH, so `python training/polygons/eval.py` ran in script mode
# with the repo root off sys.path -> ModuleNotFoundError: No module named 'training'.
#
# eval.py now inserts the repo root onto sys.path itself; this job also exports
# PYTHONPATH belt-and-suspenders. No retrain -- just re-score the kept checkpoints
# on the held-out Allen et al. 2025 validation split (val-F1 selection) and write
# the canonical test results_<split>.txt per arm, at each arm's own eval scale.
#
#   idx  arm                  --eval-inference  --image-size
#   0    crop                 tiled             448
#   1    resize               resize            640
#   2    nativecrop           tiled             448
#   3    annotationsafecrop   tiled             448

export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
source scripts/slurm_mem_probe.sh

SPLIT=out-of-distribution

AUGS=(crop     resize  nativecrop  annotationsafecrop)
EVALS=(tiled   resize  tiled       tiled)
SIZES=(448     640     448         448)

AUG=${AUGS[$SLURM_ARRAY_TASK_ID]}
EVAL_INF=${EVALS[$SLURM_ARRAY_TASK_ID]}
IMG_SIZE=${SIZES[$SLURM_ARRAY_TASK_ID]}

OUT_DIR="training/polygons/outputs/aug_compare_v024/${AUG}/${SPLIT}"

srun uv run python training/polygons/select_checkpoint.py \
  --checkpoints-dir "$OUT_DIR/logs/checkpoints" \
  --split-scheme "$SPLIT" \
  --init-mode coco \
  --image-size "$IMG_SIZE" \
  --eval-inference "$EVAL_INF" \
  --select-metric f1 \
  --output-dir "$OUT_DIR"
