#!/bin/bash
# Score-threshold sweep for the FINE-TUNED DeepForest box checkpoints.
# Runs inference once with the model emission threshold driven low (0.01) and
# re-scores P/R/F1 across --sweep-thresholds, to find where F1 peaks vs the
# current 0.1 operating point. Eval-only (no retraining). Full test set.
#SBATCH --job-name=mt_sweep_boxes_ft
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=90GB
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:$PYTHONPATH

cd /blue/ewhite/b.weinstein/src/MillionTrees

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

CKPT=$(ls -t "training/boxes/outputs/${SPLIT}/checkpoints"/boxes-best*.ckpt | head -1)
echo "Checkpoint: $CKPT"

srun uv run python training/boxes/eval.py \
  --checkpoint "$CKPT" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --batch-size 8 \
  --num-workers 8 \
  --score-threshold 0.01 \
  --sweep \
  --sweep-thresholds 0.025,0.05,0.075,0.1 \
  --viz-dir "" \
  --output-dir "training/boxes/outputs/${SPLIT}/sweep"
