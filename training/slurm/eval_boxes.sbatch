#!/bin/bash
#SBATCH --job-name=mt_eval_boxes
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
SPLITS=(random zeroshot)
CKPT_VERSIONS=(v4 v5)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
CKPT_VER=${CKPT_VERSIONS[$SLURM_ARRAY_TASK_ID]}

CKPT="training/boxes/outputs/${SPLIT}/checkpoints/boxes-best-${CKPT_VER}.ckpt"

srun uv run python training/boxes/eval.py \
  --checkpoint "$CKPT" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --batch-size 16 \
  --num-workers 8 \
  --output-dir "training/boxes/outputs/${SPLIT}"
