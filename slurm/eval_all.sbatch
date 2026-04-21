#!/bin/bash
#SBATCH --job-name=mt_eval_all
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=90GB
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -euo pipefail

export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
VIZ_DIR=$REPO/outputs/viz

cd $REPO

echo "=== eval_all start: $(date) ==="
echo "VIZ_DIR: $VIZ_DIR"

srun uv run python eval_all.py \
  --root-dir "$ROOT_DIR" \
  --viz-dir "$VIZ_DIR" \
  --splits random zeroshot \
  --batch-size 8 \
  --num-workers 8

echo "=== eval_all done: $(date) ==="
echo "Visualizations written to $VIZ_DIR"
