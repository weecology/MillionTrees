#!/bin/bash
#SBATCH --job-name=mt_eval_val_boxes
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=90GB
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Manuscript final score: out-of-distribution box model evaluated on the held-out
# Allen et al. 2025 TLS validation set (the 'validation' split, identical across
# all split schemes). Run once when the final OOD checkpoint is ready.

export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:$PYTHONPATH

cd /blue/ewhite/b.weinstein/src/MillionTrees

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLIT=out-of-distribution

# Pinned to the OOD leaderboard checkpoint (boxes-best-v2: newest, and its saved
# results reproduce the leaderboard OOD row recall 0.616 / precision 0.727).
# Override at run time with: CKPT=/path/to.ckpt sbatch eval_validation_boxes.sbatch
CKPT="${CKPT:-training/boxes/outputs/${SPLIT}/checkpoints/boxes-best-v2.ckpt}"
if [ ! -f "$CKPT" ]; then echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; fi
echo "=== Eval boxes on validation (Allen+Frey): split=$SPLIT ==="
echo "Checkpoint: $CKPT"

srun uv run python training/boxes/eval.py \
  --checkpoint "$CKPT" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --eval-split validation \
  --batch-size 16 \
  --num-workers 8 \
  --score-threshold 0.1 \
  --output-dir "training/boxes/outputs/${SPLIT}" \
  --viz-dir "outputs/viz/validation_boxes"
