#!/bin/bash
#SBATCH --job-name=mt_eval_val_boxes_repro
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=90GB
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Validation-split (Allen et al. 2025 + Frey et al. 2026 TLS) score for the seed-1
# reproduction OOD box checkpoint from train_boxes_repro.sbatch. train.py only writes
# the TEST result, so the Table 3 Val cells need this second pass.
# Writes into the repro tree -- never the canonical training/boxes/outputs/<split>/.

export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:$PYTHONPATH

cd /blue/ewhite/b.weinstein/src/MillionTrees

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLIT=out-of-distribution
OUT_DIR=training/boxes/outputs/repro_seed1/$SPLIT

CKPT="${CKPT:-$OUT_DIR/checkpoints/boxes-best.ckpt}"
if [ ! -f "$CKPT" ]; then echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; fi
echo "=== Eval repro-seed1 boxes on validation (Allen+Frey): split=$SPLIT ==="
echo "Checkpoint: $CKPT"

srun uv run python training/boxes/eval.py \
  --checkpoint "$CKPT" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --eval-split validation \
  --batch-size 16 \
  --num-workers 8 \
  --score-threshold 0.1 \
  --output-dir "$OUT_DIR" \
  --viz-dir "$OUT_DIR/viz/validation_boxes"
