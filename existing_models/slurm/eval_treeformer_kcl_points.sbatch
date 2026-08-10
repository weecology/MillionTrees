#!/bin/bash
# KCL checkpoint: local fine-tuned TreeFormer HF checkpoint
# TreePoints keypoint eval, array over random + zeroshot test splits
#SBATCH --job-name=mt_tf_kcl_points
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
CHECKPOINT=/home/veitchmichaelisj/code/DeepForest_jvm/kcl_hf_checkpoint
SPLITS=(random zeroshot)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="training/points/outputs/${SPLIT}_kcl"

echo "=== TreeFormer point eval (KCL checkpoint) split=$SPLIT ==="
echo "Root dir: $ROOT_DIR"
echo "Checkpoint: $CHECKPOINT"
echo "Output dir: $OUT_DIR"

uv run python training/points/eval.py \
  --checkpoint "$CHECKPOINT" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --batch-size 16 \
  --num-workers 8 \
  --output-dir "$OUT_DIR" \
  --viz-dir "$OUT_DIR/viz"

echo "=== Done. Results in $OUT_DIR ==="
