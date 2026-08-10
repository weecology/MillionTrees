#!/bin/bash
# Render TreeFormer raw density maps (before peak detection) for the pretrained
# model on the random test split. Array varies score_integration_radius so the
# detected peaks differ over the IDENTICAL density map -- this is the visual
# study of the min_distance lever. See docs/treeformer_hyperparameters.md.
#SBATCH --job-name=mt_tf_density_viz
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees/existing_models/treeformer
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
CHECKPOINT=${CHECKPOINT:-weecology/deepforest-tree-point}
SPLIT=${SPLIT:-random}
IMAGE_SIZE=${IMAGE_SIZE:-448}
# Shrink batch as the image (and density map) grows to stay within GPU memory.
BATCH=${BATCH:-8}
if [ "$IMAGE_SIZE" -ge 1024 ]; then BATCH=2; fi

RADII=(1 2 5)
R=${RADII[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="outputs/density_viz_${SPLIT}_sz${IMAGE_SIZE}_r${R}"

echo "=== TreeFormer density viz: checkpoint=$CHECKPOINT split=$SPLIT image_size=$IMAGE_SIZE radius=$R batch=$BATCH ==="

uv run python visualize_density.py \
  --checkpoint "$CHECKPOINT" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --image-size "$IMAGE_SIZE" \
  --score-thresh 0.1 \
  --score-integration-radius "$R" \
  --n-per-source 5 \
  --batch-size "$BATCH" \
  --num-workers 8 \
  --output-dir "$OUT_DIR"

echo "=== Done. Figures in $OUT_DIR ==="
