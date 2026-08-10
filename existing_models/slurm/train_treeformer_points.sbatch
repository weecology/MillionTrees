#!/bin/bash
# Fine-tune TreeFormer on TreePoints (random split) with a higher LR and longer
# schedule than the 1e-5/20ep default. Small LR bracket to find the right value;
# each task trains 30 epochs and evaluates the best (point_recall) checkpoint on
# the test split with the standardized post-proc (score_thresh=0.1, radius=2).
#
# NOTE: train.py monitors point_recall on the *test* subset for checkpoint/early
# stop (val_loader = test_subset), so the reported number is test-selected.
#SBATCH --job-name=mt_tf_train_points
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLIT=${SPLIT:-random}

# LR comes from the array by default, but an explicit LR env var overrides it
# for one-off runs (e.g. LR=0.05 sbatch --array=0 ...).
LRS=(5e-5 1e-4 2e-4)
LR=${LR:-${LRS[$SLURM_ARRAY_TASK_ID]}}

# Image size (and a batch that fits it). 1024 is ~5x the pixels of 448, so shrink
# the batch unless an explicit BATCH is given; pair 1024 with a large-memory GPU
# (submit with --partition=hpg-b200).
IMAGE_SIZE=${IMAGE_SIZE:-448}
if [ "$IMAGE_SIZE" -ge 1024 ]; then BATCH=${BATCH:-4}; else BATCH=${BATCH:-8}; fi

SZTAG=""; [ "$IMAGE_SIZE" -ne 448 ] && SZTAG="_sz${IMAGE_SIZE}"
OUT_DIR="training/points/outputs/${SPLIT}_ft_lr${LR}${SZTAG}"

echo "=== TreeFormer finetune: split=$SPLIT lr=$LR image_size=$IMAGE_SIZE batch=$BATCH epochs=30 ==="
echo "Root dir: $ROOT_DIR  Output dir: $OUT_DIR"

uv run --group treeformer python training/points/train.py \
  --split-scheme "$SPLIT" \
  --lr "$LR" \
  --image-size "$IMAGE_SIZE" \
  --max-epochs 30 \
  --early-stop-patience 10 \
  --batch-size "$BATCH" \
  --num-workers 8 \
  --output-dir "$OUT_DIR" \
  --comet

echo "=== Done. Results in $OUT_DIR ==="
