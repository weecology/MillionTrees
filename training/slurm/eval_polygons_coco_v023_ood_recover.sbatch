#!/bin/bash
#SBATCH --job-name=mt_polygons_coco_v023_ood_recover
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

# RECOVERY of the out-of-distribution COCO control arm of the polygon Table 6 ablation.
#
# Job 39185675_1 trained the full 20 epochs (10h17m) and then died at the very last step:
# save_eval_visualizations() hit OSError errno 122 "Disk quota exceeded" while writing an
# overlay PNG, taking the whole process down before results_out-of-distribution.txt was
# written. The MODEL SURVIVED — ModelCheckpoint wrote
#   polygon_coco_v023_cleaninit/out-of-distribution/logs/checkpoints/
#     polygons-epoch=19-val_loss=1.4872.ckpt   (366 MB, 2026-08-11 12:32)
# nine hours before the crash. So this scores that checkpoint instead of retraining, at
# roughly 0.5 GPU-hours rather than 10.
#
# NO THRESHOLD OVERRIDES. train.py's post-fit eval passes neither score_threshold nor
# eval_score_threshold, and eval.py only overrides when the flags are given, so running it
# bare reproduces train.py's scoring path exactly. Adding either flag here would make the
# OOD row incomparable to the within-distribution control (39185675_0) it pairs with.
#
# Image size, batch size and data scope are copied from
# train_polygons_coco_v023_cleaninit.sbatch for the same reason.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

SPLIT=out-of-distribution
OUT_DIR="training/weak_supervision/outputs/polygon_coco_v023_cleaninit/$SPLIT"
CKPT="$OUT_DIR/logs/checkpoints/polygons-epoch=19-val_loss=1.4872.ckpt"

if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  echo "If it is gone, the OOD control needs a full retrain via" >&2
  echo "train_polygons_coco_v023_cleaninit.sbatch --array=1." >&2
  exit 1
fi

srun uv run python training/polygons/eval.py \
  --checkpoint "$CKPT" \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme "$SPLIT" \
  --eval-split test \
  --batch-size 16 \
  --num-workers 16 \
  --image-size 448 \
  --init-mode coco \
  --data-scope subset \
  --output-dir "$OUT_DIR"
