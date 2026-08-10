#!/bin/bash
# TreeFormer pretrained baseline, random split, grid over the two recall levers
# in density_to_points:
#   - score_thresh: relative peak threshold (lower => more low-conf points)
#   - score_integration_radius: peak_local_max min_distance in density-map px
#     (downsample_ratio=4, so 1 unit ~= 4 image px); lower => closer detections,
#     more recall in dense canopy.
# Baseline is score_thresh=0.1, radius=5. NOTE: these MUST be set on the
# submodule (model.model), not model.config, or they are silently ignored.
#SBATCH --job-name=mt_tf_lowscore
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-5
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLIT=${SPLIT:-random}

# (score_thresh, score_integration_radius) per array index
SCORES=(0.05 0.01 0.10 0.10 0.05 0.10)
RADII=(  5    5    3    2    3    1  )
ST=${SCORES[$SLURM_ARRAY_TASK_ID]}
R=${RADII[$SLURM_ARRAY_TASK_ID]}
OUT_BASE=$REPO/existing_models/treeformer/outputs/${SPLIT}_st${ST}_r${R}

cd $REPO/existing_models/treeformer

echo "=== TreeFormer pretrained: split=$SPLIT score_thresh=$ST radius=$R ==="

uv run python eval_points.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --score-thresh "$ST" \
  --score-integration-radius "$R" \
  --batch-size 8 \
  --num-workers 4 \
  --output-dir "$OUT_BASE"

echo "=== Done. Results in $OUT_BASE ==="
