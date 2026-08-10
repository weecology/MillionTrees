#!/bin/bash
# Polygon-only resubmit of sweep_canopyrs_pr_curve.sbatch: the box stage already
# completed and its results are recorded (notes/canopyrs_threshold_sweep.md,
# docs/leaderboard.md). The polygon stage hit the prior job's 8h limit after
# finishing inference on the 10-imgs/source subset but completing only 1 of 12
# threshold re-scores -- mask-based re-evaluation (AP50/mask-aware
# precision/mask-accuracy) is far more expensive per threshold than for boxes.
# This run cuts cost on both axes: 5 imgs/source (half the prior subsample) and
# a coarser 7-point threshold grid (vs. 12), with a much longer time budget.
# Appends TreePolygons rows to the existing per-split threshold_sweep.csv
# (already holds the completed TreeBoxes rows).
#SBATCH --job-name=mt_sweep_canopyrs_pr_poly
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_BASE=$REPO/existing_models/canopyrs/outputs/sweep/$SPLIT

module load cuda/12.8.1 gdal/3.7.0
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH="$CUDA_HOME/bin:$PATH"

cd "$REPO/existing_models/canopyrs"
source .venv/bin/activate

HF_TOKEN_FILE=$REPO/.hf_token
if [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
fi

echo "=== CanopyRS PR-curve sweep (polygons only, resubmit): split=$SPLIT ==="

python eval_polygons.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --device cuda \
  --batch-size 2 \
  --score-threshold 0.01 \
  --per-source 5 \
  --sweep \
  --sweep-thresholds "0.1,0.2,0.3,0.4,0.5,0.6,0.7" \
  --output-dir "$OUT_BASE"

echo "=== CanopyRS polygon PR-curve sweep done -> $OUT_BASE/threshold_sweep.csv ==="
