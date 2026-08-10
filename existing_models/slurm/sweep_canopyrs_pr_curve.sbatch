#!/bin/bash
# Score-threshold sweep for the manuscript PR-curve figure: CanopyRS DINO (boxes,
# full test set) + DINO/SAM3 (polygons, 10 imgs/source to bound mask memory --
# the full test set is ~270 GiB of dense 1024x1024 masks in sweep mode, see
# eval_polygons.py), on the two splits reported in the manuscript leaderboard
# (within-distribution, out-of-distribution). --score-threshold 0.01 drives the
# model's own keep filter near zero so the full score range reaches the sweep;
# each array task writes to its own split subdir to avoid concurrent-append races
# on threshold_sweep.csv.
#SBATCH --job-name=mt_sweep_canopyrs_pr
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=250G
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

echo "=== CanopyRS PR-curve sweep: split=$SPLIT ==="

python eval_boxes.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --device cuda \
  --batch-size 4 \
  --score-threshold 0.01 \
  --sweep \
  --output-dir "$OUT_BASE"

python eval_polygons.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --device cuda \
  --batch-size 2 \
  --score-threshold 0.01 \
  --per-source 10 \
  --sweep \
  --output-dir "$OUT_BASE"

echo "=== CanopyRS PR-curve sweep done -> $OUT_BASE/threshold_sweep.csv ==="
