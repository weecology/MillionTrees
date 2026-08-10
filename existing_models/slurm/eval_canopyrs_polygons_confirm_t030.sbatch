#!/bin/bash
# Full-test-set confirmation of the polygon score_threshold=0.30 operating point
# identified by sweep_canopyrs_pr_curve_polygons.sbatch on a 5-imgs/source
# subsample (notes/canopyrs_threshold_sweep.md: F1 0.742 OOD / 0.764 within-dist,
# vs. 0.649/0.659 at the standardized 0.1). This is the plain (non-sweep)
# single-threshold eval_polygons.py invocation -- same streaming-eval code path
# as the current leaderboard row -- just at the new threshold, so the result is
# directly comparable to the other (full-test-set) rows in docs/leaderboard.md.
#SBATCH --job-name=mt_eval_canopyrs_poly_t030
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_BASE=$REPO/existing_models/canopyrs/outputs/$SPLIT/confirm_t030

module load cuda/12.8.1 gdal/3.7.0
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH="$CUDA_HOME/bin:$PATH"

cd "$REPO/existing_models/canopyrs"
source .venv/bin/activate

HF_TOKEN_FILE=$REPO/.hf_token
if [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
fi

echo "=== CanopyRS polygon full-test-set confirm: split=$SPLIT, score_threshold=0.30 ==="

python eval_polygons.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --device cuda \
  --batch-size 2 \
  --score-threshold 0.30 \
  --output-dir "$OUT_BASE"

echo "=== Done -> $OUT_BASE/results_polygons_$SPLIT.{txt,json} ==="
