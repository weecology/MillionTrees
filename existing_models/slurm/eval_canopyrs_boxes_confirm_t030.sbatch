#!/bin/bash
# Full-test-set, non-sweep box eval at the new score_threshold=0.30 (see
# docs/canopyrs_threshold_sweep.md). The threshold sweep only captures
# recall/precision/ap50/f1 (milliontrees.common.eval_sweep); it does not report
# counting_mae or detection_accuracy, which the manuscript tables also need and
# which come from the plain eval_boxes.py text/json output. This also refreshes
# existing_models/canopyrs/outputs/$SPLIT/results_boxes_*.json, which still held
# stale score_threshold=0.1 numbers even after docs/leaderboard.md was updated,
# and which scripts/make_benchmark_table.py reads for the auto-generated
# "Benchmark Results" appendix table.
#SBATCH --job-name=mt_eval_canopyrs_box_t030
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_BASE=$REPO/existing_models/canopyrs/outputs/$SPLIT

module load cuda/12.8.1 gdal/3.7.0
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH="$CUDA_HOME/bin:$PATH"

cd "$REPO/existing_models/canopyrs"
source .venv/bin/activate

echo "=== CanopyRS box full-test-set confirm: split=$SPLIT, score_threshold=0.30 ==="

python eval_boxes.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --device cuda \
  --batch-size 4 \
  --score-threshold 0.30 \
  --output-dir "$OUT_BASE" \
  --viz-dir "$OUT_BASE/viz_boxes"

echo "=== Done -> $OUT_BASE/results_boxes_$SPLIT.{txt,json} ==="
