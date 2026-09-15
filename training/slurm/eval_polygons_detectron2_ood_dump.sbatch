#!/bin/bash
#SBATCH --job-name=mt_eval_poly_d2_ood_dump
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

# Re-runs the leaderboard Detectron2 Mask R-CNN polygon eval (out-of-distribution
# split, model_final.pth, SCORE_THRESH_TEST=0.15) but with --save-predictions so the
# eval-score-threshold vs. recall/precision tradeoff can be swept offline afterwards
# with scripts/sweep_score_threshold.py, instead of re-running inference per threshold.
set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLIT=out-of-distribution
OUT_DIR="$REPO/training/polygons/outputs/detectron2/$SPLIT"
DUMP_DIR="$REPO/outputs/threshold_sweep_preds"
mkdir -p "$DUMP_DIR"

if [ ! -f "$OUT_DIR/model_final.pth" ]; then
  echo "ERROR: missing $OUT_DIR/model_final.pth" >&2; exit 1
fi

module load cuda/12.8.1 gdal/3.7.0
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO/existing_models/canopyrs"
source .venv/bin/activate

echo "=== Detectron2 polygon eval + prediction dump: split=$SPLIT ==="
echo "Weights: $OUT_DIR/model_final.pth"

python "$REPO/training/polygons/train_detectron2.py" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --eval-only \
  --eval-split test \
  --batch-size 8 \
  --image-size 448 \
  --num-workers 8 \
  --output-dir "$OUT_DIR" \
  --save-predictions "$DUMP_DIR/detectron2_polygons_${SPLIT}_test.pkl"

echo "=== done: $DUMP_DIR/detectron2_polygons_${SPLIT}_test.pkl ==="
