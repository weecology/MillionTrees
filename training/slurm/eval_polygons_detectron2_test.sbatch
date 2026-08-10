#!/bin/bash
#SBATCH --job-name=mt_eval_poly_d2_test
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

# Test-split eval of the Detectron2 Mask R-CNN that IS the leaderboard polygon model
# (training/polygons/outputs/detectron2/<split>/model_final.pth). Same config as the
# run that produced the current rows -- SCORE_THRESH_TEST=0.15 (set in
# train_detectron2.py), image_size 448 -- so the only change is the dataset version
# (v0.22) and AP now scoring at IoU 0.4 (AP40) instead of 0.5. Mirrors
# eval_validation_polygons.sbatch with --eval-split test over both splits.
set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="$REPO/training/polygons/outputs/detectron2/$SPLIT"

if [ ! -f "$OUT_DIR/model_final.pth" ]; then
  echo "ERROR: missing $OUT_DIR/model_final.pth" >&2; exit 1
fi

# Detectron2 (CanopyRS venv) CUDA ops are a fat binary for L4 (sm_89) + B200 (sm_100).
module load cuda/12.8.1 gdal/3.7.0
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO/existing_models/canopyrs"
source .venv/bin/activate

echo "=== Detectron2 polygon eval on test: split=$SPLIT ==="
echo "Weights: $OUT_DIR/model_final.pth"

python "$REPO/training/polygons/train_detectron2.py" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --eval-only \
  --eval-split test \
  --batch-size 8 \
  --image-size 448 \
  --num-workers 8 \
  --output-dir "$OUT_DIR"

echo "=== done: $OUT_DIR/results_${SPLIT}.txt ==="
