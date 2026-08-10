#!/bin/bash
#SBATCH --job-name=mt_eval_val_polygons
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

# Manuscript-final validation: the native Detectron2 Mask R-CNN that IS the current
# leaderboard polygon model (training/polygons/outputs/detectron2/<split>/model_final.pth,
# AP50 0.347 OOD), evaluated on the held-out Allen et al. 2025 + Frey et al. 2026 TLS set.
# Matches the leaderboard eval exactly: SCORE_THRESH_TEST=0.15 (set in train_detectron2.py),
# image_size 448, MIN_SIZE_TEST 800 / MAX_SIZE 1333 (argparse defaults). Runs in the
# CanopyRS from-source Detectron2 venv, same as train_polygons_detectron2.sbatch.
set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLIT=out-of-distribution
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

echo "=== Detectron2 polygon eval on validation (Allen+Frey): split=$SPLIT ==="
echo "Weights: $OUT_DIR/model_final.pth"

# --eval-only re-uses cfg.MODEL.WEIGHTS = <output-dir>/model_final.pth; --eval-split
# validation routes get_subset to the held-out set; results write to
# results_<split>_validation.{txt,json} (test outputs untouched). Version -> latest (0.21).
python "$REPO/training/polygons/train_detectron2.py" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --eval-only \
  --eval-split validation \
  --batch-size 8 \
  --image-size 448 \
  --num-workers 8 \
  --output-dir "$OUT_DIR"

echo "=== done: $OUT_DIR/results_${SPLIT}_validation.txt ==="
