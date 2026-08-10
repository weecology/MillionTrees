#!/bin/bash
#SBATCH --job-name=mt_poly_d2_smoke
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

# End-to-end smoke test of the native Detectron2 Mask R-CNN polygon driver on the
# MiniTreePolygons subset (tiny, runs in minutes). Validates the data bridge,
# DefaultTrainer loop, predictor, and MillionTrees streaming eval before the full
# two-split array (train_polygons_detectron2.sbatch). NOT a leaderboard number.
set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}

# Detectron2 (CanopyRS venv) CUDA ops are a fat binary for L4 (sm_89) + B200 (sm_100).
module load cuda/12.8.1 gdal/3.7.0
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH="$CUDA_HOME/bin:$PATH"

cd "$REPO/existing_models/canopyrs"
source .venv/bin/activate

python "$REPO/training/polygons/train_detectron2.py" \
  --root-dir "$ROOT_DIR" \
  --split-scheme within-distribution \
  --mini \
  --batch-size 4 \
  --max-epochs 2 \
  --lr 0.01 \
  --num-workers 8 \
  --output-dir "$REPO/training/polygons/outputs/detectron2_smoke"

echo "=== Detectron2 polygon smoke done ==="
