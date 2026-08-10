#!/bin/bash
#SBATCH --job-name=mt_poly_detectron2
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

# Native Detectron2 Mask R-CNN on MillionTrees TreePolygons -- the
# implementation-control twin of training/polygons/train.py (DeepForest stack).
# Same data, same MillionTrees metric, same resize eval scale; only the Mask R-CNN
# implementation differs. Comparing the two polygon scores tells us whether the
# low DeepForest polygon numbers are a data problem (both stacks agree) or a
# DeepForest implementation problem (Detectron2 does better).
#
# Runs under the CanopyRS uv venv (the from-source Detectron2 build); array index
# selects the split (0 within-distribution, 1 out-of-distribution).
set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="$REPO/training/polygons/outputs/detectron2/$SPLIT"
mkdir -p "$OUT_DIR"

# Detectron2 (CanopyRS venv) CUDA ops are a fat binary for L4 (sm_89) + B200 (sm_100).
module load cuda/12.8.1 gdal/3.7.0
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO/existing_models/canopyrs"
source .venv/bin/activate

echo "=== Detectron2 polygon train+eval: split=$SPLIT ==="

python "$REPO/training/polygons/train_detectron2.py" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --batch-size 8 \
  --max-epochs 50 \
  --lr 0.01 \
  --num-workers 16 \
  --image-size 448 \
  --output-dir "$OUT_DIR" \
  --comet \
  --comet-name "polygons-${SPLIT}-coco-lr0.01-detectron2"

echo "=== Detectron2 polygon train+eval done: $OUT_DIR ==="
