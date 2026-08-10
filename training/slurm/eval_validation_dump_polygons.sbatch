#!/bin/bash
#SBATCH --job-name=mt_valdump_polygons_ft
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

# Fine-tuned Detectron2 Mask R-CNN (the leaderboard polygon model) on the validation split
# with --save-predictions, for the AP50/AP40 x complete-tile re-scoring. Mirrors
# eval_validation_polygons.sbatch exactly (SCORE_THRESH_TEST=0.15, image_size 448).
set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLIT=out-of-distribution
OUT_DIR="$REPO/training/polygons/outputs/detectron2/$SPLIT"
DUMPS=$REPO/outputs/validation_preds

if [ ! -f "$OUT_DIR/model_final.pth" ]; then
  echo "ERROR: missing $OUT_DIR/model_final.pth" >&2; exit 1
fi

module load cuda/12.8.1 gdal/3.7.0
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO/existing_models/canopyrs"
source .venv/bin/activate

echo "=== Detectron2 polygon validation eval + prediction dump: split=$SPLIT ==="

python "$REPO/training/polygons/train_detectron2.py" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --eval-only \
  --eval-split validation \
  --batch-size 8 \
  --image-size 448 \
  --num-workers 8 \
  --output-dir "$OUT_DIR" \
  --save-predictions "$DUMPS/maskrcnn_finetuned_polygons_validation.pkl"

echo "=== done. Dump in $DUMPS ==="
