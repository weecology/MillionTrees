#!/bin/bash
#SBATCH --job-name=mt_eval_canopyrs
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLITS=(random zeroshot)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_BASE=$REPO/existing_models/canopyrs/outputs/$SPLIT

# CanopyRS (detrex + Detectron2 + SAM3) lives in its own conda env; see README.md.
CANOPYRS_ENV=${CANOPYRS_ENV:-canopyrs}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CANOPYRS_ENV"

cd $REPO/existing_models/canopyrs

echo "=== CanopyRS eval: split=$SPLIT ==="

python eval_boxes.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --device cuda \
  --batch-size 4 \
  --output-dir "$OUT_BASE" \
  --viz-dir "$OUT_BASE/viz_boxes"

python eval_polygons.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --device cuda \
  --batch-size 2 \
  --output-dir "$OUT_BASE" \
  --viz-dir "$OUT_BASE/viz_polygons"

echo "=== CanopyRS eval done. Results in $OUT_BASE ==="
