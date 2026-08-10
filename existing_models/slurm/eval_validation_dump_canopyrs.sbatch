#!/bin/bash
#SBATCH --job-name=mt_valdump_canopyrs
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# Same run as eval_validation_existing_canopyrs.sbatch (score 0.30, Allen+Frey validation)
# but with --save-predictions, so AP50/AP40 and the complete-tile subsets can be re-scored
# offline by scripts/rescore_validation_ap.py without another inference pass.
set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
OUT_BASE=$REPO/existing_models/canopyrs/outputs/validation
DUMPS=$REPO/outputs/validation_preds

module load cuda/12.8.1 gdal/3.7.0
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH="$CUDA_HOME/bin:$PATH"

cd "$REPO/existing_models/canopyrs"
source .venv/bin/activate

HF_TOKEN_FILE=$REPO/.hf_token
if [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
fi

echo "=== CanopyRS validation eval + prediction dump (score 0.30) ==="

python eval_boxes.py \
  --root-dir "$ROOT_DIR" --split-scheme out-of-distribution --eval-split validation \
  --device cuda --batch-size 4 --score-threshold 0.30 \
  --output-dir "$OUT_BASE" --viz-dir "$OUT_BASE/viz_boxes" \
  --save-predictions "$DUMPS/canopyrs_boxes_validation.pkl"

python eval_polygons.py \
  --root-dir "$ROOT_DIR" --split-scheme out-of-distribution --eval-split validation \
  --device cuda --batch-size 2 --score-threshold 0.30 \
  --output-dir "$OUT_BASE" --viz-dir "$OUT_BASE/viz_polygons" \
  --save-predictions "$DUMPS/canopyrs_polygons_validation.pkl"

echo "=== done. Dumps in $DUMPS ==="
