#!/bin/bash
#SBATCH --job-name=mt_eval_canopyrs
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
# eval_polygons.py now streams metrics batch-by-batch (TreePolygonsStreamingEvalState)
# instead of accumulating the full test set of dense 1024x1024 masks, so peak RAM is one
# batch rather than ~270 GiB. 150G is plenty; keep headroom for the box stage + loaders.
#SBATCH --mem=150G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLITS=(random zeroshot)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_BASE=$REPO/existing_models/canopyrs/outputs/$SPLIT

# CanopyRS (detrex + Detectron2 + SAM3) lives in a uv venv built by build_env.sh.
# detectron2/detrex CUDA ops are a fat binary for L4 (sm_89) + B200 (sm_100); load the
# matching CUDA + GDAL modules.
module load cuda/12.8.1 gdal/3.7.0
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH="$CUDA_HOME/bin:$PATH"

cd "$REPO/existing_models/canopyrs"
source .venv/bin/activate

# SAM3 (facebook/sam3) is gated; authenticate from the (untracked) token file in
# the main checkout.
HF_TOKEN_FILE=/blue/ewhite/b.weinstein/src/MillionTrees/.hf_token
if [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
fi

echo "=== CanopyRS eval: split=$SPLIT ==="

python eval_boxes.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --device cuda \
  --batch-size 4 \
  --score-threshold 0.1 \
  --output-dir "$OUT_BASE" \
  --viz-dir "$OUT_BASE/viz_boxes"

python eval_polygons.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --device cuda \
  --batch-size 2 \
  --score-threshold 0.1 \
  --output-dir "$OUT_BASE" \
  --viz-dir "$OUT_BASE/viz_polygons"

echo "=== CanopyRS eval done. Results in $OUT_BASE ==="
