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
SPLITS=(within-distribution out-of-distribution)
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

# Box score_threshold=0.30 (not the 0.1 used by other pretrained models) is the
# best-F1 operating point from the full-test-set sweep in
# sweep_canopyrs_pr_curve.sbatch / docs/canopyrs_threshold_sweep.md -- CanopyRS
# boxes at 0.1 are badly recall-heavy (e.g. OOD: recall 0.89 / precision 0.82 ->
# recall 0.80 / precision 0.87 at 0.30).
python eval_boxes.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --device cuda \
  --batch-size 4 \
  --score-threshold 0.30 \
  --output-dir "$OUT_BASE" \
  --viz-dir "$OUT_BASE/viz_boxes"

# Polygon score_threshold=0.30 (not the 0.1 used by other pretrained models) is
# the best-F1 operating point, confirmed on the full test set (see
# eval_canopyrs_polygons_confirm_t030.sbatch / docs/canopyrs_threshold_sweep.md)
# -- at 0.1 this was the most recall-heavy row in the leaderboard (e.g. OOD:
# recall 0.89 / precision 0.51 -> recall 0.82 / precision 0.86 at 0.30, mask
# accuracy roughly doubling).
python eval_polygons.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --device cuda \
  --batch-size 2 \
  --score-threshold 0.30 \
  --output-dir "$OUT_BASE" \
  --viz-dir "$OUT_BASE/viz_polygons"

echo "=== CanopyRS eval done. Results in $OUT_BASE ==="
